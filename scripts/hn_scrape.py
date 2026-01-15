from datetime import datetime, timezone
import requests
import pandas as pd
from sleepy import sleep_politely
from bs4 import BeautifulSoup
from typing import Dict, Any, List

base_url = "http://hn.algolia.com/api/v1/search_by_date?"
HEADERS = {"User-Agent": "MADS-LLM-2025-student-Taylor/1.0 (tkirk@sandiego.edu)"}
root = "https://news.ycombinator.com/"

def define_parameters(page: int=0, tags: str='story', query: str='politics', per_page: int=50, comment_filter: str=">10", date_string: str="2025-06-01"):
    
    dt = datetime.strptime(date_string, "%Y-%m-%d")
    timestamp = int(dt.astimezone(timezone.utc).timestamp())

    return {
        "query": query, 
        "tags": tags, 
        "page": page, 
        "hitsPerPage": per_page, 
        "numericFilters": f"num_comments{comment_filter},created_at_i>{timestamp}"
        }

def hn_pull_posts(pages: int, params: Dict[str, Any]):
    hits = []
    for page in range(pages + 1):
        params['page'] = page
        r = requests.get(url=base_url, params=params, headers=HEADERS)
        r.raise_for_status()
        hits.extend(r.json()['hits'])
        sleep_politely()
    return hits

def make_posts_df(hits: List[Dict]):
    keep_cols = ['story_id', 'author', 'points', 'created_at', 'title', 'url']
    return pd.json_normalize(hits).loc[:, keep_cols].rename(columns={"title": "post_title", "url": "outside_url"})

def get_discussion_html(story_id):
    url = root + f"item?id={story_id}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    html = r.text
    sleep_politely(sleep_range = (30.0,31.0)) # Adjust sleep according to robots.txt
    return html

# Simple HTML -> comments parser
def parse_comments_from_html(html, story_id):
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    table_row_list = soup.find_all("tr", class_='athing comtr')
    for tr in table_row_list:
        
        comment_id = tr['id']
        user = tr.find("a", class_='hnuser').text # type: ignore
        time_text = tr.find("span", class_='age')['title'] # type: ignore
        comment_text = tr.find('div', class_='comment').get_text(strip=True) # type: ignore
        if comment_text:
            rows.append({
                'story_id': story_id,
                'comment_id': comment_id,
                'user': user,
                'time_text': time_text,
                'comment_text': comment_text,
            })

    return rows


def main(pages: int):

    params = define_parameters()
    hits = hn_pull_posts(pages=pages, params=params)
    df = make_posts_df(hits=hits)

    all_comments = []
    skipped = {}
    for idx, sid in enumerate(df['story_id'].tolist()):
        if idx%2==0:
            print(f"{idx}/{len(df['story_id'])}")
        html = get_discussion_html(sid)

        try:
            all_comments.extend(parse_comments_from_html(html, sid))
        except Exception as e:
            skipped[sid] = html
            print(f"Problem with sid {sid}")
            print(f"Error: {e}")

    comments_df = pd.DataFrame(all_comments)
    print('Total comments scraped:', len(comments_df))
    return df, comments_df