from datetime import datetime, timezone
import requests
import pandas as pd
from utils.sleepy import sleep_politely
from typing import Dict, Any, List

base_url = "http://hn.algolia.com/api/v1/search_by_date?"
HEADERS = {"User-Agent": "MADS-LLM-2025-student-Taylor/1.0 (tkirk@sandiego.edu)"}

def define_parameters(page: int=0, tags: str='story', query: str='politics', per_page: int=5, comment_filter: str=">10", date_string: str="2025-07-09"):
    
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
        if not r.ok:
            print(f"HTTP {r.status_code} on posts page {page} — returning {len(hits)} hits collected so far.")
            break
        hits.extend(r.json()['hits'])
        sleep_politely()
    return hits

def make_posts_df(hits: List[Dict]):
    keep_cols = ['story_id', 'author', 'points', 'created_at', 'title', 'url']
    return pd.json_normalize(hits).loc[:, keep_cols].rename(columns={"title": "post_title", "url": "outside_url"})

def comment_scrape_api(posts_df: pd.DataFrame):
    """Scrape comments via the Algolia API for story_ids in an existing posts DataFrame."""
    story_ids = posts_df['story_id'].tolist()
    all_comments = []
    for idx, story_id in enumerate(story_ids):
        if idx % 5 == 0:
            print(f"{idx}/{len(story_ids)}")

        url = f"https://hn.algolia.com/api/v1/search?tags=comment,story_{story_id}&hitsPerPage=300"
        r = requests.get(url, headers=HEADERS)
        if not r.ok:
            print(f"HTTP {r.status_code} on story {story_id} — skipping.")
            continue

        hits = r.json()['hits']
        for hit in hits:
            all_comments.append({
                'story_id': hit['story_id'],
                'created_at': hit['created_at'],
                'story_title': hit['story_title'],
                'author': hit['author'],
                'comment_text': hit['_highlightResult']['comment_text']['value'],
            })

        sleep_politely(sleep_range=(1.0, 3.0))

    comments_df = pd.DataFrame(all_comments)
    print(f"Total comments scraped: {len(comments_df)}")
    return comments_df


def main(pages: int, per_page: int):
    params = define_parameters(per_page=per_page)
    hits = hn_pull_posts(pages=pages, params=params)
    posts_df = make_posts_df(hits=hits)
    comments_df = comment_scrape_api(posts_df)
    return posts_df, comments_df