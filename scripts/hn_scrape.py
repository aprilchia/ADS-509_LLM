from datetime import datetime, timezone
import requests
import pandas as pd
from utils.sleepy import sleep_politely
from typing import Dict, Any, List

base_url = "http://hn.algolia.com/api/v1/search_by_date?"
HEADERS = {"User-Agent": "MADS-LLM-2025-student-Taylor/1.0 (tkirk@sandiego.edu)"}

def define_parameters(page: int=0, tags: str='story', query: str='politics', per_page: int=5, comment_filter: str=">10", date_string: str="2025-07-09"):
    """Build a parameter dict for the Algolia HN search_by_date API.

    Converts date_string to a Unix timestamp and combines it with a comment
    count filter so only stories after that date and above the comment threshold
    are returned.

    Args:
        page: Page number to fetch (0-indexed). Defaults to 0.
        tags: Algolia tag filter (e.g., 'story', 'comment'). Defaults to 'story'.
        query: Search keyword. Defaults to 'politics'.
        per_page: Results per page (hitsPerPage). Defaults to 5.
        comment_filter: Numeric filter string for num_comments (e.g., '>10').
            Defaults to '>10'.
        date_string: Earliest post date in 'YYYY-MM-DD' format. Defaults to
            '2025-07-09'.

    Returns:
        dict: Parameters ready to pass to requests.get() for the Algolia API.
    """
    
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
    """Paginate through the Algolia search_by_date feed and collect story hits.

    Iterates from page 0 through pages (inclusive), updating params['page'] on
    each iteration. Stops early if any page returns a non-2xx response.
    Calls sleep_politely() between pages to respect rate limits.

    Args:
        pages: Number of additional pages to fetch beyond page 0.
        params: Algolia API parameter dict (see define_parameters).

    Returns:
        list[dict]: All story hit dicts collected across pages.
    """
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
    """Convert a list of Algolia story hit dicts into a standardized DataFrame.

    Keeps story_id, author, points, created_at, title, and url, then renames
    title → post_title and url → outside_url to match the project schema.

    Args:
        hits: List of raw hit dicts returned by the Algolia API.

    Returns:
        pd.DataFrame: Columns: story_id, author, points, created_at,
            post_title, outside_url.
    """
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
    """Orchestrate the full Hacker News scraping pipeline.

    Builds API parameters, paginates through story results, converts hits to a
    DataFrame, then fetches all comments for those stories.

    Args:
        pages: Number of feed pages to retrieve (passed to hn_pull_posts).
        per_page: Stories per page (hitsPerPage in the Algolia query).

    Returns:
        tuple: (posts_df, comments_df) as pandas DataFrames.
    """
    params = define_parameters(per_page=per_page)
    hits = hn_pull_posts(pages=pages, params=params)
    posts_df = make_posts_df(hits=hits)
    comments_df = comment_scrape_api(posts_df)
    return posts_df, comments_df