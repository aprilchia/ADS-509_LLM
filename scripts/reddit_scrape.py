import os
import json
import requests
import pandas as pd
from utils.sleepy import sleep_politely
from datetime import datetime as dt

## Setup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

HEADERS = {"User-Agent": "MADS-FinalProject/1.0 achia@sandiego.edu"}

DATA_DIR = os.path.join(ROOT_DIR, 'Downloads', 'reddit')
RAW_API_DIR = os.path.join(DATA_DIR, 'raw_api')
RAW_HTML_DIR = os.path.join(DATA_DIR, 'raw_html')

## API Collect Posts

def fetch_reddit_posts(subreddit = 'politics', limit = 100, max_pages = 10):
    """Fetch posts using Reddit's automated pagination.
    limit: number of posts per page (max 100).
    max_pages: number of pages to fetch (e.g. 10 pages * 100 limit = 1000 posts max)."""
    fp = os.path.join(RAW_API_DIR, f"{subreddit}_posts.json")
    if os.path.exists(fp):
        with open(fp, 'r', encoding = 'utf-8') as f:
            return json.load(f)

    url = f"https://www.reddit.com/r/{subreddit}/new.json" # changing from 'search' to 'new' removes the hard cap at 250 and increases to 1000
    params = {
        "q": "politics",
        "restrict_sr": 1,
        # "sort": "new",
        "t": "year",
        "limit": limit
    }

    data_blocks = []

    for page in range(max_pages):
        r = requests.get(url, headers = HEADERS, params = params, timeout = 60)
        r.raise_for_status()
        data = r.json()
        data_blocks.append(data)

        after_id = data['data']['after'] # this then pulls from the next page since Reddit uses automated pagination (otherwise it caps at 100 per request)
        if after_id:
            params['after'] = after_id
        else:
            break # no more pages available

        sleep_politely()

    # Can further increase this by doing time slices, then checking for dups when merging or making the same request with different sort params

    with open(fp, "w", encoding = "utf-8") as f:
        json.dump(data_blocks, f)

    return data_blocks

## Fetch Comment Threads

def get_comments_json(post_id, url):
    """Fetch and cache the raw comment JSON tree for a single Reddit post.

    Returns the cached file immediately if it already exists, avoiding a
    redundant API call. Otherwise fetches '{url}.json' and writes the result
    to RAW_HTML_DIR. Calls sleep_politely() after each network request.

    Args:
        post_id: Reddit post ID string used to name the cache file.
        url: Full permalink URL to the post (without the .json extension).

    Returns:
        list: Raw Reddit API JSON response (a two-element list where index 1
            contains the comment tree).
    """
    fp = os.path.join(RAW_HTML_DIR, f"comments_{post_id}.json")
    if os.path.exists(fp):
        with open(fp, "r", encoding = "utf-8") as f:
            return json.load(f)

    r = requests.get(url + ".json", headers = HEADERS, timeout = 60)
    r.raise_for_status()
    data = r.json()

    with open(fp, "w", encoding = "utf-8") as f:
        json.dump(data, f)

    sleep_politely(sleep_range = (2.0, 3.0))
    return data

## Parse Comments

def extract_comments(comment_tree, post_id, thread_link):
    """Recursively walk a Reddit comment tree and extract comment rows.

    Processes only top-level 't1' (comment) nodes and descends into nested
    replies, collecting all comments regardless of depth.

    Args:
        comment_tree: List of Reddit API child dicts (from data['children']).
        post_id: Reddit post ID to attach to each extracted row.
        thread_link: Full permalink URL to include in each row.

    Returns:
        list[dict]: Each dict contains post_id, comment_id, thread_link,
            author, created_utc (Unix timestamp), score, comment_text, and
            source='reddit'.
    """
    rows = []

    def walk(comments):
        for c in comments:
            if c["kind"] != "t1":
                continue
            d = c["data"]

            rows.append({
                "post_id": post_id,
                "comment_id": d["id"],
                "thread_link": thread_link,
                "author": d.get("author"),
                "created_utc": d.get("created_utc"),
                "score": d.get("score"),
                "comment_text": d.get("body"),
                "source": "reddit"
            })

            if d.get("replies"):
                walk(d["replies"]["data"]["children"])

    walk(comment_tree)
    return rows

## Collect and Build DataFrames

def collect_reddit_data(subreddit = 'politics', limit = 100, min_comments = 10):
    """Unifies fetch, comment retrieval, and parsing.
    Returns reddit_main and reddit_comments DataFrames ready for EDA."""

    # Ensure output directories exist
    for d in [DATA_DIR, RAW_API_DIR, RAW_HTML_DIR]:
        os.makedirs(d, exist_ok = True)

    # Fetch posts with pagination
    data_blocks = fetch_reddit_posts(subreddit = subreddit, limit = limit)

    # Flatten post metadata
    posts = []
    for block in data_blocks:
        for child in block["data"]["children"]:
            d = child["data"]
            if d.get("num_comments", 0) < min_comments:
                continue

            posts.append({
                "post_id": d["id"],
                "title": d["title"],
                "post_text": d.get("selftext", ""),
                "author": d["author"],
                "created_utc": d["created_utc"],
                "score": d["score"],
                "num_comments": d["num_comments"],
                "thread_link": "https://old.reddit.com" + d["permalink"],
                "url": "https://old.reddit.com" + d["permalink"],
                "source": "reddit"
            })

    posts_df = pd.DataFrame(posts)
    posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'], unit = 's')
    print("Posts collected:", len(posts_df))

    # Fetch and parse comments
    all_comments = []

    for idx, (_, row) in enumerate(posts_df.iterrows()):
        if idx % 10 == 0:
            print(f"Parsing {idx}/{len(posts_df)} posts")

        try:
            data = get_comments_json(row["post_id"], row["thread_link"])
            comments = data[1]["data"]["children"]
            all_comments.extend(
                extract_comments(comments, row["post_id"], row["thread_link"])
            )
        except Exception:
            continue

    comments_df = pd.DataFrame(all_comments)
    print("Total comments collected:", len(comments_df))

    ## EDA Adapter

    # Reddit → EDA main dataframe
    reddit_main = posts_df.copy()

    # Rename columns for EDA compatibility
    reddit_main = reddit_main.rename(columns = {
        "id": "post_id",
        "title": "post_title",
        "author": "post_author",
        "created_utc": "created_at",
    })

    # Keep only required columns
    reddit_main = reddit_main[
        ["post_id", "post_title", "post_author", "created_at"]
    ]

    # Add source column
    reddit_main['source'] = 'reddit'

    # Reddit → EDA comments dataframe
    reddit_comments = comments_df.copy()

    reddit_comments = reddit_comments.rename(columns = {
        "author": "username",
        "created_utc": "created_at"
    })

    # Convert to datetime
    reddit_comments["created_at"] = pd.to_datetime(
        reddit_comments["created_at"], unit = "s", errors = "coerce"
    )

    # Keep required columns
    reddit_comments = reddit_comments[["post_id", "username", "comment_text", "created_at"]]

    # Add source column
    reddit_comments['source'] = 'reddit'

    return reddit_main, reddit_comments
