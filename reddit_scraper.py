import os
import json
import time
import random
import requests
import pandas as pd

## Setup

HEADERS = {"User-Agent": "MADS-FinalProject/1.0 achia@sandiego.edu"}

def sleep_politely(sleep_range = (1.0, 2.0)):
    time.sleep(random.uniform(*sleep_range))

## API Collect Posts

def fetch_reddit_posts(subreddit = 'politics', limit = 100, raw_api_dir = None, verbose = True):
    """
    Fetch posts from a subreddit via Reddit's search API with pagination support.

    Supports fetching more than 100 posts by paginating through results.
    """
    fp = os.path.join(raw_api_dir, f'{subreddit}_posts.json')
    if os.path.exists(fp):
        with open(fp, 'r', encoding = 'utf-8') as f:
            return json.load(f)

    # About the last 6 months+
    url = f'https://www.reddit.com/r/{subreddit}/search.json'

    all_children = []
    after = None
    remaining = limit
    page = 1

    while remaining > 0:
        batch_size = min(100, remaining)
        params = {
            'q': 'politics',
            'restrict_sr': 1,
            'sort': 'new',
            't': 'year',
            'limit': batch_size
        }
        if after:
            params['after'] = after

        r = requests.get(url, headers = HEADERS, params = params, timeout = 60)
        r.raise_for_status()
        data = r.json()

        children = data.get('data', {}).get('children', [])
        if not children:
            break

        all_children.extend(children)
        remaining -= len(children)

        after = data.get('data', {}).get('after')
        if not after:
            break

        if verbose:
            print(f'  Page {page}: fetched {len(children)} posts, total: {len(all_children)}')
        page += 1
        sleep_politely()

    # Reconstruct the data structure to match original format
    combined_data = {
        'data': {
            'children': all_children,
            'after': after
        }
    }

    with open(fp, 'w', encoding = 'utf-8') as f:
        json.dump(combined_data, f)

    sleep_politely()
    return combined_data

## Fetch Comment Threads

def get_comments_json(post_id, url, raw_html_dir = None):
    """Fetch the JSON comment tree for a given post."""
    fp = os.path.join(raw_html_dir, f'comments_{post_id}.json')
    if os.path.exists(fp):
        with open(fp, 'r', encoding = 'utf-8') as f:
            return json.load(f)

    r = requests.get(url + '.json', headers = HEADERS, timeout = 60)
    r.raise_for_status()
    data = r.json()

    with open(fp, 'w', encoding = 'utf-8') as f:
        json.dump(data, f)

    sleep_politely(sleep_range = (2.0, 3.0))
    return data

## Parse Comments

def extract_comments(comment_tree, post_id):
    """Recursively extract comments from a Reddit comment tree."""
    rows = []

    def walk(comments):
        for c in comments:
            if c['kind'] != 't1':
                continue
            d = c['data']

            rows.append({
                'post_id': post_id,
                'comment_id': d['id'],
                'author': d.get('author'),
                'created_utc': d.get('created_utc'),
                'score': d.get('score'),
                'comment_text': d.get('body')
            })

            if d.get('replies'):
                walk(d['replies']['data']['children'])

    walk(comment_tree)
    return rows

## Flatten post metadata

def flatten_posts(raw_posts, min_comments = 10):
    """Flatten raw post data into a list of dictionaries."""
    posts = []
    for child in raw_posts['data']['children']:
        d = child['data']
        if d.get('num_comments', 0) < min_comments:
            continue

        posts.append({
            'post_id': d['id'],
            'title': d['title'],
            'author': d['author'],
            'created_utc': d['created_utc'],
            'score': d['score'],
            'num_comments': d['num_comments'],
            'permalink': d['permalink'],
            'url': 'https://old.reddit.com' + d['permalink']
        })
    return posts

## Unifying Function

def scrape_reddit(
    subreddit = 'politics',
    limit = 100,
    min_comments = 10,
    data_dir = 'Downloads/reddit',
    verbose = True
):
    """
    Scrape Reddit posts and comments from a subreddit.

    Parameters
    ----------
    subreddit : str
        The subreddit to scrape (default: 'politics')
    limit : int
        Maximum number of posts to fetch (default: 100)
    min_comments : int
        Minimum number of comments a post must have to be included (default: 10)
    data_dir : str
        Base directory for storing raw data (default: 'Downloads/reddit')
    verbose : bool
        Print progress messages (default: True)

    Returns
    -------
    tuple (pd.DataFrame, list)
        - Merged dataframe with posts and comments
        - List of dicts with skipped post info: {'index': int, 'post_id': str, 'error': str}
    """
    # Setup directories
    raw_api_dir = os.path.join(data_dir, 'raw_api')
    raw_html_dir = os.path.join(data_dir, 'raw_html')

    for d in [data_dir, raw_api_dir, raw_html_dir]:
        os.makedirs(d, exist_ok = True)

    # Fetch posts
    if verbose:
        print(f'Fetching posts from r/{subreddit}...')
    raw_posts = fetch_reddit_posts(subreddit, limit, raw_api_dir, verbose)

    # Flatten post metadata
    posts = flatten_posts(raw_posts, min_comments)
    posts_df = pd.DataFrame(posts)
    if verbose:
        print(f'Posts collected: {len(posts_df)}')

    if len(posts_df) == 0:
        if verbose:
            print('No posts found matching criteria.')
        return pd.DataFrame(), []

    # Loop over posts to collect comments
    all_comments = []
    skipped = []

    for idx, row in posts_df.iterrows():
        if verbose and idx % 5 == 0:
            print(f'{idx}/{len(posts_df)}')

        try:
            data = get_comments_json(row['post_id'], row['url'], raw_html_dir)
            comments = data[1]['data']['children']
            all_comments.extend(extract_comments(comments, row['post_id']))
        except Exception as e:
            skipped.append({
                'index': idx,
                'post_id': row['post_id'],
                'error': str(e)
            })
            continue

    if verbose and skipped:
        print(f'Skipped {len(skipped)} posts due to errors')

    comments_df = pd.DataFrame(all_comments)
    if verbose:
        print(f'Total comments collected: {len(comments_df)}')

    if len(comments_df) == 0:
        if verbose:
            print('No comments found.')
        return posts_df, skipped

    # Combine datasets
    merged_df = comments_df.merge(
        posts_df,
        on = 'post_id',
        how = 'left'
    )

    merged_df['post_time'] = pd.to_datetime(merged_df['created_utc_y'], unit = 's')
    merged_df['comment_time'] = pd.to_datetime(merged_df['created_utc_x'], unit = 's')

    if verbose:
        print(f'Final rows: {len(merged_df)}')

    return merged_df, skipped


def retry_skipped_posts(
    posts_df,
    skipped,
    data_dir = 'Downloads/reddit',
    verbose = True
):
    """
    Retry fetching comments for previously skipped posts.

    Parameters
    ----------
    posts_df : pd.DataFrame
        The posts dataframe from the original scrape
    skipped : list
        List of skipped post dicts from scrape_reddit
    data_dir : str
        Base directory for storing raw data
    verbose : bool
        Print progress messages

    Returns
    -------
    tuple (pd.DataFrame, list)
        - DataFrame of newly collected comments (not merged)
        - List of posts that still failed
    """
    raw_html_dir = os.path.join(data_dir, 'raw_html')
    os.makedirs(raw_html_dir, exist_ok = True)

    all_comments = []
    still_skipped = []

    for i, skip_info in enumerate(skipped):
        idx = skip_info['index']
        post_id = skip_info['post_id']
        row = posts_df[posts_df['post_id'] == post_id].iloc[0]

        if verbose:
            print(f'Retrying {i + 1}/{len(skipped)}: {post_id}')

        try:
            data = get_comments_json(post_id, row['url'], raw_html_dir)
            comments = data[1]['data']['children']
            all_comments.extend(extract_comments(comments, post_id))
        except Exception as e:
            still_skipped.append({
                'index': idx,
                'post_id': post_id,
                'error': str(e)
            })
            continue

    if verbose:
        print(f'Recovered {len(skipped) - len(still_skipped)} posts')
        if still_skipped:
            print(f'Still skipped: {len(still_skipped)} posts')

    return pd.DataFrame(all_comments), still_skipped
