import os
import json
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

## Setup

HEADERS = {'User-Agent': 'MADS-FinalProject/1.0 achia@sandiego.edu'}

def sleep_politely(sleep_range = (1.0, 2.0)):
    time.sleep(random.uniform(*sleep_range))

DATA_DIR = 'Downloads/reddit'
RAW_API_DIR = os.path.join(DATA_DIR, 'raw_api')
RAW_HTML_DIR = os.path.join(DATA_DIR, 'raw_html')

for d in [DATA_DIR, RAW_API_DIR, RAW_HTML_DIR]:
    os.makedirs(d, exist_ok = True)
  
## API Collect Posts

# Fetch posts
def fetch_reddit_posts(subreddit = 'politics', limit = 100):
    fp = os.path.join(RAW_API_DIR, f'{subreddit}_posts.json')
    if os.path.exists(fp):
        with open(fp, 'r', encoding = 'utf-8') as f:
            return json.load(f)

# About the last 6 months+
    url = f'https://www.reddit.com/r/{subreddit}/search.json'
    params = {
        'q': 'politics',
        'restrict_sr': 1,
        'sort': 'new',
        't': 'year',
        'limit': limit
    }

    r = requests.get(url, headers = HEADERS, params = params, timeout = 60)
    r.raise_for_status()
    data = r.json()

    with open(fp, 'w', encoding = 'utf-8') as f:
        json.dump(data, f)

    sleep_politely()
    return data

# Flatten post metadata
raw_posts = fetch_reddit_posts(limit = 150)

posts = []
for child in raw_posts['data']['children']:
    d = child['data']
    if d.get('num_comments', 0) < 10:
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

posts_df = pd.DataFrame(posts)
print('Posts collected:', len(posts_df))
posts_df.head()

## Fetch Comment Threads

def get_comments_json(post_id, url):
    fp = os.path.join(RAW_HTML_DIR, f'comments_{post_id}.json')
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

# Loop over posts
all_comments = []

for idx, row in posts_df.iterrows():
    if idx % 5 == 0:
        print(f'{idx}/{len(posts_df)}')

    try:
        data = get_comments_json(row['post_id'], row['url'])
        comments = data[1]['data']['children']
        all_comments.extend(extract_comments(comments, row['post_id']))
    except Exception:
        continue

comments_df = pd.DataFrame(all_comments)
print('Total comments collected:', len(comments_df))
comments_df.head()
## Combine Datasets
merged_df = comments_df.merge(
    posts_df,
    on = 'post_id',
    how = 'left'
)

merged_df['post_time'] = pd.to_datetime(merged_df['created_utc_y'], unit = 's')
merged_df['comment_time'] = pd.to_datetime(merged_df['created_utc_x'], unit = 's')

assert merged_df['comment_text'].notna().all()
print('Final rows:', len(merged_df))
merged_df.sample(5)
