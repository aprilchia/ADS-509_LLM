import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
from datetime import datetime
from sleepy import sleep_politely

# --- HELPER FUNCTIONS ---

def get_soup_metafilter(url):
    """Fetches a URL and returns a BeautifulSoup object."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) DataScienceProject/1.0 tkirk@sandiego.edu'}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return BeautifulSoup(r.text, 'html.parser')

def parse_main_page_metafilter(soup):
    """Parses the tag/search feed page for post summaries."""
    posts_data = []
    skipped_posts = []
    
    post_summaries = soup.find_all('div', class_="copy post")
    
    for idx, post in enumerate(post_summaries):
        try:
            title_link = post.find('a')
            post_title = title_link.text
            external_link = title_link['href']
            
            # The link to the actual Metafilter thread is inside a span
            thread_link = post.find('span').find('a')['href']
            
            # Metadata section (OP and Date)
            byline = post.find('span', class_='smallcopy postbyline')
            op_user = byline.find('a').text
            op_profile = byline.find('a')['href']
            
            # Date Parsing
            raw_date = byline.find('a').next_sibling.get_text(strip=True)
            clean_date = raw_date.replace('on', "").replace('-', "").strip()
            date_obj = datetime.strptime(clean_date, "%b %d, %Y").strftime("%Y-%m-%d")
            
            # Comment Count
            comment_text = post.find_all('a', string=re.compile('comments'))[0].text
            num_comments = int(comment_text.split()[0])
            
            posts_data.append({
                'title': post_title,
                'external_link': external_link,
                'thread_link': thread_link,
                'op_user': op_user,
                'op_profile': op_profile,
                'date': date_obj,
                'comment_count': num_comments
            })
            
        except Exception as e:
            skipped_posts.append({'index': idx, 'title': post.find('a').text if post.find('a') else "Unknown", 'error': str(e)})
            continue
            
    return posts_data, skipped_posts

def parse_comments_metafilter(post_soup, thread_url):
    """Parses individual comments from a specific post thread."""
    comments_data = []
    comment_blocks = post_soup.find_all('div', class_='comments')
    
    for comment in comment_blocks:
        try:
            # Comment text is usually the first navigation element
            text = comment.get_text(strip=True)
            
            # User info
            user_link = comment.find('a', target='_self')
            username = user_link.text
            profile = user_link['href']
            
            # Time and Date
            # Usually found in the secondary links within the comment div
            footer_links = comment.find_all('a', target='_self')
            # Metafilter usually puts time/date in the second 'self' link
            time_part = footer_links[1].text if len(footer_links) > 1 else None
            
            date_part = "Unknown"
            if len(footer_links) > 1 and footer_links[1].next_sibling:
                raw_sibling = footer_links[1].next_sibling
                if 'on' in raw_sibling:
                    date_part = raw_sibling.split('on')[1].replace('[', "").replace(']', "").strip()

            # Favorites (not always present)
            favs = comment.find('a', title=True)
            favorites = favs.text if favs else "0"

            # Check for external links in the comment
            ext_link = None
            all_links = comment.find_all('a')
            for l in all_links:
                if not l.has_attr('target') or l['target'] != '_self':
                    ext_link = l['href']
                    break

            comments_data.append({
                'thread_link': thread_url,
                'username': username,
                'profile': profile,
                'comment_text': text,
                'timestamp': time_part,
                'date': date_part,
                'favorites': favorites,
                'comment_ext_link': ext_link
            })
        except Exception:
            continue # Skip individual messy comments
            
    return comments_data

# --- MAIN EXECUTION FUNCTIONS ---

def scrape_main_feed_metafilter(tag='politics', pages=1):
    """Loops through feed pages and returns a DataFrame of posts."""
    all_posts = []
    all_skipped = []
    
    for p in range(1, pages + 1):
        try:
            print(f"Scraping feed page {p}...")
            url = f"https://www.metafilter.com/tags/{tag}?page={p}"
            soup = get_soup_metafilter(url)
            
            posts, skipped = parse_main_page_metafilter(soup)
            all_posts.extend(posts)
            all_skipped.extend(skipped)
            
            time.sleep(random.uniform(1, 3))
        except Exception as e:
            print(f"Error on feed page {p}: {e}")
            
    if all_skipped:
        print(f"Skipped {len(all_skipped)} posts due to parsing errors.")
        
    return pd.DataFrame(all_posts)

def scrape_all_comments_metafilter(main_df):
    """Iterates through thread links in a DataFrame to scrape all comments."""
    all_comments = []
    
    for idx, row in main_df.iterrows():
        try:
            url = row['thread_link']
            print(f"Scraping thread: {row['title'][:30]}...")
            
            soup = get_soup_metafilter(url)
            comments = parse_comments_metafilter(soup, url)
            all_comments.extend(comments)
            
            sleep_politely()
        except Exception as e:
            print(f"Error scraping thread {url}: {e}")
            
    return pd.DataFrame(all_comments)

# --- WORKFLOW ---

if __name__ == "__main__":
    # 1. Scrape the main feed for politics
    df_main = scrape_main_feed_metafilter(tag='politics', pages=2)
    
    # 2. Scrape the comments for every post found
    df_comments = scrape_all_comments_metafilter(df_main)
    
    # 3. Merge them if you want a master sentiment dataset
    # We merge on 'thread_link' to associate comments with their parent post data
    final_df = pd.merge(df_comments, df_main, on='thread_link', how='left')
    
    print("\nScraping Complete!")
    print(f"Total Posts: {len(df_main)}")
    print(f"Total Comments: {len(df_comments)}")