import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
from datetime import datetime
from utils.sleepy import sleep_politely

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

    # Get post titles and thread links from h2.posttitle elements
    posts = soup.find_all("h2", class_="posttitle")
    post_summaries = soup.find_all('div', class_="copy post")

    for idx, post in enumerate(post_summaries):
        try:
            # Get title and thread_link from h2 elements
            post_title = posts[idx].get_text(separator=" ", strip=True)
            thread_link = posts[idx].find('a')['href']

            # Get external link from the post summary (first link)
            external_link = post.find('a')['href']
            
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
    pattern = re.compile(r"\.?\s*posted\sby.*") # remove the .posted by artifact that gets pulled with the comment 
    
    for comment in comment_blocks:
        try:
            # Comment text is usually the first navigation element
            dirty_text = comment.get_text(separator=" ", strip=True)
            text = pattern.sub("", dirty_text)
            
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
            
            sleep_politely()
        except Exception as e:
            print(f"Error on feed page {p}: {e}")
            
    if all_skipped:
        print(f"Skipped {len(all_skipped)} posts due to parsing errors.")
        
    return pd.DataFrame(all_posts)

def scrape_all_comments_metafilter(main_df):
    """Iterates through thread links in a DataFrame to scrape all comments.

    Also extracts post_full_text and outside_links for each thread and adds
    them to the main_df.
    """
    all_comments = []
    post_full_texts = []
    outside_links_list = []

    for idx, row in main_df.iterrows():
        try:
            url = row['thread_link']
            if idx % 20 == 0:
                print(f"On row {idx} of {len(main_df)}")

            # Get soup for thread page
            post_soup = get_soup_metafilter(url)

            # Extract main post content and outside links
            copy_div = post_soup.find('div', class_='copy')
            if copy_div:
                post_full_text = copy_div.get_text(separator=" ", strip=True)
                outside_links = [link['href'] for link in copy_div.find_all('a')]
            else:
                post_full_text = ""
                outside_links = []

            post_full_texts.append(post_full_text)
            outside_links_list.append(outside_links)

            # Parse comments using the same soup
            comments = parse_comments_metafilter(post_soup, url)
            all_comments.extend(comments)

            time.sleep(random.uniform(1, 3))
        except Exception as e:
            print(f"Error scraping thread {url}: {e}")
            post_full_texts.append("")
            outside_links_list.append([])

    # Add new columns to main_df
    main_df['post_full_text'] = post_full_texts
    main_df['outside_links'] = outside_links_list

    return pd.DataFrame(all_comments)

def scrape_metafilter(tag='politics', pages=1):
    """Scrapes MetaFilter feed and comments, returning both DataFrames.

    This is the main entry point that combines feed scraping and comment scraping.
    It also extracts post_full_text and outside_links for each thread.

    Args:
        tag: The MetaFilter tag to scrape (default: 'politics')
        pages: Number of feed pages to scrape (default: 1)

    Returns:
        tuple: (main_df, comments_df)
            - main_df: DataFrame with post metadata, post_full_text, and outside_links
            - comments_df: DataFrame with all comments from the scraped threads
    """
    # 1. Scrape the main feed
    main_df = scrape_main_feed_metafilter(tag=tag, pages=pages)

    # 2. Scrape comments and extract post content (modifies main_df in place)
    comments_df = scrape_all_comments_metafilter(main_df)

    return main_df, comments_df


# --- WORKFLOW ---

if __name__ == "__main__":
    # Scrape MetaFilter for politics posts
    df_main, df_comments = scrape_metafilter(tag='politics', pages=2)

    # Optionally merge for a master sentiment dataset
    final_df = pd.merge(df_comments, df_main, on='thread_link', how='left')

    print("\nScraping Complete!")
    print(f"Total Posts: {len(df_main)}")
    print(f"Total Comments: {len(df_comments)}")