import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
from datetime import datetime


# Search for geopolitical YouTube videos
def get_geopolitical_videos(api_key, max_results=1000):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    search_queries = [
        'news today',
        'international relations',
        'global politics',
        'politics today',
        'world politics news'
    ]
    
    videos = []
    videos_per_query = max_results // len(search_queries)
    
    for query in search_queries:
        try:
            request = youtube.search().list(
                part='snippet',
                q=query,
                type='video',
                maxResults=min(videos_per_query, 50),
                order='relevance',
                relevanceLanguage='en',
                videoDefinition='any'
            )
            response = request.execute()
            
            for item in response['items']:
                video_id = item['id']['videoId']
                videos.append({
                    'id': video_id,
                    'title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt']
                })
            
            time.sleep(1)
        
        except HttpError as e:
            print(f"Error searching for '{query}': {e}")
            continue
        
    return videos[:max_results]

# Get video statistics including comment count
def get_video_stats(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    try:
        request = youtube.videos().list(
            part='statistics',
            id=video_id
        )
        response = request.execute()
        
        if response['items']:
            stats = response['items'][0]['statistics']
            return int(stats.get('commentCount', 0))
        return 0
    
    except HttpError as e:
        print(f"Error getting stats for video {video_id}: {e}")
        return 0

# Get top comments for a specific video with metadata
def get_top_comments(api_key, video_id, max_comments=10):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_comments,
            order='relevance',
            textFormat='plainText'
        )
        response = request.execute()
        
        comments = []
        for item in response['items']:
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'post_id': video_id,
                'username': snippet['authorDisplayName'],
                'created_at': snippet['publishedAt'],
                'comment_text': snippet['textDisplay']
            })
        
        return comments
    
    except HttpError as e:
        print(f"Error getting comments for video {video_id}: {e}")
        return []

"""
Main function to create two separate dataframes
Args:
    api_key: YouTube Data API key
    num_videos: Number of videos to process
    comments_per_video: Number of comments per video
    
Returns:
    Tuple of (videos_df, comments_df)
"""
def create_datasets(api_key, num_videos=1000, comments_per_video=10):
    print(f"Searching for {num_videos} geopolitical videos...")
    videos = get_geopolitical_videos(api_key, num_videos)
    print(f"Found {len(videos)} videos")
    
    videos_data = []
    comments_data = []
    
    for idx, video in enumerate(videos, 1):
        print(f"Processing video {idx}/{len(videos)}: {video['title'][:50]}...")
        
        # Get comment count
        comment_count = get_video_stats(api_key, video['id'])
        
        # Add to videos dataframe
        videos_data.append({
            'post_id': video['id'],
            'post_title': video['title'],
            'post_author': video['channel'],
            'created_at': video['published_at'],
            'comment_count': comment_count,
            'source': 'YouTube'
        })
        
        # Get comments
        comments = get_top_comments(api_key, video['id'], comments_per_video)
        comments_data.extend(comments)
        
        time.sleep(0.5)
    
    # Create dataframes
    videos_df = pd.DataFrame(videos_data)
    comments_df = pd.DataFrame(comments_data)
    
    print(f"\nCompleted!")
    print(f"Videos dataframe: {len(videos_df)} rows")
    print(f"Comments dataframe: {len(comments_df)} rows")
    
    return videos_df, comments_df

# Usage Example
if __name__ == "__main__":
    API_KEY = 'AIzaSyB48rSz8fCRlDmvsrURINBsHgLr_LjA5IA'
    
    # Create the two dataframes
    videos_df, comments_df = create_datasets(
        api_key=API_KEY,
        num_videos=1000,
        comments_per_video=10
    )
    
    # Display first few rows of each
    print("\n=== VIDEOS DATAFRAME ===")
    print(videos_df.head())
    print(f"\nColumns: {list(videos_df.columns)}")
    
    print("\n=== COMMENTS DATAFRAME ===")
    print(comments_df.head())
    print(f"\nColumns: {list(comments_df.columns)}")
    
    # Save to CSV
    videos_df.to_csv('youtube_videos.csv', index=False)
    comments_df.to_csv('youtube_comments.csv', index=False)
    print("\nData saved to 'youtube_videos.csv' and 'youtube_comments.csv'")
    
    # Basic statistics
    print(f"\nTotal videos: {len(videos_df)}")
    print(f"Total comments collected: {len(comments_df)}")
    print(f"Average comments per video: {len(comments_df) / len(videos_df):.1f}")