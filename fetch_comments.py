import os
import re
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
VIDEO_URL = os.getenv("YOUTUBE_VIDEO_URL")

def extract_video_id(url):
    """
    Extracts the YouTube Video ID from various URL formats.
    """
    if not url:
        return None
    
    # Regex patterns for standard, short, and embed YouTube URLs
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&?\/.]|$)',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'youtube\.com\/embed\/([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
            
    return None

def fetch_all_comments(video_id, api_key):
    """
    Fetches all top-level comments and replies for a given YouTube Video ID.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None
    
    print(f"[INFO] Starting to fetch comments for Video ID: {video_id}")
    
    try:
        while True:
            # Request comment threads
            request = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat='plainText'
            )
            response = request.execute()
            
            for item in response.get('items', []):
                # Top-level comment
                top_comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                top_comment_clean = re.sub(r'[\r\n]+', ' ', top_comment).strip()
                if top_comment_clean:
                    comments.append(top_comment_clean)
                
                # Replies (if any)
                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_text = reply['snippet']['textDisplay']
                        reply_text_clean = re.sub(r'[\r\n]+', ' ', reply_text).strip()
                        if reply_text_clean:
                            comments.append(reply_text_clean)
            
            total_fetched = len(comments)
            print(f"[PROGRESS] Total comments fetched so far: {total_fetched}")
            
            # Check for next page token
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
    except HttpError as e:
        print(f"[ERROR] An HTTP error occurred: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        
    return comments

def main():
    # Validate environment variables
    if not API_KEY or not VIDEO_URL:
        print("[ERROR] Missing YOUTUBE_API_KEY or YOUTUBE_VIDEO_URL in .env file.")
        return
        
    video_id = extract_video_id(VIDEO_URL)
    if not video_id:
        print(f"[ERROR] Could not extract a valid Video ID from URL: {VIDEO_URL}")
        return
        
    # Fetch comments
    raw_comments = fetch_all_comments(video_id, API_KEY)
    
    if not raw_comments:
        print("[WARNING] No comments were retrieved.")
        return
        
    # Ensure 'datasets' directory exists
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ONLY the text column to CSV
    df = pd.DataFrame({'text': raw_comments})
    output_filepath = os.path.join(output_dir, f"comments.csv")
    
    df.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f"[SUCCESS] Total {len(df)} comments saved to '{output_filepath}'.")

if __name__ == "__main__":
    main()