"""
Script: Download Twitter posts using X (Twitter) API
Purpose: Download complete tweet content from post_links in opensource datasets to rebuild the dataset
Reference: https://docs.x.com/x-api/posts/get-post-by-id
"""

import json
import os
import time
import re
from pathlib import Path
from typing import Dict, List, Optional
import requests
from urllib.parse import urlparse


class TwitterAPIClient:
    """Twitter API Client"""
    
    def __init__(self, bearer_token: str):
        """
        Initialize API client
        
        Args:
            bearer_token: Twitter API Bearer Token
        """
        self.bearer_token = bearer_token
        self.base_url = "https://api.x.com/2/tweets"
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": "v2TweetLookupPython"
        }
        # API rate limiting (adjust based on your API tier)
        self.requests_per_15min = 300  # Basic tier
        self.request_count = 0
        self.window_start = time.time()
    
    def extract_tweet_id(self, post_link: str) -> Optional[str]:
        """
        Extract tweet ID from tweet link
        
        Args:
            post_link: Twitter post link
            
        Returns:
            tweet ID or None
        """
        # Supported URL formats:
        # https://x.com/username/status/1234567890
        # https://twitter.com/username/status/1234567890
        match = re.search(r'/status/(\d+)', post_link)
        if match:
            return match.group(1)
        return None
    
    def rate_limit_check(self):
        """Check and handle rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.window_start
        
        # Reset counter if more than 15 minutes have passed
        if elapsed >= 900:  # 15 minutes = 900 seconds
            self.request_count = 0
            self.window_start = current_time
        
        # Wait if limit is reached
        if self.request_count >= self.requests_per_15min:
            wait_time = 900 - elapsed + 1  # Wait until next window
            print(f"Rate limit reached, waiting {wait_time:.0f} seconds...")
            time.sleep(wait_time)
            self.request_count = 0
            self.window_start = time.time()
    
    def get_tweet(self, tweet_id: str) -> Optional[Dict]:
        """
        Get a single tweet
        
        Args:
            tweet_id: Tweet ID
            
        Returns:
            Tweet data dict or None
        """
        self.rate_limit_check()
        
        # API parameters - get text and media info
        params = {
            "ids": tweet_id,
            "tweet.fields": "created_at,text,attachments,entities",
            "expansions": "attachments.media_keys",
            "media.fields": "url,preview_image_url,type,alt_text"
        }
        
        try:
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=30
            )
            self.request_count += 1
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    return data
                else:
                    print(f"Warning: Tweet {tweet_id} returned no data")
                    return None
            elif response.status_code == 429:
                # Rate limit
                print(f"API rate limit, waiting...")
                time.sleep(60)
                return self.get_tweet(tweet_id)  # Retry
            else:
                print(f"Error: Tweet {tweet_id} request failed (status: {response.status_code})")
                print(f"Response: {response.text}")
                return None
        except Exception as e:
            print(f"Exception: Tweet {tweet_id} request error: {str(e)}")
            return None
    
    def extract_text_and_media(self, tweet_data: Dict) -> tuple[Optional[str], List[str]]:
        """
        Extract text and media URLs from API response
        
        Args:
            tweet_data: Tweet data returned by API
            
        Returns:
            (text, media_urls)
        """
        if not tweet_data or 'data' not in tweet_data:
            return None, []
        
        # Extract text
        text = tweet_data['data'][0].get('text', '')
        
        # Extract media URLs
        media_urls = []
        if 'includes' in tweet_data and 'media' in tweet_data['includes']:
            for media in tweet_data['includes']['media']:
                # For photos use url field; for videos use preview_image_url
                if media.get('type') == 'photo' and 'url' in media:
                    media_urls.append(media['url'])
                elif 'preview_image_url' in media:
                    media_urls.append(media['preview_image_url'])
        
        return text, media_urls


def clean_text(text: str) -> str:
    """
    Clean text content
    - Remove URLs (http://... or https://...)
    - Replace @username with <user>
    - Keep emojis and ensure RoBERTa can process them correctly
    
    Args:
        text: Original text
        
    Returns:
        Cleaned text
    """
    # 1. Remove URLs (http://... or https://...)
    # Match various URL formats
    url_pattern = r'https?://[^\s\n\r]+'
    text = re.sub(url_pattern, '', text)
    
    # Remove links starting with x.com (Twitter's new domain)
    x_pattern = r'https?://x\.com/[^\s\n\r]*'
    text = re.sub(x_pattern, '', text)
    
    # 2. Replace @username with <user>
    # Match @ followed by alphanumeric and underscore
    user_pattern = r'@\w+'
    text = re.sub(user_pattern, '<user>', text)
    
    # 3. Clean extra whitespace
    # Remove multiple consecutive spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text


def load_json(file_path: str) -> List[Dict]:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], file_path: str):
    """Save JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_checkpoint(data: List[Dict], checkpoint_path: str):
    """Save checkpoint"""
    save_json(data, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def download_image(url: str, save_path: Path) -> bool:
    """Download image from URL
    
    Args:
        url: Image URL
        save_path: Path to save the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"  Warning: Failed to download image (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"  Error downloading image: {e}")
        return False


def process_dataset(
    dataset: List[Dict],
    api_client: TwitterAPIClient,
    output_path: str,
    images_dir: Path,
    checkpoint_interval: int = 100
) -> List[Dict]:
    """
    Process dataset, download tweet content from API
    
    Args:# Clean the text: remove URLs and replace @mentions with <user>
                        cleaned_text = clean_text(text)
                        new_item['text'] = cleaned_
        dataset: Opensource dataset
        api_client: API client
        output_path: Output file path
        images_dir: Directory to save images
        checkpoint_interval: Checkpoint save interval
        
    Returns:
        Processed dataset
    """
    processed_data = []
    stats = {
        'total': len(dataset),
        'twitter_processed': 0,
        'twitter_success': 0,
        'twitter_failed': 0,
        'images_downloaded': 0,
        'images_failed': 0,
        'other_sources': 0,
        'skipped': 0
    }
    
    checkpoint_path = output_path.replace('.json', '_checkpoint.json')
    
    for idx, item in enumerate(dataset):
        new_item = item.copy()
        source = item.get('source', '')
        
        if source == 'Twitter' and 'post_link' in item:
            is_generated = item.get('generated', False)
            
            if is_generated:
                # For generated data, text is synthetic and can be kept
                # Only download images, don't replace text
                print(f"[{idx+1}/{len(dataset)}] Processing generated data ID: {item.get('id')} (keeping text, downloading images only)")
                
                image_urls = item.get('image_urls', [])
                if image_urls:
                    item_id = item.get('id')
                    downloaded_images = []
                    for img_idx, img_url in enumerate(image_urls, 1):
                        # Determine file extension from URL
                        ext = '.jpg'  # Default
                        if '.png' in img_url:
                            ext = '.png'
                        elif '.gif' in img_url:
                            ext = '.gif'
                        
                        img_filename = f"{item_id}_{img_idx}{ext}"
                        img_path = images_dir / img_filename
                        
                        if download_image(img_url, img_path):
                            downloaded_images.append(f"./images/{img_filename}")
                            stats['images_downloaded'] += 1
                        else:
                            stats['images_failed'] += 1
                    
                    # Update images field with local paths
                    if downloaded_images:
                        new_item['images'] = downloaded_images
                
                stats['twitter_success'] += 1
                stats['twitter_processed'] += 1
            else:
                # For non-generated data, download from API
                post_link = item['post_link']
                tweet_id = api_client.extract_tweet_id(post_link)
                
                if tweet_id:
                    print(f"[{idx+1}/{len(dataset)}] Downloading Tweet ID: {tweet_id}")
                    tweet_data = api_client.get_tweet(tweet_id)
                    
                    if tweet_data:
                        text, media_urls = api_client.extract_text_and_media(tweet_data)
                        if text:
                            # Clean the text: remove URLs and replace @mentions with <user>
                            cleaned_text = clean_text(text)
                            new_item['text'] = cleaned_text
                            # Update image_urls if API returned media URLs
                            if media_urls:
                                new_item['image_urls'] = media_urls
                                
                                # Download images
                                item_id = item.get('id')
                                downloaded_images = []
                                for img_idx, img_url in enumerate(media_urls, 1):
                                    # Determine file extension from URL
                                    ext = '.jpg'  # Default
                                    if '.png' in img_url:
                                        ext = '.png'
                                    elif '.gif' in img_url:
                                        ext = '.gif'
                                    
                                    img_filename = f"{item_id}_{img_idx}{ext}"
                                    img_path = images_dir / img_filename
                                    
                                    if download_image(img_url, img_path):
                                        downloaded_images.append(f"./images/{img_filename}")
                                        stats['images_downloaded'] += 1
                                    else:
                                        stats['images_failed'] += 1
                                
                                # Update images field with local paths
                                if downloaded_images:
                                    new_item['images'] = downloaded_images
                            
                            stats['twitter_success'] += 1
                        else:
                            print(f"Warning: Tweet {tweet_id} could not extract text")
                            stats['twitter_failed'] += 1
                    else:
                        print(f"Warning: Tweet {tweet_id} download failed")
                        stats['twitter_failed'] += 1
                    
                    stats['twitter_processed'] += 1
                    
                    # Add small delay to avoid rapid requests
                    time.sleep(0.1)
                else:
                    print(f"Warning: Cannot extract Tweet ID from link: {post_link}")
                    stats['twitter_failed'] += 1
        else:
            # Non-Twitter data or data that already has text
            if source == 'Twitter':
                stats['skipped'] += 1
            else:
                stats['other_sources'] += 1
        
        processed_data.append(new_item)
        
        # Save checkpoint periodically
        if (idx + 1) % checkpoint_interval == 0:
            save_checkpoint(processed_data, checkpoint_path)
            print(f"Progress: {idx+1}/{len(dataset)}")
            print(f"  Success: {stats['twitter_success']}, Failed: {stats['twitter_failed']}")
            print(f"  Images downloaded: {stats['images_downloaded']}, Failed: {stats['images_failed']}")
    
    return processed_data, stats


def main():
    # ========================================
    # Set your Twitter Bearer Token here
    # ========================================
    bearer_token = "YOUR_BEARER_TOKEN_HERE"  # Replace with your actual token
    if not bearer_token or bearer_token == "YOUR_BEARER_TOKEN_HERE":
        print("Error: Please set your Twitter Bearer Token in the code")
        print("Edit the bearer_token variable in download_tweets_from_api.py")
        print("How to get token: https://developer.x.com/en/docs/authentication/oauth-2-0/bearer-tokens")
        return
    
    # Define file paths
    data_dir = Path('data/MMSD3')
    images_dir = Path('data/images')
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create API client
    api_client = TwitterAPIClient(bearer_token)
    
    # Process train/test/val datasets
    # Generated datasets will contain complete text downloaded from X API, with identical train/test/val splits
    datasets = [
        ('train_data_opensource.json', 'train_data.json'),
        ('test_data_opensource.json', 'test_data.json'),
        ('val_data_opensource.json', 'val_data.json')
    ]
    
    for input_file, output_file in datasets:
        input_path = data_dir / input_file
        output_path = data_dir / output_file
        
        if not input_path.exists():
            print(f"Warning: File does not exist {input_path}, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print(f"{'='*60}")
        
        # Load data
        dataset = load_json(input_path)
        print(f"Dataset size: {len(dataset)} items")
        
        # Process dataset
        processed_data, stats = process_dataset(
            dataset,
            api_client,
            str(output_path),
            images_dir,
            checkpoint_interval=100
        )
        
        # Save final results
        save_json(processed_data, output_path)
        
        print(f"\nProcessing statistics:")
        print(f"  Total data: {stats['total']}")
        print(f"  Twitter processed: {stats['twitter_processed']}")
        print(f"  Twitter success: {stats['twitter_success']}")
        print(f"  Twitter failed: {stats['twitter_failed']}")
        print(f"  Images downloaded: {stats['images_downloaded']}")
        print(f"  Images failed: {stats['images_failed']}")
        print(f"  Other sources: {stats['other_sources']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "="*60)
    print("All datasets processed!")
    print("="*60)


if __name__ == '__main__':
    main()
