import os
import requests
import pandas as pd
from urllib.parse import urlparse
import hashlib
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

def get_image_filename_from_url(url):
    """Generate a consistent filename from URL"""
    if pd.isna(url) or not url:
        return None
    
    # Create hash from URL for consistent naming
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return f"{url_hash}.jpg"

def download_single_image(url, save_path, timeout=10, max_retries=3):
    """Download a single image with retry logic"""
    if pd.isna(url) or not url:
        return False, "No URL provided"
    
    if os.path.exists(save_path):
        return True, "Already exists"
    
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False, f"Not an image: {content_type}"
            
            # Save the image
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True, "Downloaded successfully"
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return False, f"Failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)  # Wait before retry
    
    return False, "Unknown error"

def download_images_batch(urls, save_dir, max_workers=10, batch_size=100):
    """Download images in batches with progress tracking"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare download tasks
    download_tasks = []
    for url in urls:
        if pd.notna(url) and url:
            filename = get_image_filename_from_url(url)
            if filename:
                save_path = os.path.join(save_dir, filename)
                download_tasks.append((url, save_path))
    
    print(f"Downloading {len(download_tasks)} images to {save_dir}")
    
    successful_downloads = 0
    failed_downloads = 0
    already_exists = 0
    
    # Process in batches to avoid overwhelming the server
    for i in range(0, len(download_tasks), batch_size):
        batch = download_tasks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(download_tasks)-1)//batch_size + 1}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks in the batch
            future_to_task = {
                executor.submit(download_single_image, url, save_path): (url, save_path)
                for url, save_path in batch
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_task), total=len(batch), desc="Downloading"):
                url, save_path = future_to_task[future]
                try:
                    success, message = future.result()
                    if success:
                        if "Already exists" in message:
                            already_exists += 1
                        else:
                            successful_downloads += 1
                    else:
                        failed_downloads += 1
                except Exception as e:
                    failed_downloads += 1
        
        # Small delay between batches
        if i + batch_size < len(download_tasks):
            time.sleep(1)
    
    print(f"\nDownload Summary:")
    print(f"  Successful: {successful_downloads}")
    print(f"  Already existed: {already_exists}")
    print(f"  Failed: {failed_downloads}")
    print(f"  Total processed: {len(download_tasks)}")
    
    return successful_downloads, already_exists, failed_downloads

def extract_images_from_csv(csv_path, image_column='image_link', save_dir='dataset/images/', max_workers=10):
    """Extract and download images from CSV file"""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if image_column not in df.columns:
        print(f"Error: Column '{image_column}' not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    # Get unique URLs to avoid duplicates
    unique_urls = df[image_column].dropna().unique()
    print(f"Found {len(unique_urls)} unique image URLs")
    
    # Download images
    success, exists, failed = download_images_batch(
        unique_urls, save_dir, max_workers=max_workers
    )
    
    # Create mapping file
    mapping_data = []
    for url in unique_urls:
        filename = get_image_filename_from_url(url)
        if filename:
            mapping_data.append({
                'image_url': url,
                'filename': filename,
                'local_path': os.path.join(save_dir, filename)
            })
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_path = os.path.join(save_dir, 'image_mapping.csv')
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Image mapping saved to {mapping_path}")
    
    return True

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and download product images')
    parser.add_argument('--train-csv', type=str, 
                       default='dataset/processed_subset/train_clean_subset.csv',
                       help='Path to training CSV file')
    parser.add_argument('--test-csv', type=str,
                       default='dataset/processed_subset/test_clean_subset.csv', 
                       help='Path to test CSV file')
    parser.add_argument('--save-dir', type=str, default='dataset/images/',
                       help='Directory to save images')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Number of parallel download workers')
    parser.add_argument('--image-column', type=str, default='image_link',
                       help='Name of the image URL column')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AMAZON ML CHALLENGE - IMAGE EXTRACTION")
    print("="*60)
    
    # Process training data
    if os.path.exists(args.train_csv):
        print(f"\nProcessing training data: {args.train_csv}")
        extract_images_from_csv(
            args.train_csv, 
            args.image_column, 
            args.save_dir, 
            args.max_workers
        )
    else:
        print(f"Training CSV not found: {args.train_csv}")
    
    # Process test data
    if os.path.exists(args.test_csv):
        print(f"\nProcessing test data: {args.test_csv}")
        extract_images_from_csv(
            args.test_csv, 
            args.image_column, 
            args.save_dir, 
            args.max_workers
        )
    else:
        print(f"Test CSV not found: {args.test_csv}")
    
    print("\n" + "="*60)
    print("IMAGE EXTRACTION COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()