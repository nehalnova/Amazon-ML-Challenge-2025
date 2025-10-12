import re
import os
import pandas as pd
import multiprocessing
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib
from PIL import Image
import torch
from torchvision import transforms
import joblib

def download_image(image_link, savefolder):
    if(isinstance(image_link, str)):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if(not os.path.exists(image_save_path)):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)    
            except Exception as ex:
                print('Warning: Not able to download - {}\n{}'.format(image_link, ex))
        else:
            return
    return

def download_images(image_links, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    results = []
    download_image_partial = partial(download_image, savefolder=download_folder)
    with multiprocessing.Pool(100) as pool:
        for result in tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)):
            results.append(result)
        pool.close()
        pool.join()

def load_and_preprocess_image(image_path, transform=None):
    """Load and preprocess image for model input"""
    try:
        image = Image.open(image_path).convert('RGB')
        if transform:
            image = transform(image)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a blank image tensor if loading fails
        if transform:
            blank_image = Image.new('RGB', (224, 224), color='white')
            return transform(blank_image)
        return None

def get_image_path_from_url(image_url, download_folder):
    """Get local image path from URL"""
    filename = Path(image_url).name
    return os.path.join(download_folder, filename)

def clean_text_advanced(text):
    """Advanced text cleaning for product descriptions"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\-\.\,\(\)]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_price_features(text):
    """Extract price-related features from text"""
    features = {}
    
    # Look for quantity indicators
    qty_patterns = [r'pack of (\d+)', r'(\d+) pack', r'(\d+)x', r'x(\d+)', r'(\d+) pieces']
    for pattern in qty_patterns:
        match = re.search(pattern, text.lower())
        if match:
            features['quantity'] = int(match.group(1))
            break
    else:
        features['quantity'] = 1
    
    # Look for size/weight indicators
    size_patterns = [r'(\d+\.?\d*)\s*(kg|g|lb|oz|ml|l|litre|liter)', 
                     r'(\d+\.?\d*)\s*(inch|inches|cm|mm|ft|feet)']
    for pattern in size_patterns:
        match = re.search(pattern, text.lower())
        if match:
            features['size_value'] = float(match.group(1))
            features['size_unit'] = match.group(2)
            break
    else:
        features['size_value'] = 0
        features['size_unit'] = 'unknown'
    
    # Brand indicators (common brand keywords)
    brand_keywords = ['apple', 'samsung', 'nike', 'adidas', 'sony', 'lg', 'hp', 'dell']
    features['has_brand'] = any(brand in text.lower() for brand in brand_keywords)
    
    return features

def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath}, epoch: {epoch}, loss: {loss}")
    return epoch, loss