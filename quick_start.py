#!/usr/bin/env python3
"""
Quick Start Script for Amazon ML Challenge
Simple execution without complex configuration
"""

import sys
import os
sys.path.append('src')

def quick_train():
    """Quick training with default settings"""
    print("Quick Start - Amazon ML Challenge")
    print("=" * 50)
    
    try:
        from src.pipeline import PricingPipeline
        
        # Initialize with default config
        pipeline = PricingPipeline()
        
        # Run the full pipeline
        results = pipeline.run_full_pipeline(download_images_flag=False)
        
        print("\nQuick training completed!")
        print("Check 'test_out.csv' for predictions")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have the processed data in dataset/processed_subset/")
        print("2. Run: python src/1_data_preprocessing_subset.py first")
        print("3. Check if all dependencies are installed")
        return None

if __name__ == "__main__":
    quick_train()
