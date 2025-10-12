import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append('.')

from src.pipeline import PricingPipeline
from src.validation import CrossValidator, create_validation_report
from src.models import TraditionalMLWrapper
from src.ensemble import create_ensemble_models

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Amazon ML Challenge pricing models')
    
    parser.add_argument('--download-images', action='store_true',
                       help='Download product images (takes time)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--ensemble-type', type=str, default='stacking',
                       choices=['weighted', 'stacking', 'blending', 'adaptive'],
                       help='Type of ensemble to use')
    
    return parser.parse_args()

def create_custom_config(args):
    """Create custom configuration based on arguments"""
    config = {
        'data_paths': {
            'train_csv': 'dataset/processed_subset/train_clean_subset.csv',
            'test_csv': 'dataset/processed_subset/test_clean_subset.csv',
            'image_folder': 'dataset/images/'
        },
        'model_params': {
            'text_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'image_model': 'resnet50',
            'hidden_dim': 256
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 50,
            'early_stopping_patience': 10
        },
        'validation': {
            'cv_folds': args.cv_folds,
            'test_size': 0.2
        },
        'ensemble': {
            'use_ensemble': True,
            'ensemble_type': args.ensemble_type
        }
    }
    return config

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration
    config = create_custom_config(args)
    
    try:
        # Initialize pipeline
        print("\n" + "="*60)
        print("INITIALIZING PIPELINE")
        print("="*60)
        
        pipeline = PricingPipeline(config)
        
        # Load data
        print("\n1. Loading and preprocessing data...")
        train_df, test_df = pipeline.load_data(download_images_flag=args.download_images)
        
        print(f"Training data: {train_df.shape}")
        print(f"Test data: {test_df.shape}")
        
        # Train traditional models
        print("\n2. Training traditional ML models...")
        traditional_models = pipeline.train_traditional_models()
        
        print(f"Trained {len(traditional_models)} traditional models")
        
        # Create ensemble
        print("\n3. Creating ensemble model...")
        ensemble = pipeline.create_ensemble()
        
        if ensemble:
            print(f"Created {config['ensemble']['ensemble_type']} ensemble")
        else:
            print("Ensemble creation failed, will use individual models")
        
        # Validate models
        print("\n4. Validating models with cross-validation...")
        validation_results = pipeline.validate_models()
        
        if validation_results is not None:
            print("Cross-validation completed")
            
            # Save validation results
            validation_results.to_csv(
                os.path.join(args.output_dir, 'validation_results.csv')
            )
            
            # Create validation report
            create_validation_report(
                pipeline.models,
                os.path.join(args.output_dir, 'validation_report.txt')
            )
        
        # Make predictions
        print("\n5. Making predictions on test data...")
        predictions = pipeline.predict(use_ensemble=True)
        
        print(f"Generated {len(predictions)} predictions")
        print(f"Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        
        # Save predictions
        print("\n6. Saving predictions...")
        output_path = os.path.join(args.output_dir, 'test_out.csv')
        output_df = pipeline.save_predictions(predictions, output_path)
        
        print(f"Predictions saved to {output_path}")
        
        # Save model summary
        print("\n7. Creating model summary...")
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'data_shapes': {
                'train': train_df.shape,
                'test': test_df.shape
            },
            'models_trained': list(pipeline.models.keys()),
            'ensemble_used': ensemble is not None,
            'prediction_stats': {
                'count': len(predictions),
                'min_price': float(predictions.min()),
                'max_price': float(predictions.max()),
                'mean_price': float(predictions.mean()),
                'std_price': float(predictions.std())
            }
        }
        
        import json
        with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"Results saved in: {args.output_dir}/")
        print("Files created:")
        print(f"  - test_out.csv (final predictions)")
        print(f"  - validation_results.csv (CV results)")
        print(f"  - validation_report.txt (detailed report)")
        print(f"  - training_summary.json (run summary)")
        print(f"  - validation_results.png (plots)")
        
        # Display best model performance
        if validation_results is not None:
            best_model = validation_results.index[0]  # Already sorted by SMAPE
            best_smape = validation_results.loc[best_model, 'SMAPE_mean']
            print(f"Best Model: {best_model}")
            print(f"Best SMAPE: {best_smape:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
