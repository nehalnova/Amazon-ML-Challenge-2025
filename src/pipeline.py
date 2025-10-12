import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .models import TraditionalMLWrapper, DeepLearningWrapper, get_image_transforms
from .ensemble import (
    WeightedEnsemble, StackingEnsemble, BlendingEnsemble, 
    AdaptiveEnsemble, create_ensemble_models
)
from .validation import CrossValidator, smape, calculate_metrics
from .utils import (
    download_images, load_and_preprocess_image, get_image_path_from_url,
    clean_text_advanced, extract_price_features, save_model_checkpoint, load_model_checkpoint
)

class ProductDataset(Dataset):
    """Dataset class for product data"""
    
    def __init__(self, texts, image_paths, prices=None, transform=None):
        self.texts = texts
        self.image_paths = image_paths
        self.prices = prices
        self.transform = transform
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        image_path = self.image_paths[idx]
        
        # Load image
        image = load_and_preprocess_image(image_path, self.transform)
        if image is None:
            # Create blank tensor if image loading fails
            image = torch.zeros(3, 224, 224)
        
        if self.prices is not None:
            price = float(self.prices[idx])
            return text, image, price
        else:
            return text, image

class PricingPipeline:
    """End-to-end pricing prediction pipeline"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.ensemble = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_transform = get_image_transforms()
        print(f"Initialized pricing pipeline - Device: {self.device}")
        
    def _get_default_config(self):
        """Get default configuration"""
        return {
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
                'cv_folds': 5,
                'test_size': 0.2
            },
            'ensemble': {
                'use_ensemble': True,
                'ensemble_type': 'stacking'  # weighted, stacking, blending, adaptive
            }
        }
    
    def load_data(self, download_images_flag=False):
        """Load and prepare data from processed subset"""
        print("Loading data...")
        
        # Load CSV files (already processed with clean_text column)
        train_df = pd.read_csv(self.config['data_paths']['train_csv'])
        test_df = pd.read_csv(self.config['data_paths']['test_csv'])
        
        print(f"Train data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Use existing clean_text column if available, otherwise clean catalog_content
        if 'clean_text' not in train_df.columns:
            train_df['clean_text'] = train_df['catalog_content'].apply(clean_text_advanced)
        if 'clean_text' not in test_df.columns:
            test_df['clean_text'] = test_df['catalog_content'].apply(clean_text_advanced)
        
        self.train_df = train_df
        self.test_df = test_df
        
        return train_df, test_df
    
    def prepare_datasets(self):
        """Prepare PyTorch datasets for training"""
        print("Preparing datasets...")
        
        # Create image paths from image links (for future use)
        self.train_df['image_path'] = self.train_df['image_link'].apply(
            lambda x: f"dataset/images/{hash(x) % 100000}.jpg" if pd.notna(x) else None
        )
        self.test_df['image_path'] = self.test_df['image_link'].apply(
            lambda x: f"dataset/images/{hash(x) % 100000}.jpg" if pd.notna(x) else None
        )
        
        # Training dataset
        train_dataset = ProductDataset(
            texts=self.train_df['clean_text'].tolist(),
            image_paths=self.train_df['image_path'].tolist(),
            prices=self.train_df['price'].values,
            transform=self.image_transform
        )
        
        # Test dataset
        test_dataset = ProductDataset(
            texts=self.test_df['clean_text'].tolist(),
            image_paths=self.test_df['image_path'].tolist(),
            transform=self.image_transform
        )
        
        return train_dataset, test_dataset
    
    def train_deep_learning_models(self):
        """Train deep learning models"""
        print("Training deep learning models...")
        
        # Prepare data
        X_text = self.train_df['clean_text'].tolist()
        y = self.train_df['price'].values
        
        # Initialize deep learning models
        dl_models = {
            'dl_text_only': DeepLearningWrapper('text_only', epochs=20, batch_size=64),
            'dl_multimodal': DeepLearningWrapper('multimodal', epochs=20, batch_size=32)
        }
        
        trained_dl = {}
        
        for name, model in dl_models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_text, y)
                trained_dl[name] = model
                print(f"{name} training completed")
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        self.models.update(trained_dl)
        return trained_dl
    
    def train_traditional_models(self):
        """Train traditional ML models"""
        print("Training traditional ML models...")
        
        # Prepare text data
        X_text = self.train_df['clean_text'].tolist()
        y = self.train_df['price'].values
        
        # Initialize traditional models with reduced complexity
        traditional_models = {
            'xgboost': TraditionalMLWrapper('xgboost', n_estimators=200, max_depth=6),
            'catboost': TraditionalMLWrapper('catboost', iterations=200, depth=6),
            'random_forest': TraditionalMLWrapper('random_forest', n_estimators=100),
            'ridge': TraditionalMLWrapper('ridge', alpha=1.0)
        }
        
        trained_traditional = {}
        
        for name, model in traditional_models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_text, y)
                trained_traditional[name] = model
                print(f"{name} training completed")
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        self.models.update(trained_traditional)
        return trained_traditional
    
    def create_ensemble(self):
        """Create ensemble from trained models"""
        print("Creating ensemble...")
        
        # Get all models for ensemble (traditional ML + deep learning)
        base_models = [
            model for name, model in self.models.items() 
            if isinstance(model, (TraditionalMLWrapper, DeepLearningWrapper))
        ]
        
        if len(base_models) < 2:
            print("Not enough models for ensemble")
            return None
        
        # Create ensemble based on type
        ensemble_type = self.config['ensemble']['ensemble_type']
        
        if ensemble_type == 'weighted':
            weights = [0.3, 0.3, 0.25, 0.15][:len(base_models)]
            ensemble = WeightedEnsemble(base_models, weights)
        elif ensemble_type == 'stacking':
            meta_model = TraditionalMLWrapper('ridge', alpha=0.1)
            ensemble = StackingEnsemble(base_models, meta_model, cv_folds=5)
        elif ensemble_type == 'blending':
            meta_model = TraditionalMLWrapper('ridge', alpha=0.1)
            ensemble = BlendingEnsemble(base_models, meta_model, holdout_size=0.2)
        elif ensemble_type == 'adaptive':
            ensemble = AdaptiveEnsemble(base_models, learning_rate=0.01, n_iterations=100)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        # Train ensemble
        X_text = self.train_df['clean_text'].tolist()
        y = self.train_df['price'].values
        
        ensemble.fit(X_text, y)
        self.ensemble = ensemble
        
        return ensemble
    
    def validate_models(self):
        """Validate all models using cross-validation"""
        print("Validating models...")
        
        # Prepare data for validation
        X_text = self.train_df['clean_text'].tolist()
        y = self.train_df['price'].values
        
        # Initialize validator
        validator = CrossValidator(
            cv_strategy='kfold',
            n_splits=self.config['validation']['cv_folds']
        )
        
        # Validate all models (traditional + deep learning)
        all_models = {
            name: model for name, model in self.models.items()
            if isinstance(model, (TraditionalMLWrapper, DeepLearningWrapper))
        }
        
        if all_models:
            results = validator.validate_multiple_models(all_models, X_text, y)
            
            # Validate ensemble if available
            if self.ensemble:
                ensemble_result = validator.validate_model(self.ensemble, X_text, y, 'Ensemble')
                
            # Plot results
            validator.plot_cv_results('validation_results.png')
            
            return results
        else:
            print("No traditional models available for validation")
            return None
    
    def predict(self, use_ensemble=True):
        """Make predictions on test data"""
        print("Making predictions...")
        
        # Prepare test data
        X_test = self.test_df['clean_text'].tolist()
        
        if use_ensemble and self.ensemble:
            print("Using ensemble for predictions")
            predictions = self.ensemble.predict(X_test)
        else:
            # Use best performing model (traditional or deep learning)
            all_models = {
                name: model for name, model in self.models.items()
                if isinstance(model, (TraditionalMLWrapper, DeepLearningWrapper))
            }
            
            if all_models:
                # Use first available model (in practice, use best from validation)
                model_name = list(all_models.keys())[0]
                model = all_models[model_name]
                print(f"Using {model_name} for predictions")
                predictions = model.predict(X_test)
            else:
                raise ValueError("No models available for prediction")
        
        # Ensure positive predictions
        predictions = np.maximum(predictions, 0.01)
        
        return predictions
    
    def save_predictions(self, predictions, output_path='test_out.csv'):
        """Save predictions in required format"""
        print(f"Saving predictions to {output_path}")
        
        output_df = pd.DataFrame({
            'sample_id': self.test_df['sample_id'],
            'price': predictions
        })
        
        output_df.to_csv(output_path, index=False)
        
        print(f"Predictions saved successfully!")
        print(f"Total predictions: {len(output_df)}")
        print(f"Sample predictions:\n{output_df.head()}")
        
        return output_df
    
    def run_full_pipeline(self, download_images_flag=False):
        """Run the complete pipeline"""
        print("="*60)
        print("AMAZON ML CHALLENGE - PRICING PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Load data
            self.load_data(download_images_flag)
            
            # Step 2: Prepare datasets
            train_dataset, test_dataset = self.prepare_datasets()
            
            # Step 3: Train traditional ML models
            traditional_models = self.train_traditional_models()
            
            # Step 4: Train deep learning models
            deep_learning_models = self.train_deep_learning_models()
            
            # Step 5: Create ensemble
            ensemble = self.create_ensemble()
            
            # Step 6: Validate models
            validation_results = self.validate_models()
            
            # Step 7: Make predictions
            predictions = self.predict(use_ensemble=True)
            
            # Step 8: Save predictions
            output_df = self.save_predictions(predictions)
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return output_df, validation_results
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise

def main():
    """Main function to run the pipeline"""
    # Initialize pipeline
    pipeline = PricingPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(download_images_flag=False)
    
    return results

if __name__ == "__main__":
    main()
