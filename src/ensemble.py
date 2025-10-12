import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import joblib
import os
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """Weighted ensemble of multiple models"""
    
    def __init__(self, models: List[Any], weights: List[float] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit all models in the ensemble"""
        print("Training ensemble models...")
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}: {type(model).__name__}")
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"Error training model {i+1}: {e}")
                continue
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        predictions = []
        
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X)
                predictions.append(pred * self.weights[i])
            except Exception as e:
                print(f"Error predicting with model {i+1}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Sum weighted predictions
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred

class StackingEnsemble(BaseEstimator, RegressorMixin):
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, base_models: List[Any], meta_model: Any, cv_folds: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit base models and meta-learner"""
        print("Training stacking ensemble...")
        
        # Generate out-of-fold predictions for meta-learner
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Processing fold {fold+1}/{self.cv_folds}")
            
            X_train_fold = [X[i] for i in train_idx] if isinstance(X, list) else X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_val_fold = [X[i] for i in val_idx] if isinstance(X, list) else X.iloc[val_idx]
            
            for i, model in enumerate(self.base_models):
                try:
                    # Clone and fit model on fold
                    fold_model = self._clone_model(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    # Predict on validation set
                    val_pred = fold_model.predict(X_val_fold)
                    meta_features[val_idx, i] = val_pred
                    
                except Exception as e:
                    print(f"Error in fold {fold+1}, model {i+1}: {e}")
                    continue
        
        # Train base models on full dataset
        print("Training base models on full dataset...")
        for i, model in enumerate(self.base_models):
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"Error training base model {i+1}: {e}")
                continue
        
        # Train meta-learner
        print("Training meta-learner...")
        self.meta_model.fit(meta_features, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make stacking predictions"""
        if not self.is_fitted:
            raise ValueError("Stacking ensemble not fitted yet")
        
        # Get base model predictions
        base_predictions = np.zeros((len(X), len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            try:
                pred = model.predict(X)
                base_predictions[:, i] = pred
            except Exception as e:
                print(f"Error predicting with base model {i+1}: {e}")
                continue
        
        # Meta-learner prediction
        final_pred = self.meta_model.predict(base_predictions)
        return final_pred
    
    def _clone_model(self, model):
        """Clone a model for cross-validation"""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # For custom models, create new instance
            return type(model)(**model.get_params() if hasattr(model, 'get_params') else {})

class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """Blending ensemble using holdout validation"""
    
    def __init__(self, base_models: List[Any], meta_model: Any, holdout_size: float = 0.2):
        self.base_models = base_models
        self.meta_model = meta_model
        self.holdout_size = holdout_size
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit base models and meta-learner using blending"""
        print("Training blending ensemble...")
        
        # Split data for blending
        from sklearn.model_selection import train_test_split
        
        if isinstance(X, list):
            X_blend, X_holdout, y_blend, y_holdout = train_test_split(
                list(range(len(X))), y, test_size=self.holdout_size, random_state=42
            )
            X_blend_data = [X[i] for i in X_blend]
            X_holdout_data = [X[i] for i in X_holdout]
        else:
            X_blend_data, X_holdout_data, y_blend, y_holdout = train_test_split(
                X, y, test_size=self.holdout_size, random_state=42
            )
        
        # Train base models on blend set
        print("Training base models...")
        holdout_predictions = np.zeros((len(X_holdout_data), len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            try:
                print(f"Training base model {i+1}/{len(self.base_models)}")
                model.fit(X_blend_data, y_blend)
                
                # Predict on holdout set
                holdout_pred = model.predict(X_holdout_data)
                holdout_predictions[:, i] = holdout_pred
                
            except Exception as e:
                print(f"Error training base model {i+1}: {e}")
                continue
        
        # Train meta-learner on holdout predictions
        print("Training meta-learner...")
        self.meta_model.fit(holdout_predictions, y_holdout)
        
        # Retrain base models on full dataset
        print("Retraining base models on full dataset...")
        for i, model in enumerate(self.base_models):
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"Error retraining base model {i+1}: {e}")
                continue
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make blending predictions"""
        if not self.is_fitted:
            raise ValueError("Blending ensemble not fitted yet")
        
        # Get base model predictions
        base_predictions = np.zeros((len(X), len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            try:
                pred = model.predict(X)
                base_predictions[:, i] = pred
            except Exception as e:
                print(f"Error predicting with base model {i+1}: {e}")
                continue
        
        # Meta-learner prediction
        final_pred = self.meta_model.predict(base_predictions)
        return final_pred

class AdaptiveEnsemble(BaseEstimator, RegressorMixin):
    """Adaptive ensemble that learns optimal weights"""
    
    def __init__(self, models: List[Any], learning_rate: float = 0.01, n_iterations: int = 100):
        self.models = models
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit models and learn optimal weights"""
        print("Training adaptive ensemble...")
        
        # Train all base models
        for i, model in enumerate(self.models):
            try:
                print(f"Training model {i+1}/{len(self.models)}")
                model.fit(X, y)
            except Exception as e:
                print(f"Error training model {i+1}: {e}")
                continue
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                print(f"Error getting predictions: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        predictions = np.array(predictions).T  # Shape: (n_samples, n_models)
        
        # Initialize weights
        self.weights = np.ones(len(self.models)) / len(self.models)
        
        # Optimize weights using gradient descent
        print("Optimizing ensemble weights...")
        for iteration in range(self.n_iterations):
            # Current ensemble prediction
            ensemble_pred = np.dot(predictions, self.weights)
            
            # Calculate gradients
            error = ensemble_pred - y
            gradients = np.dot(predictions.T, error) / len(y)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            # Ensure weights are non-negative and sum to 1
            self.weights = np.maximum(self.weights, 0)
            self.weights /= np.sum(self.weights)
            
            if iteration % 20 == 0:
                mse = np.mean(error ** 2)
                print(f"Iteration {iteration}, MSE: {mse:.4f}")
        
        print(f"Final weights: {self.weights}")
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make adaptive ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Adaptive ensemble not fitted yet")
        
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        predictions = np.array(predictions).T
        ensemble_pred = np.dot(predictions, self.weights)
        return ensemble_pred

def create_ensemble_models():
    """Create different ensemble configurations"""
    from .models import TraditionalMLWrapper
    
    # Base models
    base_models = [
        TraditionalMLWrapper('xgboost', n_estimators=1000, max_depth=8, learning_rate=0.1),
        TraditionalMLWrapper('catboost', iterations=1000, depth=8, learning_rate=0.1),
        TraditionalMLWrapper('random_forest', n_estimators=500, max_depth=15),
        TraditionalMLWrapper('ridge', alpha=1.0)
    ]
    
    # Meta-learner
    meta_model = TraditionalMLWrapper('ridge', alpha=0.1)
    
    ensembles = {
        'weighted': WeightedEnsemble(base_models, weights=[0.3, 0.3, 0.25, 0.15]),
        'stacking': StackingEnsemble(base_models, meta_model, cv_folds=5),
        'blending': BlendingEnsemble(base_models, meta_model, holdout_size=0.2),
        'adaptive': AdaptiveEnsemble(base_models, learning_rate=0.01, n_iterations=100)
    }
    
    return ensembles

def save_ensemble(ensemble, filepath):
    """Save ensemble model"""
    joblib.dump(ensemble, filepath)
    print(f"Ensemble saved to {filepath}")

def load_ensemble(filepath):
    """Load ensemble model"""
    ensemble = joblib.load(filepath)
    print(f"Ensemble loaded from {filepath}")
    return ensemble
