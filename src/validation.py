import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        SMAPE score (0-200, lower is better)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle zero values to avoid division by zero
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    
    smape_score = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return smape_score

def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    y_true = np.where(y_true == 0, 1e-8, y_true)
    
    mape_score = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape_score

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        'SMAPE': smape(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

class CrossValidator:
    """Cross-validation framework with SMAPE evaluation"""
    
    def __init__(self, cv_strategy='kfold', n_splits=5, random_state=42):
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = {}
        
    def _get_cv_splitter(self, X, y):
        """Get cross-validation splitter based on strategy"""
        if self.cv_strategy == 'kfold':
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        elif self.cv_strategy == 'stratified':
            # Create price bins for stratification
            price_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        elif self.cv_strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def validate_model(self, model, X, y, model_name='Model'):
        """Perform cross-validation on a single model"""
        print(f"\nValidating {model_name}...")
        
        cv_splitter = self._get_cv_splitter(X, y)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            print(f"Processing fold {fold + 1}/{self.n_splits}")
            
            try:
                # Split data
                if isinstance(X, list):
                    X_train = [X[i] for i in train_idx]
                    X_val = [X[i] for i in val_idx]
                else:
                    X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                    X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                
                y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
                
                # Clone and train model
                fold_model = self._clone_model(model)
                fold_model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = fold_model.predict(X_val)
                
                # Calculate metrics
                fold_metrics = calculate_metrics(y_val, y_pred)
                fold_metrics['fold'] = fold + 1
                fold_results.append(fold_metrics)
                
                print(f"Fold {fold + 1} SMAPE: {fold_metrics['SMAPE']:.4f}")
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {e}")
                continue
        
        # Aggregate results
        if fold_results:
            results_df = pd.DataFrame(fold_results)
            
            # Calculate mean and std for each metric
            summary = {}
            for metric in ['SMAPE', 'MAPE', 'MAE', 'MSE', 'RMSE', 'R2']:
                summary[f'{metric}_mean'] = results_df[metric].mean()
                summary[f'{metric}_std'] = results_df[metric].std()
            
            self.cv_results[model_name] = {
                'fold_results': results_df,
                'summary': summary
            }
            
            print(f"\n{model_name} CV Results:")
            print(f"SMAPE: {summary['SMAPE_mean']:.4f} ± {summary['SMAPE_std']:.4f}")
            print(f"MAE: {summary['MAE_mean']:.4f} ± {summary['MAE_std']:.4f}")
            print(f"R2: {summary['R2_mean']:.4f} ± {summary['R2_std']:.4f}")
            
            return summary
        else:
            print(f"No successful folds for {model_name}")
            return None
    
    def validate_multiple_models(self, models_dict, X, y):
        """Validate multiple models and compare results"""
        print("Starting multi-model validation...")
        
        all_results = {}
        
        for model_name, model in models_dict.items():
            try:
                result = self.validate_model(model, X, y, model_name)
                if result:
                    all_results[model_name] = result
            except Exception as e:
                print(f"Error validating {model_name}: {e}")
                continue
        
        # Create comparison DataFrame
        if all_results:
            comparison_df = pd.DataFrame(all_results).T
            comparison_df = comparison_df.sort_values('SMAPE_mean')
            
            print("\n" + "="*60)
            print("MODEL COMPARISON (sorted by SMAPE)")
            print("="*60)
            print(comparison_df[['SMAPE_mean', 'SMAPE_std', 'MAE_mean', 'R2_mean']].round(4))
            
            return comparison_df
        else:
            print("No models were successfully validated")
            return None
    
    def _clone_model(self, model):
        """Clone a model for cross-validation"""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # For custom models, create new instance
            return type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
    
    def plot_cv_results(self, save_path=None):
        """Plot cross-validation results"""
        if not self.cv_results:
            print("No CV results to plot")
            return
        
        # Create subplots
        n_models = len(self.cv_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Validation Results', fontsize=16)
        
        # Prepare data for plotting
        model_names = list(self.cv_results.keys())
        smape_means = [self.cv_results[name]['summary']['SMAPE_mean'] for name in model_names]
        smape_stds = [self.cv_results[name]['summary']['SMAPE_std'] for name in model_names]
        
        mae_means = [self.cv_results[name]['summary']['MAE_mean'] for name in model_names]
        mae_stds = [self.cv_results[name]['summary']['MAE_std'] for name in model_names]
        
        r2_means = [self.cv_results[name]['summary']['R2_mean'] for name in model_names]
        r2_stds = [self.cv_results[name]['summary']['R2_std'] for name in model_names]
        
        # SMAPE comparison
        axes[0, 0].bar(model_names, smape_means, yerr=smape_stds, capsize=5)
        axes[0, 0].set_title('SMAPE Comparison')
        axes[0, 0].set_ylabel('SMAPE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0, 1].bar(model_names, mae_means, yerr=mae_stds, capsize=5)
        axes[0, 1].set_title('MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R2 comparison
        axes[1, 0].bar(model_names, r2_means, yerr=r2_stds, capsize=5)
        axes[1, 0].set_title('R² Comparison')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # SMAPE distribution across folds
        smape_data = []
        labels = []
        for name in model_names:
            fold_results = self.cv_results[name]['fold_results']
            smape_data.append(fold_results['SMAPE'].values)
            labels.append(name)
        
        axes[1, 1].boxplot(smape_data, labels=labels)
        axes[1, 1].set_title('SMAPE Distribution Across Folds')
        axes[1, 1].set_ylabel('SMAPE')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

class HoldoutValidator:
    """Simple holdout validation"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def validate_model(self, model, X, y, model_name='Model'):
        """Perform holdout validation"""
        from sklearn.model_selection import train_test_split
        
        print(f"Holdout validation for {model_name}...")
        
        # Split data
        if isinstance(X, list):
            indices = list(range(len(X)))
            train_idx, val_idx = train_test_split(
                indices, test_size=self.test_size, random_state=self.random_state
            )
            X_train = [X[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = calculate_metrics(y_val, y_pred)
        
        print(f"{model_name} Holdout Results:")
        print(f"SMAPE: {metrics['SMAPE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"R2: {metrics['R2']:.4f}")
        
        return metrics, y_val, y_pred

def plot_predictions(y_true, y_pred, title="Predictions vs Actual", save_path=None):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Predictions vs Actual')
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = y_pred - y_true
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    
    # Distribution of residuals
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    # Q-Q plot
    plt.subplot(2, 2, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def create_validation_report(cv_results, output_path="validation_report.txt"):
    """Create a comprehensive validation report"""
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("AMAZON ML CHALLENGE - VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("-"*30 + "\n")
        
        for model_name, results in cv_results.items():
            f.write(f"\n{model_name}:\n")
            summary = results['summary']
            f.write(f"  SMAPE: {summary['SMAPE_mean']:.4f} ± {summary['SMAPE_std']:.4f}\n")
            f.write(f"  MAE:   {summary['MAE_mean']:.4f} ± {summary['MAE_std']:.4f}\n")
            f.write(f"  RMSE:  {summary['RMSE_mean']:.4f} ± {summary['RMSE_std']:.4f}\n")
            f.write(f"  R²:    {summary['R2_mean']:.4f} ± {summary['R2_std']:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Report generated successfully!\n")
    
    print(f"Validation report saved to {output_path}")
