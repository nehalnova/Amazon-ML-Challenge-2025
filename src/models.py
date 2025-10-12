import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import catboost as cb
import joblib
import os
from torchvision import models, transforms
from PIL import Image

class SimpleTextFeatureExtractor:
    """Simple text feature extractor using TF-IDF"""
    
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.is_fitted = False
        
    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
        
    def transform(self, texts):
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

class SimpleTextEncoder(nn.Module):
    """Simple text encoder using TF-IDF features"""
    
    def __init__(self, input_dim=3000, hidden_dim=256):
        super(SimpleTextEncoder, self).__init__()
        self.text_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, text_features):
        return self.text_fc(text_features)

class ImageEncoder(nn.Module):
    """Image encoder using pre-trained CNN models"""
    
    def __init__(self, model_name='resnet50', hidden_dim=256, pretrained=True):
        super(ImageEncoder, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Identity()  # Remove final classification layer
            backbone_dim = 2048
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier = nn.Identity()
            backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Additional layers for image processing
        self.image_fc = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def forward(self, images):
        # Extract features using backbone
        features = self.backbone(images)
        
        # Pass through additional layers
        image_features = self.image_fc(features)
        return image_features

class MultimodalPricingModel(nn.Module):
    """Multimodal model combining text and image features for price prediction"""
    
    def __init__(self, text_input_dim=3000, image_model_name='resnet50', hidden_dim=256):
        super(MultimodalPricingModel, self).__init__()
        
        self.text_encoder = SimpleTextEncoder(text_input_dim, hidden_dim)
        self.image_encoder = ImageEncoder(image_model_name, hidden_dim)
        
        # Fusion layer
        fusion_input_dim = (hidden_dim // 2) * 2  # Text + Image features
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)  # Price prediction
        )
        
    def forward(self, text_features, images):
        # Encode text and images
        text_encoded = self.text_encoder(text_features)
        image_encoded = self.image_encoder(images)
        
        # Concatenate features
        combined_features = torch.cat([text_encoded, image_encoded], dim=1)
        
        # Predict price
        price_pred = self.fusion_layer(combined_features)
        return price_pred.squeeze()

class TextOnlyModel(nn.Module):
    """Text-only model for price prediction"""
    
    def __init__(self, text_input_dim=3000, hidden_dim=512):
        super(TextOnlyModel, self).__init__()
        
        self.text_encoder = SimpleTextEncoder(text_input_dim, hidden_dim)
        
        # Prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 8, 1)
        )
        
    def forward(self, text_features):
        text_encoded = self.text_encoder(text_features)
        price_pred = self.predictor(text_encoded)
        return price_pred.squeeze()

class ImageOnlyModel(nn.Module):
    """Image-only model for price prediction"""
    
    def __init__(self, model_name='resnet50', hidden_dim=512):
        super(ImageOnlyModel, self).__init__()
        
        self.image_encoder = ImageEncoder(model_name, hidden_dim)
        
        # Prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 8, 1)
        )
        
    def forward(self, images):
        image_features = self.image_encoder(images)
        price_pred = self.predictor(image_features)
        return price_pred.squeeze()

class TraditionalMLWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for traditional ML models with TF-IDF feature extraction"""
    
    def __init__(self, model_type='xgboost', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.text_encoder = SimpleTextFeatureExtractor(max_features=3000)
        
    def _initialize_model(self):
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=self.kwargs.get('n_estimators', 500),
                max_depth=self.kwargs.get('max_depth', 6),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'catboost':
            self.model = cb.CatBoostRegressor(
                iterations=self.kwargs.get('iterations', 500),
                depth=self.kwargs.get('depth', 6),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                random_state=42,
                verbose=False
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=self.kwargs.get('n_estimators', 300),
                max_depth=self.kwargs.get('max_depth', 12),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'ridge':
            self.model = Ridge(
                alpha=self.kwargs.get('alpha', 1.0),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X, y):
        """Fit the model with TF-IDF features"""
        if self.model is None:
            self._initialize_model()
        
        # Extract text features using TF-IDF
        text_features = self.text_encoder.fit_transform(X)
        
        # Fit the model
        self.model.fit(text_features, y)
        return self
    
    def predict(self, X):
        """Predict using TF-IDF features"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Extract text features
        text_features = self.text_encoder.transform(X)
        
        # Make predictions
        predictions = self.model.predict(text_features)
        return predictions
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'text_encoder': self.text_encoder,
            'model_type': self.model_type
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.text_encoder = saved_data['text_encoder']
        self.model_type = saved_data['model_type']

def get_image_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

class DeepLearningWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for PyTorch deep learning models"""
    
    def __init__(self, model_type='text_only', epochs=50, batch_size=32, learning_rate=0.001):
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.text_encoder = SimpleTextFeatureExtractor(max_features=3000)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actual_text_dim = None
        
    def _initialize_model(self, text_dim):
        if self.model_type == 'text_only':
            self.model = TextOnlyModel(text_input_dim=text_dim, hidden_dim=256)
        elif self.model_type == 'multimodal':
            self.model = MultimodalPricingModel(text_input_dim=text_dim, hidden_dim=256)
        elif self.model_type == 'image_only':
            self.model = ImageOnlyModel(model_name='resnet50', hidden_dim=256)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        
    def fit(self, X, y, X_images=None):
        """Fit the deep learning model"""
        # Extract text features first to get actual dimensions
        text_features = self.text_encoder.fit_transform(X)
        self.actual_text_dim = text_features.shape[1]
        
        if self.model is None:
            self._initialize_model(self.actual_text_dim)
        
        # Convert to tensors
        text_tensor = torch.FloatTensor(text_features).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(text_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize optimizer and criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_text, batch_y in dataloader:
                optimizer.zero_grad()
                
                if self.model_type == 'text_only':
                    predictions = self.model(batch_text)
                elif self.model_type == 'multimodal':
                    # For multimodal without images, use dummy image tensor
                    dummy_images = torch.zeros(batch_text.size(0), 3, 224, 224).to(self.device)
                    predictions = self.model(batch_text, dummy_images)
                else:
                    predictions = self.model(batch_text)
                
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}")
        
        return self
    
    def predict(self, X, X_images=None):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Extract text features
        text_features = self.text_encoder.transform(X)
        text_tensor = torch.FloatTensor(text_features).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'text_only':
                predictions = self.model(text_tensor)
            elif self.model_type == 'multimodal':
                # For multimodal without images, use dummy image tensor
                dummy_images = torch.zeros(text_tensor.size(0), 3, 224, 224).to(self.device)
                predictions = self.model(text_tensor, dummy_images)
            else:
                predictions = self.model(text_tensor)
        
        return predictions.cpu().numpy()
