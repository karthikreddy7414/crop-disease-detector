"""
Simplified Image Disease Classifier for crop disease identification.
A lightweight version without TensorFlow dependencies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)

class ImageDiseaseClassifier:
    """
    Simplified classifier for crop disease identification from images.
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.disease_classes = [
            'healthy', 'rust', 'powdery_mildew', 'fusarium_wilt',
            'bacterial_blight', 'late_blight', 'anthracnose',
            'leaf_spot', 'mosaic_virus', 'root_rot'
        ]
    
    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train all classification models.
        
        Args:
            X: Feature matrix
            y: Label vector
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary of model performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = model.score(X_test_scaled, y_test)
            
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            logger.info(f"{model_name} accuracy: {accuracy:.3f}")
        
        self.is_trained = True
        return results
    
    def predict_disease(self, features: np.ndarray) -> Dict:
        """
        Predict disease from image features.
        
        Args:
            features: Image feature vector
            
        Returns:
            Dictionary with predictions from all models
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Ensure correct input shape
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            # Get prediction and probability
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            
            # Get confidence score
            confidence = np.max(proba)
            
            predictions[model_name] = {
                'predicted_disease': self.disease_classes[pred],
                'confidence': confidence,
                'probabilities': dict(zip(self.disease_classes, proba))
            }
        
        # Ensemble prediction
        ensemble_pred = self._ensemble_prediction(predictions)
        
        return {
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_pred
        }
    
    def _ensemble_prediction(self, predictions: Dict) -> Dict:
        """
        Create ensemble prediction from all models.
        
        Args:
            predictions: Dictionary of individual model predictions
            
        Returns:
            Ensemble prediction
        """
        # Average probabilities across models
        avg_proba = {}
        for disease in self.disease_classes:
            proba_sum = 0
            model_count = 0
            
            for model_name, pred_data in predictions.items():
                if disease in pred_data['probabilities']:
                    proba_sum += pred_data['probabilities'][disease]
                    model_count += 1
            
            avg_proba[disease] = proba_sum / model_count if model_count > 0 else 0
        
        # Get highest probability disease
        best_disease = max(avg_proba, key=avg_proba.get)
        best_confidence = avg_proba[best_disease]
        
        return {
            'predicted_disease': best_disease,
            'confidence': best_confidence,
            'probabilities': avg_proba
        }
    
    def save_models(self, filepath: str):
        """
        Save trained models to file.
        
        Args:
            filepath: Path to save models
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'disease_classes': self.disease_classes,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """
        Load trained models from file.
        
        Args:
            filepath: Path to load models from
        """
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.disease_classes = model_data['disease_classes']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Models loaded from {filepath}")
