"""
Simplified Multimodal Fusion Model for DNA-based crop disease identification.
A lightweight version without TensorFlow dependencies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)

class MultimodalFusionModel:
    """
    Simplified multimodal fusion model that combines DNA and image data.
    """
    
    def __init__(self, dna_feature_dim: int = 100, image_feature_dim: int = 2048, 
                 num_classes: int = 10):
        self.dna_feature_dim = dna_feature_dim
        self.image_feature_dim = image_feature_dim
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Disease classes
        self.disease_classes = [
            'healthy', 'rust', 'powdery_mildew', 'fusarium_wilt',
            'bacterial_blight', 'late_blight', 'anthracnose',
            'leaf_spot', 'mosaic_virus', 'root_rot'
        ]
    
    def build_fusion_model(self) -> VotingClassifier:
        """
        Build multimodal fusion model using ensemble methods.
        
        Returns:
            Ensemble voting classifier
        """
        # Create individual models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        
        # Create voting classifier
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('svm', svm_model),
                ('mlp', mlp_model)
            ],
            voting='soft'
        )
        
        return self.model
    
    def train_model(self, dna_data: np.ndarray, image_data: np.ndarray, 
                   labels: np.ndarray, validation_split: float = 0.2) -> Dict:
        """
        Train the multimodal fusion model.
        
        Args:
            dna_data: DNA feature data
            image_data: Image feature data
            labels: Disease labels
            validation_split: Validation data split
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.model = self.build_fusion_model()
        
        # Combine DNA and image features
        combined_features = np.concatenate([dna_data, image_data], axis=1)
        
        # Split data
        split_idx = int(len(combined_features) * (1 - validation_split))
        X_train, X_val = combined_features[:split_idx], combined_features[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        val_predictions = self.model.predict(X_val_scaled)
        val_accuracy = np.mean(val_predictions == y_val)
        
        self.is_trained = True
        
        return {
            'validation_accuracy': val_accuracy,
            'validation_predictions': val_predictions.tolist(),
            'true_labels': y_val.tolist()
        }
    
    def predict_disease(self, dna_features: np.ndarray, image_features: np.ndarray) -> Dict:
        """
        Predict disease from DNA and image features.
        
        Args:
            dna_features: DNA feature vector
            image_features: Image feature vector
            
        Returns:
            Prediction results dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure correct input shapes
        if len(dna_features.shape) == 1:
            dna_features = np.expand_dims(dna_features, axis=0)
        if len(image_features.shape) == 1:
            image_features = np.expand_dims(image_features, axis=0)
        
        # Combine features
        combined_features = np.concatenate([dna_features, image_features], axis=1)
        features_scaled = self.scaler.transform(combined_features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return {
            'predicted_disease': self.disease_classes[prediction],
            'confidence': confidence,
            'all_probabilities': {
                disease: float(prob) for disease, prob in zip(self.disease_classes, probabilities)
            }
        }
    
    def evaluate_model(self, dna_data: np.ndarray, image_data: np.ndarray, 
                      labels: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            dna_data: DNA feature data
            image_data: Image feature data
            labels: True labels
            
        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Combine features
        combined_features = np.concatenate([dna_data, image_data], axis=1)
        features_scaled = self.scaler.transform(combined_features)
        
        # Get predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == labels)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(labels, predictions, target_names=self.disease_classes),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'disease_classes': self.disease_classes,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Fusion model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model.
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.disease_classes = model_data['disease_classes']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Fusion model loaded from {filepath}")
