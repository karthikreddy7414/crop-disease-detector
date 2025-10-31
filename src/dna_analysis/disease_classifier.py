"""
DNA-based Disease Classifier for crop disease identification.
Uses genetic markers and sequence analysis to classify diseases.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

logger = logging.getLogger(__name__)

class DNADiseaseClassifier:
    """
    Classifies crop diseases based on DNA sequence analysis.
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.disease_classes = []
        
        # Disease-specific genetic markers
        self.disease_markers = {
            'rust': {
                'resistance_genes': ['Lr34', 'Lr67', 'Lr68'],
                'susceptibility_markers': ['S1', 'S2'],
                'qtl_regions': ['QTL_rust_1', 'QTL_rust_2']
            },
            'powdery_mildew': {
                'resistance_genes': ['Pm1', 'Pm2', 'Pm3'],
                'susceptibility_markers': ['S3', 'S4'],
                'qtl_regions': ['QTL_pm_1', 'QTL_pm_2']
            },
            'fusarium_wilt': {
                'resistance_genes': ['Foc1', 'Foc2', 'Foc3'],
                'susceptibility_markers': ['S5', 'S6'],
                'qtl_regions': ['QTL_fus_1', 'QTL_fus_2']
            },
            'bacterial_blight': {
                'resistance_genes': ['Xa1', 'Xa2', 'Xa3'],
                'susceptibility_markers': ['S7', 'S8'],
                'qtl_regions': ['QTL_bb_1', 'QTL_bb_2']
            },
            'late_blight': {
                'resistance_genes': ['R1', 'R2', 'R3'],
                'susceptibility_markers': ['S9', 'S10'],
                'qtl_regions': ['QTL_lb_1', 'QTL_lb_2']
            }
        }
    
    def prepare_training_data(self, sequences: List[Dict], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from DNA sequences and labels.
        
        Args:
            sequences: List of sequence dictionaries with features
            labels: List of disease labels
            
        Returns:
            Tuple of (features, labels) arrays
        """
        # Extract features from sequences
        features = []
        for seq_data in sequences:
            feature_vector = self._extract_features(seq_data)
            features.append(feature_vector)
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Store feature names for later use
        self.feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        self.disease_classes = list(set(labels))
        
        return features, labels
    
    def _extract_features(self, seq_data: Dict) -> np.ndarray:
        """
        Extract features from sequence data.
        
        Args:
            seq_data: Dictionary containing sequence information
            
        Returns:
            Feature vector
        """
        features = []
        
        # Basic sequence features
        features.extend([
            seq_data.get('length', 0),
            seq_data.get('gc_content', 0),
            seq_data.get('at_content', 0),
            seq_data.get('complexity', 0)
        ])
        
        # Genetic marker features
        for disease, markers in self.disease_markers.items():
            for marker_type, marker_list in markers.items():
                for marker in marker_list:
                    features.append(seq_data.get(f'{disease}_{marker_type}_{marker}', 0))
        
        # SNP features
        snp_features = seq_data.get('snp_features', {})
        for snp_id in snp_features:
            features.append(snp_features[snp_id])
        
        # QTL features
        qtl_features = seq_data.get('qtl_features', {})
        for qtl_id in qtl_features:
            features.append(qtl_features[qtl_id])
        
        return np.array(features)
    
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
    
    def predict_disease(self, sequence_data: Dict) -> Dict:
        """
        Predict disease from sequence data.
        
        Args:
            sequence_data: Dictionary containing sequence information
            
        Returns:
            Dictionary with predictions from all models
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Extract features
        features = self._extract_features(sequence_data)
        features_scaled = self.scaler.transform([features])
        
        predictions = {}
        
        for model_name, model in self.models.items():
            # Get prediction and probability
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            
            # Get confidence score
            confidence = np.max(proba)
            
            predictions[model_name] = {
                'predicted_disease': pred,
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
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict:
        """
        Get feature importance from specified model.
        
        Args:
            model_name: Name of model to get importance from
            
        Returns:
            Dictionary of feature importance
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before getting feature importance")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            return feature_importance
        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return {}
