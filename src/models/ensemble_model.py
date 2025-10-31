"""
Ensemble Disease Classifier for DNA-based crop disease identification.
Combines multiple models for improved accuracy and robustness.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)

class EnsembleDiseaseClassifier:
    """
    Ensemble classifier that combines multiple models for disease identification.
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.is_trained = False
        self.disease_classes = [
            'healthy', 'rust', 'powdery_mildew', 'fusarium_wilt',
            'bacterial_blight', 'late_blight', 'anthracnose',
            'leaf_spot', 'mosaic_virus', 'root_rot'
        ]
    
    def add_model(self, name: str, model, model_type: str = 'classifier'):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Model object
            model_type: Type of model ('classifier', 'regressor')
        """
        self.models[name] = {
            'model': model,
            'type': model_type
        }
        logger.info(f"Added model: {name}")
    
    def build_ensemble(self, voting_method: str = 'soft') -> VotingClassifier:
        """
        Build ensemble model using voting classifier.
        
        Args:
            voting_method: Voting method ('hard' or 'soft')
            
        Returns:
            Ensemble voting classifier
        """
        if not self.models:
            raise ValueError("No models added to ensemble")
        
        # Prepare estimators for voting
        estimators = []
        for name, model_info in self.models.items():
            if model_info['type'] == 'classifier':
                estimators.append((name, model_info['model']))
        
        if not estimators:
            raise ValueError("No classifier models found")
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting=voting_method
        )
        
        return self.ensemble_model
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                      validation_split: float = 0.2) -> Dict:
        """
        Train the ensemble model.
        
        Args:
            X: Feature matrix
            y: Label vector
            validation_split: Validation data split
            
        Returns:
            Training results dictionary
        """
        if self.ensemble_model is None:
            self.ensemble_model = self.build_ensemble()
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_predictions = self.ensemble_model.predict(X_val)
        val_accuracy = np.mean(val_predictions == y_val)
        
        self.is_trained = True
        
        return {
            'validation_accuracy': val_accuracy,
            'validation_predictions': val_predictions.tolist(),
            'true_labels': y_val.tolist()
        }
    
    def predict_disease(self, features: np.ndarray) -> Dict:
        """
        Predict disease using ensemble model.
        
        Args:
            features: Feature vector
            
        Returns:
            Prediction results dictionary
        """
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making predictions")
        
        # Ensure correct input shape
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        
        # Get prediction
        prediction = self.ensemble_model.predict(features)[0]
        
        # Get prediction probabilities if available
        if hasattr(self.ensemble_model, 'predict_proba'):
            probabilities = self.ensemble_model.predict_proba(features)[0]
            confidence = np.max(probabilities)
        else:
            probabilities = None
            confidence = 1.0
        
        return {
            'predicted_disease': self.disease_classes[prediction],
            'confidence': confidence,
            'all_probabilities': {
                disease: float(prob) for disease, prob in zip(self.disease_classes, probabilities)
            } if probabilities is not None else None
        }
    
    def predict_with_individual_models(self, features: np.ndarray) -> Dict:
        """
        Get predictions from individual models in the ensemble.
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary with individual model predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making predictions")
        
        # Ensure correct input shape
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        
        individual_predictions = {}
        
        for name, model_info in self.models.items():
            if model_info['type'] == 'classifier':
                try:
                    prediction = model_info['model'].predict(features)[0]
                    individual_predictions[name] = {
                        'prediction': self.disease_classes[prediction],
                        'prediction_index': int(prediction)
                    }
                    
                    # Get probabilities if available
                    if hasattr(model_info['model'], 'predict_proba'):
                        probabilities = model_info['model'].predict_proba(features)[0]
                        individual_predictions[name]['probabilities'] = {
                            disease: float(prob) for disease, prob in zip(self.disease_classes, probabilities)
                        }
                        individual_predictions[name]['confidence'] = float(np.max(probabilities))
                    else:
                        individual_predictions[name]['confidence'] = 1.0
                        
                except Exception as e:
                    logger.warning(f"Error getting prediction from {name}: {e}")
                    individual_predictions[name] = {'error': str(e)}
        
        return individual_predictions
    
    def evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate ensemble model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before evaluation")
        
        # Get predictions
        predictions = self.ensemble_model.predict(X)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y)
        
        # Get probabilities if available
        if hasattr(self.ensemble_model, 'predict_proba'):
            probabilities = self.ensemble_model.predict_proba(X)
            confidence_scores = np.max(probabilities, axis=1)
            avg_confidence = np.mean(confidence_scores)
        else:
            avg_confidence = 1.0
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'classification_report': classification_report(y, predictions, target_names=self.disease_classes),
            'confusion_matrix': confusion_matrix(y, predictions).tolist()
        }
        
        return metrics
    
    def get_model_weights(self) -> Dict:
        """
        Get weights of individual models in the ensemble.
        
        Returns:
            Dictionary of model weights
        """
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before getting weights")
        
        weights = {}
        for name, model_info in self.models.items():
            if hasattr(model_info['model'], 'feature_importances_'):
                weights[name] = model_info['model'].feature_importances_.tolist()
            else:
                weights[name] = None
        
        return weights
    
    def save_ensemble(self, filepath: str):
        """
        Save ensemble model.
        
        Args:
            filepath: Path to save ensemble
        """
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before saving")
        
        ensemble_data = {
            'ensemble_model': self.ensemble_model,
            'models': self.models,
            'disease_classes': self.disease_classes,
            'is_trained': self.is_trained
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """
        Load ensemble model.
        
        Args:
            filepath: Path to load ensemble from
        """
        ensemble_data = joblib.load(filepath)
        
        self.ensemble_model = ensemble_data['ensemble_model']
        self.models = ensemble_data['models']
        self.disease_classes = ensemble_data['disease_classes']
        self.is_trained = ensemble_data['is_trained']
        
        logger.info(f"Ensemble model loaded from {filepath}")
    
    def add_bagging_model(self, base_model, n_estimators: int = 10, 
                         random_state: int = 42) -> BaggingClassifier:
        """
        Add bagging ensemble model.
        
        Args:
            base_model: Base model for bagging
            n_estimators: Number of estimators
            random_state: Random state for reproducibility
            
        Returns:
            Bagging classifier
        """
        bagging_model = BaggingClassifier(
            base_estimator=base_model,
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        self.add_model(f'bagging_{base_model.__class__.__name__}', bagging_model)
        
        return bagging_model
    
    def get_ensemble_confidence(self, features: np.ndarray) -> Dict:
        """
        Get confidence scores from ensemble prediction.
        
        Args:
            features: Feature vector
            
        Returns:
            Confidence analysis dictionary
        """
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before getting confidence")
        
        # Get individual model predictions
        individual_predictions = self.predict_with_individual_models(features)
        
        # Calculate agreement between models
        predictions = [pred['prediction_index'] for pred in individual_predictions.values() 
                      if 'prediction_index' in pred]
        
        if not predictions:
            return {'error': 'No valid predictions from individual models'}
        
        # Calculate consensus
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        consensus_score = np.max(counts) / len(predictions)
        
        # Get most common prediction
        most_common_idx = unique_predictions[np.argmax(counts)]
        most_common_disease = self.disease_classes[most_common_idx]
        
        return {
            'consensus_score': consensus_score,
            'most_common_prediction': most_common_disease,
            'prediction_distribution': {
                self.disease_classes[idx]: int(count) for idx, count in zip(unique_predictions, counts)
            },
            'individual_predictions': individual_predictions
        }
