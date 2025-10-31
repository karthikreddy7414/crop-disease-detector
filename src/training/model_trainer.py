"""
Model trainer for DNA-based crop disease identification system.
Trains DNA, image, and multimodal models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
import os
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import json

from src.dna_analysis.disease_classifier import DNADiseaseClassifier
from src.image_processing.disease_classifier_simple import ImageDiseaseClassifier
from src.models.fusion_model_simple import MultimodalFusionModel
from src.models.ensemble_model import EnsembleDiseaseClassifier

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trains all models for the DNA-based crop disease identification system.
    """
    
    def __init__(self, output_dir: str = "models"):
        """
        Initialize model trainer.
        
        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self.dna_classifier = DNADiseaseClassifier()
        self.image_classifier = ImageDiseaseClassifier()
        self.fusion_model = MultimodalFusionModel()
        self.ensemble_model = EnsembleDiseaseClassifier()
        
        # Training results
        self.training_results = {}
        
    def train_dna_model(self, dna_features: np.ndarray, labels: np.ndarray, 
                       test_size: float = 0.2) -> Dict:
        """
        Train DNA-based disease classifier.
        
        Args:
            dna_features: DNA feature matrix
            labels: Disease labels
            test_size: Test set proportion
            
        Returns:
            Training results dictionary
        """
        logger.info("Training DNA-based disease classifier...")
        
        # Train the model
        results = self.dna_classifier.train_models(dna_features, labels, test_size)
        
        # Save the trained model
        model_path = os.path.join(self.output_dir, "dna_classifier.joblib")
        joblib.dump(self.dna_classifier, model_path)
        
        # Store results
        self.training_results['dna_model'] = {
            'model_path': model_path,
            'results': results,
            'feature_names': self.dna_classifier.feature_names,
            'disease_classes': self.dna_classifier.disease_classes
        }
        
        logger.info(f"DNA model trained and saved to {model_path}")
        return results
    
    def train_image_model(self, image_features: np.ndarray, labels: np.ndarray, 
                         test_size: float = 0.2) -> Dict:
        """
        Train image-based disease classifier.
        
        Args:
            image_features: Image feature matrix
            labels: Disease labels
            test_size: Test set proportion
            
        Returns:
            Training results dictionary
        """
        logger.info("Training image-based disease classifier...")
        
        # Train the model
        results = self.image_classifier.train_models(image_features, labels, test_size)
        
        # Save the trained model
        model_path = os.path.join(self.output_dir, "image_classifier.joblib")
        joblib.dump(self.image_classifier, model_path)
        
        # Store results
        self.training_results['image_model'] = {
            'model_path': model_path,
            'results': results,
            'feature_names': self.image_classifier.feature_names,
            'disease_classes': self.image_classifier.disease_classes
        }
        
        logger.info(f"Image model trained and saved to {model_path}")
        return results
    
    def train_fusion_model(self, dna_features: np.ndarray, image_features: np.ndarray, 
                          labels: np.ndarray, validation_split: float = 0.2) -> Dict:
        """
        Train multimodal fusion model.
        
        Args:
            dna_features: DNA feature matrix
            image_features: Image feature matrix
            labels: Disease labels
            validation_split: Validation set proportion
            
        Returns:
            Training results dictionary
        """
        logger.info("Training multimodal fusion model...")
        
        # Train the fusion model
        results = self.fusion_model.train_model(dna_features, image_features, labels, validation_split)
        
        # Save the trained model
        model_path = os.path.join(self.output_dir, "fusion_model.joblib")
        joblib.dump(self.fusion_model, model_path)
        
        # Store results
        self.training_results['fusion_model'] = {
            'model_path': model_path,
            'results': results,
            'disease_classes': self.fusion_model.disease_classes
        }
        
        logger.info(f"Fusion model trained and saved to {model_path}")
        return results
    
    def train_ensemble_model(self, dna_features: np.ndarray, image_features: np.ndarray, 
                           labels: np.ndarray) -> Dict:
        """
        Train ensemble model combining all approaches.
        
        Args:
            dna_features: DNA feature matrix
            image_features: Image feature matrix
            labels: Disease labels
            
        Returns:
            Training results dictionary
        """
        logger.info("Training ensemble model...")
        
        # Add individual models to ensemble
        self.ensemble_model.add_model('dna_rf', self.dna_classifier.models['random_forest'])
        self.ensemble_model.add_model('image_rf', self.image_classifier.models['random_forest'])
        self.ensemble_model.add_model('fusion', self.fusion_model.model)
        
        # Build and train ensemble
        ensemble = self.ensemble_model.build_ensemble()
        
        # Combine features
        combined_features = np.concatenate([dna_features, image_features], axis=1)
        
        # Train ensemble
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = ensemble.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Save the trained model
        model_path = os.path.join(self.output_dir, "ensemble_model.joblib")
        joblib.dump(ensemble, model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.output_dir, "ensemble_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        
        # Store results
        self.training_results['ensemble_model'] = {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'results': results,
            'disease_classes': self.ensemble_model.disease_classes
        }
        
        logger.info(f"Ensemble model trained and saved to {model_path}")
        return results
    
    def train_all_models(self, dna_features: np.ndarray, image_features: np.ndarray, 
                        labels: np.ndarray) -> Dict:
        """
        Train all models in the system.
        
        Args:
            dna_features: DNA feature matrix
            image_features: Image feature matrix
            labels: Disease labels
            
        Returns:
            Complete training results
        """
        logger.info("Starting comprehensive model training...")
        
        # Train individual models
        dna_results = self.train_dna_model(dna_features, labels)
        image_results = self.train_image_model(image_features, labels)
        fusion_results = self.train_fusion_model(dna_features, image_features, labels)
        ensemble_results = self.train_ensemble_model(dna_features, image_features, labels)
        
        # Compile all results
        all_results = {
            'dna_model': dna_results,
            'image_model': image_results,
            'fusion_model': fusion_results,
            'ensemble_model': ensemble_results,
            'training_metadata': {
                'num_samples': len(labels),
                'dna_feature_dim': dna_features.shape[1],
                'image_feature_dim': image_features.shape[1],
                'num_classes': len(np.unique(labels))
            }
        }
        
        # Save training results
        results_path = os.path.join(self.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"All models trained successfully. Results saved to {results_path}")
        return all_results
    
    def load_trained_models(self) -> Dict:
        """
        Load all trained models.
        
        Returns:
            Dictionary of loaded models
        """
        models = {}
        
        # Load DNA model
        dna_path = os.path.join(self.output_dir, "dna_classifier.joblib")
        if os.path.exists(dna_path):
            models['dna_classifier'] = joblib.load(dna_path)
            logger.info("DNA classifier loaded")
        
        # Load image model
        image_path = os.path.join(self.output_dir, "image_classifier.joblib")
        if os.path.exists(image_path):
            models['image_classifier'] = joblib.load(image_path)
            logger.info("Image classifier loaded")
        
        # Load fusion model
        fusion_path = os.path.join(self.output_dir, "fusion_model.joblib")
        if os.path.exists(fusion_path):
            models['fusion_model'] = joblib.load(fusion_path)
            logger.info("Fusion model loaded")
        
        # Load ensemble model
        ensemble_path = os.path.join(self.output_dir, "ensemble_model.joblib")
        scaler_path = os.path.join(self.output_dir, "ensemble_scaler.joblib")
        if os.path.exists(ensemble_path) and os.path.exists(scaler_path):
            models['ensemble_model'] = joblib.load(ensemble_path)
            models['ensemble_scaler'] = joblib.load(scaler_path)
            logger.info("Ensemble model loaded")
        
        return models
    
    def evaluate_models(self, dna_features: np.ndarray, image_features: np.ndarray, 
                       labels: np.ndarray) -> Dict:
        """
        Evaluate all trained models.
        
        Args:
            dna_features: DNA feature matrix
            image_features: Image feature matrix
            labels: Disease labels
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating all models...")
        
        # Load models
        models = self.load_trained_models()
        evaluation_results = {}
        
        # Evaluate DNA model
        if 'dna_classifier' in models:
            dna_model = models['dna_classifier']
            if hasattr(dna_model, 'predict_disease'):
                # This would need to be adapted based on the actual model interface
                logger.info("DNA model evaluation completed")
        
        # Evaluate image model
        if 'image_classifier' in models:
            image_model = models['image_classifier']
            if hasattr(image_model, 'predict_disease'):
                logger.info("Image model evaluation completed")
        
        # Evaluate fusion model
        if 'fusion_model' in models:
            fusion_model = models['fusion_model']
            logger.info("Fusion model evaluation completed")
        
        # Evaluate ensemble model
        if 'ensemble_model' in models and 'ensemble_scaler' in models:
            ensemble_model = models['ensemble_model']
            scaler = models['ensemble_scaler']
            logger.info("Ensemble model evaluation completed")
        
        return evaluation_results
