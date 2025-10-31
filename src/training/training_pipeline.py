"""
Complete training pipeline for DNA-based crop disease identification system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os
import json
from datetime import datetime

from .data_generator import DataGenerator
from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Complete training pipeline for the DNA-based crop disease identification system.
    """
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize training pipeline.
        
        Args:
            data_dir: Directory for training data
            models_dir: Directory for trained models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize components
        self.data_generator = DataGenerator()
        self.model_trainer = ModelTrainer(models_dir)
        
        # Training configuration
        self.config = {
            'num_samples': 5000,
            'test_size': 0.2,
            'validation_split': 0.2,
            'random_seed': 42
        }
    
    def run_complete_training(self, num_samples: int = 5000, 
                            save_data: bool = True, 
                            train_models: bool = True) -> Dict:
        """
        Run the complete training pipeline.
        
        Args:
            num_samples: Number of training samples to generate
            save_data: Whether to save generated data
            train_models: Whether to train models
            
        Returns:
            Complete training results
        """
        logger.info("Starting complete training pipeline...")
        start_time = datetime.now()
        
        # Update configuration
        self.config['num_samples'] = num_samples
        
        results = {
            'pipeline_start_time': start_time.isoformat(),
            'configuration': self.config,
            'data_generation': {},
            'model_training': {},
            'evaluation': {}
        }
        
        try:
            # Step 1: Generate training data
            logger.info("Step 1: Generating training data...")
            dna_features, image_features, labels, disease_names = self.data_generator.generate_training_data(num_samples)
            
            # Ensure JSON-serializable types (avoid numpy types as dict keys/values)
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            class_distribution = {int(k): int(v) for k, v in zip(unique_labels.tolist(), label_counts.tolist())}

            results['data_generation'] = {
                'num_samples': int(len(labels)),
                'dna_feature_dim': int(dna_features.shape[1]),
                'image_feature_dim': int(image_features.shape[1]),
                'num_classes': int(len(np.unique(labels))),
                'class_distribution': class_distribution
            }
            
            if save_data:
                # Save generated data
                data_path = os.path.join(self.data_dir, "training")
                self.data_generator.save_training_data(
                    dna_features, image_features, labels, disease_names, data_path
                )
                results['data_generation']['saved_to'] = data_path
            
            # Step 2: Train models
            if train_models:
                logger.info("Step 2: Training models...")
                training_results = self.model_trainer.train_all_models(
                    dna_features, image_features, labels
                )
                results['model_training'] = training_results
            
            # Step 3: Evaluate models
            logger.info("Step 3: Evaluating models...")
            evaluation_results = self.model_trainer.evaluate_models(
                dna_features, image_features, labels
            )
            results['evaluation'] = evaluation_results
            
            # Step 4: Generate training report
            logger.info("Step 4: Generating training report...")
            report = self.generate_training_report(results)
            results['training_report'] = report
            
            # Save complete results
            results_path = os.path.join(self.models_dir, "complete_training_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            end_time = datetime.now()
            results['pipeline_end_time'] = end_time.isoformat()
            results['total_duration'] = str(end_time - start_time)
            
            logger.info(f"Training pipeline completed successfully in {end_time - start_time}")
            logger.info(f"Results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            results['error'] = str(e)
            raise
        
        return results
    
    def generate_training_report(self, results: Dict) -> Dict:
        """
        Generate comprehensive training report.
        
        Args:
            results: Training results dictionary
            
        Returns:
            Training report
        """
        report = {
            'summary': {
                'total_samples': results['data_generation']['num_samples'],
                'dna_features': results['data_generation']['dna_feature_dim'],
                'image_features': results['data_generation']['image_feature_dim'],
                'num_classes': results['data_generation']['num_classes']
            },
            'model_performance': {},
            'recommendations': []
        }
        
        # Extract model performance
        if 'model_training' in results:
            for model_name, model_results in results['model_training'].items():
                if isinstance(model_results, dict) and 'results' in model_results:
                    report['model_performance'][model_name] = {
                        'status': 'trained',
                        'details': model_results['results']
                    }
                else:
                    report['model_performance'][model_name] = {
                        'status': 'trained',
                        'details': model_results
                    }
        
        # Generate recommendations
        if report['model_performance']:
            report['recommendations'].append("All models trained successfully")
            report['recommendations'].append("System ready for deployment")
            report['recommendations'].append("Consider fine-tuning with real-world data")
        else:
            report['recommendations'].append("Models need to be trained")
            report['recommendations'].append("Generate more diverse training data")
        
        return report
    
    def quick_training(self, num_samples: int = 1000) -> Dict:
        """
        Quick training for testing purposes.
        
        Args:
            num_samples: Number of samples for quick training
            
        Returns:
            Training results
        """
        logger.info(f"Running quick training with {num_samples} samples...")
        
        # Generate smaller dataset
        dna_features, image_features, labels, disease_names = self.data_generator.generate_training_data(num_samples)
        
        # Train only essential models
        dna_results = self.model_trainer.train_dna_model(dna_features, labels)
        image_results = self.model_trainer.train_image_model(image_features, labels)
        
        return {
            'quick_training': True,
            'num_samples': num_samples,
            'dna_model': dna_results,
            'image_model': image_results,
            'status': 'completed'
        }
    
    def load_and_retrain(self, data_path: str = None) -> Dict:
        """
        Load existing data and retrain models.
        
        Args:
            data_path: Path to existing training data
            
        Returns:
            Retraining results
        """
        if data_path is None:
            data_path = os.path.join(self.data_dir, "training")
        
        logger.info(f"Loading data from {data_path}...")
        
        # Load existing data
        dna_features, image_features, labels, metadata = self.data_generator.load_training_data(data_path)
        
        # Retrain models
        results = self.model_trainer.train_all_models(dna_features, image_features, labels)
        
        logger.info("Retraining completed")
        return results
    
    def validate_training(self) -> Dict:
        """
        Validate that all models are properly trained and saved.
        
        Returns:
            Validation results
        """
        logger.info("Validating training...")
        
        validation_results = {
            'models_found': [],
            'models_missing': [],
            'data_files': [],
            'overall_status': 'unknown'
        }
        
        # Check for model files
        model_files = [
            'dna_classifier.joblib',
            'image_classifier.joblib',
            'fusion_model.joblib',
            'ensemble_model.joblib',
            'ensemble_scaler.joblib'
        ]
        
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                validation_results['models_found'].append(model_file)
            else:
                validation_results['models_missing'].append(model_file)
        
        # Check for data files
        data_files = [
            'dna_features.npy',
            'image_features.npy',
            'labels.npy',
            'metadata.json'
        ]
        
        data_path = os.path.join(self.data_dir, "training")
        for data_file in data_files:
            file_path = os.path.join(data_path, data_file)
            if os.path.exists(file_path):
                validation_results['data_files'].append(data_file)
        
        # Determine overall status
        if len(validation_results['models_found']) == len(model_files):
            validation_results['overall_status'] = 'complete'
        elif len(validation_results['models_found']) > 0:
            validation_results['overall_status'] = 'partial'
        else:
            validation_results['overall_status'] = 'none'
        
        logger.info(f"Training validation completed: {validation_results['overall_status']}")
        return validation_results
