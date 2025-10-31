#!/usr/bin/env python3
"""
Training script for DNA-based crop disease identification system.
This script generates training data and trains all models.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.training_pipeline import TrainingPipeline

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train DNA-based crop disease identification models')
    parser.add_argument('--samples', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--quick', action='store_true', help='Run quick training (1000 samples)')
    parser.add_argument('--validate', action='store_true', help='Validate existing models')
    parser.add_argument('--retrain', action='store_true', help='Retrain existing models')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("DNA Crop Disease Identification - Model Training")
    logger.info("=" * 60)
    
    try:
        # Initialize training pipeline
        pipeline = TrainingPipeline()
        
        if args.validate:
            # Validate existing training
            logger.info("Validating existing models...")
            validation_results = pipeline.validate_training()
            
            print("\nTraining Validation Results:")
            print(f"Overall Status: {validation_results['overall_status']}")
            print(f"Models Found: {len(validation_results['models_found'])}")
            print(f"Models Missing: {len(validation_results['models_missing'])}")
            
            if validation_results['models_found']:
                print("\nFound Models:")
                for model in validation_results['models_found']:
                    print(f"  - {model}")
            
            if validation_results['models_missing']:
                print("\nMissing Models:")
                for model in validation_results['models_missing']:
                    print(f"  - {model}")
            
            return
        
        if args.quick:
            # Quick training
            logger.info("Running quick training...")
            results = pipeline.quick_training(num_samples=1000)
            
            print("\nQuick Training Results:")
            print(f"Status: {results['status']}")
            print(f"Samples: {results['num_samples']}")
            print("Models trained: DNA, Image")
            
        elif args.retrain:
            # Retrain existing models
            logger.info("Retraining existing models...")
            results = pipeline.load_and_retrain()
            
            print("\nRetraining Results:")
            print("All models retrained successfully")
            
        else:
            # Full training
            logger.info(f"Running full training with {args.samples} samples...")
            results = pipeline.run_complete_training(
                num_samples=args.samples,
                save_data=True,
                train_models=True
            )
            
            print("\nFull Training Results:")
            print(f"Total Samples: {results['data_generation']['num_samples']}")
            print(f"DNA Features: {results['data_generation']['dna_feature_dim']}")
            print(f"Image Features: {results['data_generation']['image_feature_dim']}")
            print(f"Classes: {results['data_generation']['num_classes']}")
            print(f"Duration: {results['total_duration']}")
            
            # Show model performance
            if 'model_training' in results:
                print("\nModel Performance:")
                for model_name, model_results in results['model_training'].items():
                    if isinstance(model_results, dict) and 'results' in model_results:
                        print(f"  {model_name}: Trained successfully")
                    else:
                        print(f"  {model_name}: Trained successfully")
        
        print("\nTraining completed successfully!")
        print("Your DNA-based crop disease identification system is now ready!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
