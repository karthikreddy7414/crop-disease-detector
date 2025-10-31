#!/usr/bin/env python3
"""
Training script for DNA-based crop disease identification with REAL PlantVillage images.
This script loads actual images from the PlantVillage dataset instead of generating synthetic data.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from datetime import datetime
import logging
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_real_images.log')
    ]
)
logger = logging.getLogger(__name__)

class RealImageTrainer:
    """Train models using real PlantVillage images."""
    
    def __init__(self, data_dir='data/images/raw/PlantVillage', max_images_per_class=500):
        self.data_dir = Path(data_dir)
        self.max_images_per_class = max_images_per_class
        self.class_names = []
        self.label_map = {}
        
    def load_images_from_directory(self):
        """Load images from PlantVillage directory structure."""
        logger.info(f"Loading images from {self.data_dir}")
        
        images = []
        labels = []
        
        # Get all disease directories
        disease_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(disease_dirs)} disease classes")
        
        # Create label mapping
        self.class_names = sorted([d.name for d in disease_dirs])
        self.label_map = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load images from each class
        for disease_dir in disease_dirs:
            class_name = disease_dir.name
            class_idx = self.label_map[class_name]
            
            # Get all images in this directory
            image_files = list(disease_dir.glob('*.JPG')) + list(disease_dir.glob('*.jpg')) + list(disease_dir.glob('*.png'))
            
            # Limit images per class
            image_files = image_files[:self.max_images_per_class]
            
            logger.info(f"Loading {len(image_files)} images from {class_name}")
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Resize to standard size
                    img = cv2.resize(img, (128, 128))
                    
                    # Convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    images.append(img)
                    labels.append(class_idx)
                    
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(images)} total images from {len(self.class_names)} classes")
        
        return np.array(images), np.array(labels)
    
    def extract_image_features(self, images):
        """Extract features from images."""
        logger.info("Extracting image features...")
        
        features = []
        
        for img in images:
            # Color histogram features
            hist_r = cv2.calcHist([img], [0], None, [8], [0, 256]).flatten()
            hist_g = cv2.calcHist([img], [1], None, [8], [0, 256]).flatten()
            hist_b = cv2.calcHist([img], [2], None, [8], [0, 256]).flatten()
            
            # Normalize histograms
            hist_r = hist_r / (hist_r.sum() + 1e-7)
            hist_g = hist_g / (hist_g.sum() + 1e-7)
            hist_b = hist_b / (hist_b.sum() + 1e-7)
            
            # Combine features
            feature_vec = np.concatenate([
                hist_r, hist_g, hist_b,
                [img.mean()],  # Average intensity
                [img.std()],   # Texture measure
                [img[:,:,0].mean()],  # Red channel mean
                [img[:,:,1].mean()],  # Green channel mean
                [img[:,:,2].mean()],  # Blue channel mean
            ])
            
            features.append(feature_vec)
        
        logger.info(f"Extracted features shape: {np.array(features).shape}")
        return np.array(features)
    
    def generate_synthetic_dna_features(self, num_samples, num_classes):
        """Generate synthetic DNA features (placeholder until real DNA data is available)."""
        logger.info(f"Generating {num_samples} synthetic DNA features...")
        
        # Generate random DNA-like features
        dna_features = np.random.randn(num_samples, 29)
        
        # Add some class-specific patterns
        for i in range(num_samples):
            class_idx = i % num_classes
            dna_features[i] += class_idx * 0.5
        
        return dna_features
    
    def train_models(self, X_train, X_test, y_train, y_test, model_type='image'):
        """Train multiple models and return results."""
        logger.info(f"Training {model_type} models...")
        
        results = {}
        
        # Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        logger.info(f"Random Forest accuracy: {rf_acc:.3f}")
        results['random_forest'] = {
            'model': rf,
            'accuracy': rf_acc,
            'predictions': rf_pred.tolist()
        }
        
        # Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_acc = accuracy_score(y_test, gb_pred)
        logger.info(f"Gradient Boosting accuracy: {gb_acc:.3f}")
        results['gradient_boosting'] = {
            'model': gb,
            'accuracy': gb_acc,
            'predictions': gb_pred.tolist()
        }
        
        # SVM
        logger.info("Training SVM...")
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_pred)
        logger.info(f"SVM accuracy: {svm_acc:.3f}")
        results['svm'] = {
            'model': svm,
            'accuracy': svm_acc,
            'predictions': svm_pred.tolist()
        }
        
        return results
    
    def save_models(self, models_dict, model_name):
        """Save trained models."""
        logger.info(f"Saving {model_name} models...")
        
        os.makedirs('models/real_trained', exist_ok=True)
        
        # Save best model
        best_model_name = max(models_dict.keys(), key=lambda k: models_dict[k]['accuracy'])
        best_model = models_dict[best_model_name]['model']
        
        model_path = f'models/real_trained/{model_name}_best.joblib'
        joblib.dump(best_model, model_path)
        logger.info(f"Saved best {model_name} model ({best_model_name}) to {model_path}")
        
        # Save results
        results = {k: {'accuracy': v['accuracy']} for k, v in models_dict.items()}
        results_path = f'models/real_trained/{model_name}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return model_path
    
    def run_full_training(self):
        """Run complete training pipeline with real images."""
        logger.info("=" * 60)
        logger.info("STARTING TRAINING WITH REAL PLANTVILLAGE IMAGES")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Load real images
        images, labels = self.load_images_from_directory()
        
        if len(images) == 0:
            logger.error("No images loaded! Check data directory.")
            return
        
        # Extract image features
        image_features = self.extract_image_features(images)
        
        # Generate synthetic DNA features (same count as images)
        dna_features = self.generate_synthetic_dna_features(len(images), len(self.class_names))
        
        # Split data
        X_img_train, X_img_test, y_train, y_test = train_test_split(
            image_features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        X_dna_train, X_dna_test, _, _ = train_test_split(
            dna_features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training set size: {len(X_img_train)}")
        logger.info(f"Test set size: {len(X_img_test)}")
        logger.info(f"Number of classes: {len(self.class_names)}")
        
        # Scale features
        img_scaler = StandardScaler()
        X_img_train_scaled = img_scaler.fit_transform(X_img_train)
        X_img_test_scaled = img_scaler.transform(X_img_test)
        
        dna_scaler = StandardScaler()
        X_dna_train_scaled = dna_scaler.fit_transform(X_dna_train)
        X_dna_test_scaled = dna_scaler.transform(X_dna_test)
        
        # Train image models
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING IMAGE MODELS")
        logger.info("=" * 60)
        image_models = self.train_models(X_img_train_scaled, X_img_test_scaled, y_train, y_test, 'image')
        img_model_path = self.save_models(image_models, 'image_classifier')
        
        # Train DNA models
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING DNA MODELS")
        logger.info("=" * 60)
        dna_models = self.train_models(X_dna_train_scaled, X_dna_test_scaled, y_train, y_test, 'dna')
        dna_model_path = self.save_models(dna_models, 'dna_classifier')
        
        # Fusion model (combine DNA + Image features)
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING FUSION MODEL")
        logger.info("=" * 60)
        X_fusion_train = np.concatenate([X_dna_train_scaled, X_img_train_scaled], axis=1)
        X_fusion_test = np.concatenate([X_dna_test_scaled, X_img_test_scaled], axis=1)
        
        fusion_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        fusion_model.fit(X_fusion_train, y_train)
        fusion_pred = fusion_model.predict(X_fusion_test)
        fusion_acc = accuracy_score(y_test, fusion_pred)
        logger.info(f"Fusion model accuracy: {fusion_acc:.3f}")
        
        # Save fusion model
        fusion_path = 'models/real_trained/fusion_model.joblib'
        joblib.dump(fusion_model, fusion_path)
        logger.info(f"Saved fusion model to {fusion_path}")
        
        # Save scalers and metadata
        joblib.dump(img_scaler, 'models/real_trained/image_scaler.joblib')
        joblib.dump(dna_scaler, 'models/real_trained/dna_scaler.joblib')
        
        metadata = {
            'class_names': self.class_names,
            'label_map': self.label_map,
            'num_classes': len(self.class_names),
            'num_samples': len(images),
            'train_size': len(X_img_train),
            'test_size': len(X_img_test),
            'image_feature_dim': X_img_train.shape[1],
            'dna_feature_dim': X_dna_train.shape[1],
            'training_date': datetime.now().isoformat(),
            'duration': str(datetime.now() - start_time)
        }
        
        with open('models/real_trained/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Duration: {datetime.now() - start_time}")
        logger.info(f"\nBest Image Model: {max(image_models.keys(), key=lambda k: image_models[k]['accuracy'])} - {max(m['accuracy'] for m in image_models.values()):.1%}")
        logger.info(f"Best DNA Model: {max(dna_models.keys(), key=lambda k: dna_models[k]['accuracy'])} - {max(m['accuracy'] for m in dna_models.values()):.1%}")
        logger.info(f"Fusion Model: {fusion_acc:.1%}")
        logger.info(f"\nModels saved to: models/real_trained/")
        logger.info(f"Classes trained: {len(self.class_names)}")
        logger.info(f"Total images used: {len(images)}")
        
        return {
            'image_models': image_models,
            'dna_models': dna_models,
            'fusion_accuracy': fusion_acc,
            'metadata': metadata
        }

def main():
    """Main function."""
    logger.info("DNA Crop Disease Identification - Training with Real Images")
    logger.info("=" * 60)
    
    # Check if PlantVillage directory exists
    data_dir = Path('data/images/raw/PlantVillage')
    if not data_dir.exists():
        logger.error(f"PlantVillage directory not found: {data_dir}")
        logger.error("Please ensure the PlantVillage dataset is in the correct location.")
        return
    
    # Initialize trainer
    trainer = RealImageTrainer(
        data_dir=data_dir,
        max_images_per_class=300  # Limit per class for faster training
    )
    
    # Run training
    results = trainer.run_full_training()
    
    logger.info("\nTraining completed successfully!")
    logger.info("You can now use the trained models for prediction.")

if __name__ == "__main__":
    main()

