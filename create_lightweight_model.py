#!/usr/bin/env python3
"""
Create lightweight model for GitHub/Render deployment.
Trains on 10 most common disease classes with smaller model size.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import json
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LightweightModelCreator:
    """Create lightweight model for deployment."""
    
    def __init__(self):
        # Select 10 most common and important diseases
        self.selected_classes = [
            'Apple___Apple_scab',
            'Apple___healthy',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Pepper,_bell___healthy'
        ]
        
        self.data_dir = Path('data/images/raw/PlantVillage')
        self.max_images_per_class = 200  # Reduced from 300
        
    def load_images(self):
        """Load images from selected classes only."""
        logger.info("Loading images for lightweight model...")
        logger.info(f"Selected {len(self.selected_classes)} disease classes")
        
        images = []
        labels = []
        class_map = {name: idx for idx, name in enumerate(self.selected_classes)}
        
        for class_name in self.selected_classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_name}")
                continue
            
            image_files = list(class_dir.glob('*.JPG'))[:self.max_images_per_class]
            logger.info(f"Loading {len(image_files)} images from {class_name}")
            
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    img = cv2.resize(img, (128, 128))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    images.append(img)
                    labels.append(class_map[class_name])
                    
                except Exception as e:
                    continue
        
        logger.info(f"Loaded {len(images)} total images")
        return np.array(images), np.array(labels), class_map
    
    def extract_features(self, images):
        """Extract image features."""
        logger.info("Extracting features...")
        features = []
        
        for img in images:
            hist_r = cv2.calcHist([img], [0], None, [8], [0, 256]).flatten()
            hist_g = cv2.calcHist([img], [1], None, [8], [0, 256]).flatten()
            hist_b = cv2.calcHist([img], [2], None, [8], [0, 256]).flatten()
            
            hist_r = hist_r / (hist_r.sum() + 1e-7)
            hist_g = hist_g / (hist_g.sum() + 1e-7)
            hist_b = hist_b / (hist_b.sum() + 1e-7)
            
            feature_vec = np.concatenate([
                hist_r, hist_g, hist_b,
                [img.mean()], [img.std()],
                [img[:,:,0].mean()], [img[:,:,1].mean()], [img[:,:,2].mean()],
            ])
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def train(self):
        """Train lightweight model."""
        logger.info("="*60)
        logger.info("CREATING LIGHTWEIGHT MODEL FOR DEPLOYMENT")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        # Load images
        images, labels, class_map = self.load_images()
        
        if len(images) == 0:
            logger.error("No images loaded!")
            return
        
        # Extract features
        features = self.extract_features(images)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train simplified Random Forest (smaller, faster)
        logger.info("Training lightweight Random Forest...")
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced from 100
            max_depth=15,     # Limited depth
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        logger.info(f"Lightweight model accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Save to lightweight directory
        output_dir = Path('models/lightweight')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving lightweight model...")
        joblib.dump(model, output_dir / 'image_classifier.joblib', compress=3)
        joblib.dump(scaler, output_dir / 'image_scaler.joblib', compress=3)
        
        # Save metadata
        metadata = {
            'class_names': self.selected_classes,
            'label_map': {v: k for k, v in class_map.items()},
            'num_classes': len(self.selected_classes),
            'num_samples': len(images),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': float(accuracy),
            'model_type': 'lightweight',
            'training_date': datetime.now().isoformat(),
            'duration': str(datetime.now() - start_time)
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Check sizes
        model_size = (output_dir / 'image_classifier.joblib').stat().st_size / (1024*1024)
        scaler_size = (output_dir / 'image_scaler.joblib').stat().st_size / (1024*1024)
        total_size = model_size + scaler_size
        
        logger.info("="*60)
        logger.info("LIGHTWEIGHT MODEL CREATED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Model size: {model_size:.2f} MB")
        logger.info(f"Scaler size: {scaler_size:.2f} MB")
        logger.info(f"Total size: {total_size:.2f} MB")
        logger.info(f"Accuracy: {accuracy*100:.1f}%")
        logger.info(f"Classes: {len(self.selected_classes)}")
        logger.info(f"GitHub compatible: {'✅ Yes' if total_size < 100 else '❌ No'}")
        logger.info(f"Saved to: {output_dir}")
        logger.info("="*60)
        
        return metadata

if __name__ == "__main__":
    creator = LightweightModelCreator()
    creator.train()

