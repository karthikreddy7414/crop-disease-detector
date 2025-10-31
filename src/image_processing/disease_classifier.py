"""
CNN-based Image Disease Classifier for crop disease identification.
Uses deep learning models to classify diseases from crop images.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)

class ImageDiseaseClassifier:
    """
    CNN-based classifier for crop disease identification from images.
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = []
        
        # Disease classes for crop diseases
        self.disease_classes = [
            'healthy', 'rust', 'powdery_mildew', 'fusarium_wilt', 
            'bacterial_blight', 'late_blight', 'anthracnose', 
            'leaf_spot', 'mosaic_virus', 'root_rot'
        ]
    
    def build_cnn_model(self, model_type='custom') -> keras.Model:
        """
        Build CNN model for disease classification.
        
        Args:
            model_type: Type of model ('custom', 'resnet', 'efficientnet')
            
        Returns:
            Compiled Keras model
        """
        if model_type == 'custom':
            return self._build_custom_cnn()
        elif model_type == 'resnet':
            return self._build_resnet_model()
        elif model_type == 'efficientnet':
            return self._build_efficientnet_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _build_custom_cnn(self) -> keras.Model:
        """
        Build custom CNN architecture.
        
        Returns:
            Custom CNN model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_resnet_model(self) -> keras.Model:
        """
        Build ResNet-based model.
        
        Returns:
            ResNet model
        """
        # Use pre-trained ResNet50 as base
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_efficientnet_model(self) -> keras.Model:
        """
        Build EfficientNet-based model.
        
        Returns:
            EfficientNet model
        """
        # Use pre-trained EfficientNetB0 as base
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model: keras.Model, learning_rate: float = 0.001) -> keras.Model:
        """
        Compile model with optimizer and loss function.
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled model
        """
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return model
    
    def train_model(self, train_data, val_data, epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the CNN model.
        
        Args:
            train_data: Training data generator
            val_data: Validation data generator
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history.history
    
    def predict_disease(self, image: np.ndarray) -> Dict:
        """
        Predict disease from image.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            top_predictions.append({
                'disease': self.disease_classes[idx],
                'confidence': float(predictions[0][idx])
            })
        
        return {
            'predicted_disease': self.disease_classes[top_indices[0]],
            'confidence': float(predictions[0][top_indices[0]]),
            'top_predictions': top_predictions,
            'all_probabilities': {
                disease: float(prob) for disease, prob in zip(self.disease_classes, predictions[0])
            }
        }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        # Resize image
        resized = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def evaluate_model(self, test_data) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            test_data: Test data generator
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Evaluate model
        results = self.model.evaluate(test_data, verbose=0)
        
        # Get predictions for detailed analysis
        predictions = self.model.predict(test_data, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Get true labels
        y_true = test_data.classes
        
        # Calculate metrics
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'top_3_accuracy': results[2],
            'classification_report': classification_report(y_true, y_pred, target_names=self.disease_classes),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    def fine_tune_model(self, train_data, val_data, epochs: int = 20, learning_rate: float = 1e-5):
        """
        Fine-tune the model with lower learning rate.
        
        Args:
            train_data: Training data generator
            val_data: Validation data generator
            epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for fine-tuning
        """
        if self.model is None:
            raise ValueError("Model must be trained before fine-tuning")
        
        # Unfreeze some layers for fine-tuning
        for layer in self.model.layers[-10:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Fine-tune
        self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            verbose=1
        )
    
    def save_model(self, filepath: str):
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model.
        
        Args:
            filepath: Path to load model from
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_maps(self, image: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Get feature maps from specific layer.
        
        Args:
            image: Input image
            layer_name: Name of layer to extract features from
            
        Returns:
            Feature maps from specified layer
        """
        if self.model is None:
            raise ValueError("Model must be loaded before extracting features")
        
        # Create feature extraction model
        feature_extractor = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Extract features
        features = feature_extractor.predict(processed_image, verbose=0)
        
        return features
    
    def visualize_attention(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize attention maps for disease regions.
        
        Args:
            image: Input image
            
        Returns:
            Attention visualization
        """
        if self.model is None:
            raise ValueError("Model must be loaded before visualization")
        
        # Get feature maps from last convolutional layer
        feature_maps = self.get_feature_maps(image, 'conv2d_6')  # Last conv layer
        
        # Global average pooling
        gap = np.mean(feature_maps, axis=(1, 2))
        
        # Weighted feature maps
        weighted_maps = np.zeros_like(feature_maps[0])
        for i in range(feature_maps.shape[-1]):
            weighted_maps += gap[0, i] * feature_maps[0, :, :, i]
        
        # Normalize and resize to original image size
        attention_map = cv2.resize(weighted_maps, (image.shape[1], image.shape[0]))
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Apply colormap
        attention_vis = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Overlay on original image
        overlay = cv2.addWeighted(image, 0.6, attention_vis, 0.4, 0)
        
        return overlay
