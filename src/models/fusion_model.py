"""
Multimodal Fusion Model for DNA-based crop disease identification.
Combines DNA analysis and computer vision for comprehensive disease detection.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)

class MultimodalFusionModel:
    """
    Multimodal fusion model that combines DNA and image data for disease identification.
    """
    
    def __init__(self, dna_feature_dim: int = 100, image_feature_dim: int = 2048, 
                 num_classes: int = 10, fusion_method: str = 'attention'):
        self.dna_feature_dim = dna_feature_dim
        self.image_feature_dim = image_feature_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.model = None
        self.dna_encoder = None
        self.image_encoder = None
        self.fusion_layer = None
        
        # Disease classes
        self.disease_classes = [
            'healthy', 'rust', 'powdery_mildew', 'fusarium_wilt',
            'bacterial_blight', 'late_blight', 'anthracnose',
            'leaf_spot', 'mosaic_virus', 'root_rot'
        ]
    
    def build_fusion_model(self) -> keras.Model:
        """
        Build multimodal fusion model.
        
        Returns:
            Compiled fusion model
        """
        # DNA input branch
        dna_input = layers.Input(shape=(self.dna_feature_dim,), name='dna_input')
        dna_features = layers.Dense(256, activation='relu')(dna_input)
        dna_features = layers.BatchNormalization()(dna_features)
        dna_features = layers.Dropout(0.3)(dna_features)
        dna_features = layers.Dense(128, activation='relu')(dna_features)
        dna_features = layers.BatchNormalization()(dna_features)
        dna_features = layers.Dropout(0.3)(dna_features)
        
        # Image input branch
        image_input = layers.Input(shape=(self.image_feature_dim,), name='image_input')
        image_features = layers.Dense(512, activation='relu')(image_input)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Dropout(0.3)(image_features)
        image_features = layers.Dense(256, activation='relu')(image_features)
        image_features = layers.BatchNormalization()(image_features)
        image_features = layers.Dropout(0.3)(image_features)
        
        # Fusion layer
        if self.fusion_method == 'concatenation':
            fused_features = self._concatenation_fusion(dna_features, image_features)
        elif self.fusion_method == 'attention':
            fused_features = self._attention_fusion(dna_features, image_features)
        elif self.fusion_method == 'bilinear':
            fused_features = self._bilinear_fusion(dna_features, image_features)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Classification head
        x = layers.Dense(512, activation='relu')(fused_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        output = layers.Dense(self.num_classes, activation='softmax', name='disease_output')(x)
        
        # Create model
        model = models.Model(
            inputs=[dna_input, image_input],
            outputs=output,
            name='multimodal_fusion_model'
        )
        
        return model
    
    def _concatenation_fusion(self, dna_features, image_features):
        """Simple concatenation fusion."""
        return layers.Concatenate()([dna_features, image_features])
    
    def _attention_fusion(self, dna_features, image_features):
        """Attention-based fusion."""
        # Calculate attention weights
        attention_weights = layers.Dense(1, activation='sigmoid')(layers.Concatenate()([dna_features, image_features]))
        
        # Apply attention
        weighted_dna = layers.Multiply()([dna_features, attention_weights])
        weighted_image = layers.Multiply()([image_features, 1 - attention_weights])
        
        # Concatenate weighted features
        return layers.Concatenate()([weighted_dna, weighted_image])
    
    def _bilinear_fusion(self, dna_features, image_features):
        """Bilinear fusion for multimodal features."""
        # Reshape features for bilinear operation
        dna_reshaped = layers.Reshape((128, 1))(dna_features)
        image_reshaped = layers.Reshape((1, 256))(image_features)
        
        # Bilinear pooling
        bilinear = layers.Multiply()([dna_reshaped, image_reshaped])
        bilinear = layers.Flatten()(bilinear)
        
        # Combine with original features
        combined = layers.Concatenate()([dna_features, image_features, bilinear])
        
        return combined
    
    def compile_model(self, learning_rate: float = 0.001) -> keras.Model:
        """
        Compile the fusion model.
        
        Args:
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled model
        """
        if self.model is None:
            self.model = self.build_fusion_model()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return self.model
    
    def train_model(self, dna_data: np.ndarray, image_data: np.ndarray, 
                   labels: np.ndarray, validation_split: float = 0.2, 
                   epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the multimodal fusion model.
        
        Args:
            dna_data: DNA feature data
            image_data: Image feature data
            labels: Disease labels
            validation_split: Validation data split
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.model = self.compile_model()
        
        # Convert labels to categorical
        labels_categorical = keras.utils.to_categorical(labels, self.num_classes)
        
        # Callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_fusion_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            [dna_data, image_data],
            labels_categorical,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history.history
    
    def predict_disease(self, dna_features: np.ndarray, image_features: np.ndarray) -> Dict:
        """
        Predict disease from DNA and image features.
        
        Args:
            dna_features: DNA feature vector
            image_features: Image feature vector
            
        Returns:
            Prediction results dictionary
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure correct input shapes
        if len(dna_features.shape) == 1:
            dna_features = np.expand_dims(dna_features, axis=0)
        if len(image_features.shape) == 1:
            image_features = np.expand_dims(image_features, axis=0)
        
        # Make prediction
        predictions = self.model.predict([dna_features, image_features], verbose=0)
        
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
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Convert labels to categorical
        labels_categorical = keras.utils.to_categorical(labels, self.num_classes)
        
        # Evaluate model
        results = self.model.evaluate([dna_data, image_data], labels_categorical, verbose=0)
        
        # Get predictions for detailed analysis
        predictions = self.model.predict([dna_data, image_data], verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'top_3_accuracy': results[2],
            'classification_report': classification_report(labels, y_pred, target_names=self.disease_classes),
            'confusion_matrix': confusion_matrix(labels, y_pred).tolist()
        }
        
        return metrics
    
    def get_feature_importance(self, dna_features: np.ndarray, image_features: np.ndarray) -> Dict:
        """
        Get feature importance for DNA and image features.
        
        Args:
            dna_features: DNA feature vector
            image_features: Image feature vector
            
        Returns:
            Feature importance dictionary
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get intermediate layer outputs
        dna_encoder = keras.Model(
            inputs=self.model.input[0],
            outputs=self.model.get_layer('dna_features').output
        )
        image_encoder = keras.Model(
            inputs=self.model.input[1],
            outputs=self.model.get_layer('image_features').output
        )
        
        # Get encoded features
        dna_encoded = dna_encoder.predict(dna_features, verbose=0)
        image_encoded = image_encoder.predict(image_features, verbose=0)
        
        return {
            'dna_feature_importance': dna_encoded[0].tolist(),
            'image_feature_importance': image_encoded[0].tolist(),
            'dna_feature_magnitude': np.linalg.norm(dna_encoded[0]),
            'image_feature_magnitude': np.linalg.norm(image_encoded[0])
        }
    
    def save_model(self, filepath: str):
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        logger.info(f"Fusion model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model.
        
        Args:
            filepath: Path to load model from
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Fusion model loaded from {filepath}")
    
    def fine_tune_model(self, dna_data: np.ndarray, image_data: np.ndarray, 
                       labels: np.ndarray, epochs: int = 20, learning_rate: float = 1e-5):
        """
        Fine-tune the fusion model.
        
        Args:
            dna_data: DNA feature data
            image_data: Image feature data
            labels: Disease labels
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
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Convert labels to categorical
        labels_categorical = keras.utils.to_categorical(labels, self.num_classes)
        
        # Fine-tune
        self.model.fit(
            [dna_data, image_data],
            labels_categorical,
            epochs=epochs,
            verbose=1
        )
