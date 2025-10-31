"""
Image Data Augmentation for crop disease identification.
Provides various augmentation techniques for crop images.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import albumentations as A
import logging

logger = logging.getLogger(__name__)

class ImageAugmentation:
    """
    Image augmentation for crop disease identification.
    """
    
    def __init__(self):
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        self.severe_augmentation = self._create_severe_augmentation()
        self.light_augmentation = self._create_light_augmentation()
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """
        Create standard augmentation pipeline.
        
        Returns:
            Albumentations compose object
        """
        return A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=45, p=0.5),
            A.Transpose(p=0.3),
            
            # Scale and crop
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.RandomCrop(height=224, width=224, p=0.8),
            A.Resize(height=224, width=224, p=1.0),
            
            # Color transformations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.3),
            
            # Weather effects
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=0, drop_width=1, drop_color=(200, 200, 200), blur_value=5, brightness_coefficient=0.7, rain_type="drizzle", p=0.2),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.3),
            
            # Elastic transformations
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_severe_augmentation(self) -> A.Compose:
        """
        Create severe augmentation pipeline for challenging conditions.
        
        Returns:
            Severe augmentation pipeline
        """
        return A.Compose([
            # Heavy geometric transformations
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=90, p=0.7),
            A.Transpose(p=0.5),
            
            # Heavy scale and crop
            A.RandomScale(scale_limit=0.4, p=0.7),
            A.RandomCrop(height=224, width=224, p=1.0),
            A.Resize(height=224, width=224, p=1.0),
            
            # Heavy color transformations
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=60, val_shift_limit=40, p=0.7),
            A.RandomGamma(gamma_limit=(60, 140), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            
            # Heavy noise and blur
            A.GaussNoise(var_limit=(20.0, 100.0), p=0.5),
            A.GaussianBlur(blur_limit=(5, 15), p=0.5),
            A.MotionBlur(blur_limit=15, p=0.5),
            
            # Heavy weather effects
            A.RandomRain(slant_lower=-20, slant_upper=20, drop_length=0, drop_width=2, drop_color=(150, 150, 150), blur_value=10, brightness_coefficient=0.5, rain_type="heavy", p=0.4),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=2, num_shadows_upper=4, shadow_dimension=10, p=0.5),
            
            # Heavy elastic transformations
            A.ElasticTransform(alpha=2, sigma=100, alpha_affine=100, p=0.5),
            A.GridDistortion(num_steps=10, distort_limit=0.6, p=0.5),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_light_augmentation(self) -> A.Compose:
        """
        Create light augmentation pipeline for subtle variations.
        
        Returns:
            Light augmentation pipeline
        """
        return A.Compose([
            # Light geometric transformations
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=15, p=0.3),
            
            # Light scale
            A.RandomScale(scale_limit=0.1, p=0.3),
            A.Resize(height=224, width=224, p=1.0),
            
            # Light color transformations
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            
            # Light noise
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def augment_image(self, image: np.ndarray, augmentation_type: str = 'standard') -> np.ndarray:
        """
        Augment single image.
        
        Args:
            image: Input image array
            augmentation_type: Type of augmentation ('light', 'standard', 'severe')
            
        Returns:
            Augmented image array
        """
        if augmentation_type == 'light':
            pipeline = self.light_augmentation
        elif augmentation_type == 'severe':
            pipeline = self.severe_augmentation
        else:
            pipeline = self.augmentation_pipeline
        
        augmented = pipeline(image=image)
        return augmented['image']
    
    def augment_batch(self, images: List[np.ndarray], augmentation_type: str = 'standard') -> List[np.ndarray]:
        """
        Augment batch of images.
        
        Args:
            images: List of input images
            augmentation_type: Type of augmentation
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for image in images:
            augmented = self.augment_image(image, augmentation_type)
            augmented_images.append(augmented)
        
        return augmented_images
    
    def create_augmented_dataset(self, images: List[np.ndarray], labels: List[int], 
                                num_augmentations: int = 5, augmentation_type: str = 'standard') -> Tuple[List[np.ndarray], List[int]]:
        """
        Create augmented dataset.
        
        Args:
            images: List of input images
            labels: List of corresponding labels
            num_augmentations: Number of augmentations per image
            augmentation_type: Type of augmentation
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []
        
        for image, label in zip(images, labels):
            # Add original image
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(num_augmentations):
                augmented = self.augment_image(image, augmentation_type)
                augmented_images.append(augmented)
                augmented_labels.append(label)
        
        return augmented_images, augmented_labels
    
    def create_disease_specific_augmentation(self, disease_type: str) -> A.Compose:
        """
        Create disease-specific augmentation pipeline.
        
        Args:
            disease_type: Type of disease
            
        Returns:
            Disease-specific augmentation pipeline
        """
        if disease_type in ['rust', 'powdery_mildew']:
            # Focus on color variations for fungal diseases
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=30, p=0.7),
                A.RandomGamma(gamma_limit=(70, 130), p=0.5),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
                A.GaussNoise(var_limit=(15.0, 60.0), p=0.4),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        elif disease_type in ['bacterial_blight', 'late_blight']:
            # Focus on geometric variations for bacterial diseases
            return A.Compose([
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=45, p=0.7),
                A.RandomScale(scale_limit=0.3, p=0.6),
                A.ElasticTransform(alpha=1.5, sigma=75, alpha_affine=75, p=0.5),
                A.GridDistortion(num_steps=8, distort_limit=0.4, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        else:
            # Default augmentation for other diseases
            return self.augmentation_pipeline
    
    def augment_for_field_conditions(self, image: np.ndarray, condition: str) -> np.ndarray:
        """
        Augment image for specific field conditions.
        
        Args:
            image: Input image
            condition: Field condition ('sunny', 'cloudy', 'rainy', 'windy')
            
        Returns:
            Augmented image for field condition
        """
        if condition == 'sunny':
            # Bright, high contrast
            augmentation = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=40, val_shift_limit=30, p=1.0),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=8, p=0.7),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        elif condition == 'cloudy':
            # Lower contrast, muted colors
            augmentation = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=1.0),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        elif condition == 'rainy':
            # Add rain effects
            augmentation = A.Compose([
                A.RandomRain(slant_lower=-15, slant_upper=15, drop_length=0, drop_width=1, drop_color=(200, 200, 200), blur_value=5, brightness_coefficient=0.8, rain_type="drizzle", p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.GaussNoise(var_limit=(15.0, 40.0), p=0.6),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        elif condition == 'windy':
            # Add motion blur
            augmentation = A.Compose([
                A.MotionBlur(blur_limit=10, p=1.0),
                A.ElasticTransform(alpha=1.2, sigma=60, alpha_affine=60, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        else:
            # Default augmentation
            augmentation = self.augmentation_pipeline
        
        augmented = augmentation(image=image)
        return augmented['image']
    
    def create_mixed_augmentation(self, image: np.ndarray, num_variations: int = 3) -> List[np.ndarray]:
        """
        Create multiple variations of an image with mixed augmentations.
        
        Args:
            image: Input image
            num_variations: Number of variations to create
            
        Returns:
            List of augmented images
        """
        variations = []
        
        for i in range(num_variations):
            # Randomly choose augmentation type
            aug_type = np.random.choice(['light', 'standard', 'severe'], p=[0.2, 0.6, 0.2])
            augmented = self.augment_image(image, aug_type)
            variations.append(augmented)
        
        return variations
