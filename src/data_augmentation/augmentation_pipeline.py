"""
Augmentation Pipeline for DNA-based crop disease identification.
Combines image and DNA augmentation techniques.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from Bio.SeqRecord import SeqRecord
import logging
from .image_augmentation import ImageAugmentation
from .dna_augmentation import DNAAugmentation

logger = logging.getLogger(__name__)

class AugmentationPipeline:
    """
    Combined augmentation pipeline for DNA and image data.
    """
    
    def __init__(self):
        self.image_augmentation = ImageAugmentation()
        self.dna_augmentation = DNAAugmentation()
    
    def augment_multimodal_data(self, dna_sequence: SeqRecord, image: np.ndarray, 
                               num_augmentations: int = 5) -> Tuple[List[SeqRecord], List[np.ndarray]]:
        """
        Augment both DNA and image data together.
        
        Args:
            dna_sequence: DNA sequence record
            image: Image array
            num_augmentations: Number of augmentations to create
            
        Returns:
            Tuple of (augmented_sequences, augmented_images)
        """
        augmented_sequences = []
        augmented_images = []
        
        for i in range(num_augmentations):
            # Augment DNA sequence
            aug_sequence = self.dna_augmentation.augment_sequence(
                dna_sequence, 
                augmentation_type='standard'
            )
            aug_sequence.id = f"{dna_sequence.id}_multimodal_{i+1}"
            
            # Augment image
            aug_image = self.image_augmentation.augment_image(
                image, 
                augmentation_type='standard'
            )
            
            augmented_sequences.append(aug_sequence)
            augmented_images.append(aug_image)
        
        return augmented_sequences, augmented_images
    
    def create_training_dataset(self, dna_sequences: List[SeqRecord], images: List[np.ndarray], 
                               labels: List[int], augmentation_ratio: float = 2.0) -> Tuple[List[SeqRecord], List[np.ndarray], List[int]]:
        """
        Create augmented training dataset.
        
        Args:
            dna_sequences: List of DNA sequences
            images: List of images
            labels: List of labels
            augmentation_ratio: Ratio of augmented data to original data
            
        Returns:
            Tuple of (augmented_sequences, augmented_images, augmented_labels)
        """
        augmented_sequences = []
        augmented_images = []
        augmented_labels = []
        
        # Add original data
        augmented_sequences.extend(dna_sequences)
        augmented_images.extend(images)
        augmented_labels.extend(labels)
        
        # Add augmented data
        num_augmentations = int(len(dna_sequences) * augmentation_ratio)
        
        for i in range(num_augmentations):
            # Randomly select original sample
            idx = np.random.randint(0, len(dna_sequences))
            
            # Augment DNA sequence
            aug_sequence = self.dna_augmentation.augment_sequence(
                dna_sequences[idx], 
                augmentation_type='standard'
            )
            aug_sequence.id = f"{dna_sequences[idx].id}_aug_{i+1}"
            
            # Augment image
            aug_image = self.image_augmentation.augment_image(
                images[idx], 
                augmentation_type='standard'
            )
            
            augmented_sequences.append(aug_sequence)
            augmented_images.append(aug_image)
            augmented_labels.append(labels[idx])
        
        return augmented_sequences, augmented_images, augmented_labels
    
    def augment_for_disease_type(self, dna_sequence: SeqRecord, image: np.ndarray, 
                                disease_type: str) -> Tuple[SeqRecord, np.ndarray]:
        """
        Augment data specifically for disease type.
        
        Args:
            dna_sequence: DNA sequence
            image: Image array
            disease_type: Type of disease
            
        Returns:
            Tuple of (augmented_sequence, augmented_image)
        """
        # Disease-specific DNA augmentation
        if disease_type in ['rust', 'powdery_mildew']:
            # Fungal diseases - focus on resistance gene variations
            aug_sequence = self.dna_augmentation.augment_sequence(
                dna_sequence, 
                augmentation_type='light'
            )
        elif disease_type in ['bacterial_blight', 'late_blight']:
            # Bacterial diseases - more aggressive augmentation
            aug_sequence = self.dna_augmentation.augment_sequence(
                dna_sequence, 
                augmentation_type='severe'
            )
        else:
            # Default augmentation
            aug_sequence = self.dna_augmentation.augment_sequence(
                dna_sequence, 
                augmentation_type='standard'
            )
        
        # Disease-specific image augmentation
        aug_image = self.image_augmentation.augment_image(
            image, 
            augmentation_type='standard'
        )
        
        return aug_sequence, aug_image
    
    def create_balanced_dataset(self, dna_sequences: List[SeqRecord], images: List[np.ndarray], 
                               labels: List[int], target_samples_per_class: int = 1000) -> Tuple[List[SeqRecord], List[np.ndarray], List[int]]:
        """
        Create balanced dataset with augmentation.
        
        Args:
            dna_sequences: List of DNA sequences
            images: List of images
            labels: List of labels
            target_samples_per_class: Target number of samples per class
            
        Returns:
            Tuple of (balanced_sequences, balanced_images, balanced_labels)
        """
        unique_labels = np.unique(labels)
        balanced_sequences = []
        balanced_images = []
        balanced_labels = []
        
        for label in unique_labels:
            # Get samples for this class
            class_indices = np.where(np.array(labels) == label)[0]
            class_sequences = [dna_sequences[i] for i in class_indices]
            class_images = [images[i] for i in class_indices]
            
            # Add original samples
            balanced_sequences.extend(class_sequences)
            balanced_images.extend(class_images)
            balanced_labels.extend([label] * len(class_sequences))
            
            # Calculate how many augmentations needed
            current_samples = len(class_sequences)
            needed_samples = target_samples_per_class - current_samples
            
            if needed_samples > 0:
                # Create augmented samples
                for i in range(needed_samples):
                    # Randomly select original sample
                    idx = np.random.randint(0, len(class_sequences))
                    
                    # Augment
                    aug_sequence = self.dna_augmentation.augment_sequence(
                        class_sequences[idx], 
                        augmentation_type='standard'
                    )
                    aug_sequence.id = f"{class_sequences[idx].id}_balanced_{i+1}"
                    
                    aug_image = self.image_augmentation.augment_image(
                        class_images[idx], 
                        augmentation_type='standard'
                    )
                    
                    balanced_sequences.append(aug_sequence)
                    balanced_images.append(aug_image)
                    balanced_labels.append(label)
        
        return balanced_sequences, balanced_images, balanced_labels
    
    def augment_for_field_conditions(self, dna_sequence: SeqRecord, image: np.ndarray, 
                                    field_condition: str) -> Tuple[SeqRecord, np.ndarray]:
        """
        Augment data for specific field conditions.
        
        Args:
            dna_sequence: DNA sequence
            image: Image array
            field_condition: Field condition ('sunny', 'cloudy', 'rainy', 'windy')
            
        Returns:
            Tuple of (augmented_sequence, augmented_image)
        """
        # Augment DNA sequence
        aug_sequence = self.dna_augmentation.augment_sequence(
            dna_sequence, 
            augmentation_type='standard'
        )
        
        # Augment image for field condition
        aug_image = self.image_augmentation.augment_for_field_conditions(
            image, 
            field_condition
        )
        
        return aug_sequence, aug_image
    
    def create_robust_dataset(self, dna_sequences: List[SeqRecord], images: List[np.ndarray], 
                             labels: List[int], robustness_level: str = 'medium') -> Tuple[List[SeqRecord], List[np.ndarray], List[int]]:
        """
        Create robust dataset with multiple augmentation strategies.
        
        Args:
            dna_sequences: List of DNA sequences
            images: List of images
            labels: List of labels
            robustness_level: Level of robustness ('low', 'medium', 'high')
            
        Returns:
            Tuple of (robust_sequences, robust_images, robust_labels)
        """
        if robustness_level == 'low':
            num_variations = 2
            dna_aug_type = 'light'
            image_aug_type = 'light'
        elif robustness_level == 'high':
            num_variations = 5
            dna_aug_type = 'severe'
            image_aug_type = 'severe'
        else:  # medium
            num_variations = 3
            dna_aug_type = 'standard'
            image_aug_type = 'standard'
        
        robust_sequences = []
        robust_images = []
        robust_labels = []
        
        # Add original data
        robust_sequences.extend(dna_sequences)
        robust_images.extend(images)
        robust_labels.extend(labels)
        
        # Create variations
        for i in range(len(dna_sequences)):
            for j in range(num_variations):
                # Augment DNA sequence
                aug_sequence = self.dna_augmentation.augment_sequence(
                    dna_sequences[i], 
                    augmentation_type=dna_aug_type
                )
                aug_sequence.id = f"{dna_sequences[i].id}_robust_{j+1}"
                
                # Augment image
                aug_image = self.image_augmentation.augment_image(
                    images[i], 
                    augmentation_type=image_aug_type
                )
                
                robust_sequences.append(aug_sequence)
                robust_images.append(aug_image)
                robust_labels.append(labels[i])
        
        return robust_sequences, robust_images, robust_labels
