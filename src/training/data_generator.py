"""
Data generator for training DNA-based crop disease identification models.
Generates synthetic training data for DNA sequences and crop images.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
import string
from PIL import Image, ImageDraw, ImageFont
import os
import logging

logger = logging.getLogger(__name__)

class DataGenerator:
    """
    Generates synthetic training data for DNA sequences and crop images.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Disease classes
        self.disease_classes = [
            'healthy', 'rust', 'powdery_mildew', 'fusarium_wilt',
            'bacterial_blight', 'late_blight', 'anthracnose',
            'leaf_spot', 'mosaic_virus', 'root_rot'
        ]
        
        # Crop types
        self.crop_types = ['tomato', 'rice', 'wheat', 'corn', 'potato']
        
        # Disease-specific genetic markers
        self.disease_markers = {
            'rust': ['ATCGATCG', 'GCTAGCTA', 'TAGCTAGC'],
            'powdery_mildew': ['CGTACGTA', 'TACGTACG', 'ACGTACGT'],
            'fusarium_wilt': ['GATCGATC', 'ATCGATCG', 'TCGATCGA'],
            'bacterial_blight': ['CGATCGAT', 'GATCGATC', 'ATCGATCG'],
            'late_blight': ['TACGTACG', 'ACGTACGT', 'CGTACGTA'],
            'anthracnose': ['GCTAGCTA', 'CTAGCTAG', 'TAGCTAGC'],
            'leaf_spot': ['ATCGATCG', 'TCGATCGA', 'CGATCGAT'],
            'mosaic_virus': ['CGTACGTA', 'GTACGTAC', 'TACGTACG'],
            'root_rot': ['GATCGATC', 'ATCGATCG', 'TCGATCGA'],
            'healthy': ['ATCGATCG', 'GCTAGCTA', 'TAGCTAGC']
        }
    
    def generate_dna_sequence(self, length: int = 1000, disease: str = 'healthy') -> str:
        """
        Generate a synthetic DNA sequence.
        
        Args:
            length: Length of the sequence
            disease: Disease type to influence sequence characteristics
            
        Returns:
            DNA sequence string
        """
        # Base nucleotides
        nucleotides = ['A', 'T', 'C', 'G']
        
        # Generate base sequence
        sequence = ''.join(np.random.choice(nucleotides, length))
        
        # Add disease-specific markers if not healthy
        if disease != 'healthy' and disease in self.disease_markers:
            markers = self.disease_markers[disease]
            for marker in markers:
                # Insert marker at random position
                pos = np.random.randint(0, length - len(marker))
                sequence = sequence[:pos] + marker + sequence[pos + len(marker):]
        
        return sequence
    
    def generate_dna_features(self, sequence: str) -> Dict:
        """
        Generate features from DNA sequence.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Dictionary of extracted features
        """
        # Basic sequence statistics
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
        at_content = (sequence.count('A') + sequence.count('T')) / len(sequence) * 100
        
        # Dinucleotide frequencies
        dinucleotides = ['AA', 'AT', 'AC', 'AG', 'TA', 'TT', 'TC', 'TG',
                        'CA', 'CT', 'CC', 'CG', 'GA', 'GT', 'GC', 'GG']
        dinuc_freq = {dinuc: sequence.count(dinuc) / (len(sequence) - 1) 
                     for dinuc in dinucleotides}
        
        # Trinucleotide frequencies
        trinucleotides = ['AAA', 'AAT', 'AAC', 'AAG', 'ATA', 'ATT', 'ATC', 'ATG']
        trinuc_freq = {trinuc: sequence.count(trinuc) / (len(sequence) - 2) 
                      for trinuc in trinucleotides}
        
        # Motif detection
        motif_count = sum(sequence.count(marker) for marker in 
                         [marker for markers in self.disease_markers.values() 
                          for marker in markers])
        
        # Sequence complexity
        complexity = len(set(sequence)) / len(sequence)
        
        return {
            'sequence_length': len(sequence),
            'gc_content': gc_content,
            'at_content': at_content,
            'complexity': complexity,
            'motif_count': motif_count,
            **dinuc_freq,
            **trinuc_freq
        }
    
    def generate_image_features(self, disease: str, crop_type: str) -> Dict:
        """
        Generate synthetic image features.
        
        Args:
            disease: Disease type
            crop_type: Type of crop
            
        Returns:
            Dictionary of image features
        """
        # Create synthetic image
        img = Image.new('RGB', (224, 224), color='green')
        draw = ImageDraw.Draw(img)
        
        # Add disease-specific visual patterns
        if disease == 'rust':
            # Orange/brown spots
            for _ in range(np.random.randint(5, 15)):
                x, y = np.random.randint(0, 224, 2)
                draw.ellipse([x-5, y-5, x+5, y+5], fill='orange')
        elif disease == 'powdery_mildew':
            # White powdery coating
            for _ in range(np.random.randint(20, 40)):
                x, y = np.random.randint(0, 224, 2)
                draw.ellipse([x-2, y-2, x+2, y+2], fill='white')
        elif disease == 'leaf_spot':
            # Dark spots
            for _ in range(np.random.randint(3, 10)):
                x, y = np.random.randint(0, 224, 2)
                draw.ellipse([x-8, y-8, x+8, y+8], fill='brown')
        
        # Convert to numpy array for feature extraction
        img_array = np.array(img)
        
        # Extract color features
        r_mean, g_mean, b_mean = img_array.mean(axis=(0, 1))
        r_std, g_std, b_std = img_array.std(axis=(0, 1))
        
        # Extract texture features (simplified)
        gray = np.mean(img_array, axis=2)
        texture_energy = np.var(gray)
        texture_entropy = -np.sum((np.histogram(gray, bins=256)[0] / gray.size) * 
                                 np.log2(np.histogram(gray, bins=256)[0] / gray.size + 1e-10))
        
        # Shape features
        edges_h = np.abs(np.diff(gray, axis=0))
        edges_v = np.abs(np.diff(gray, axis=1))
        edges = np.concatenate([edges_h.flatten(), edges_v.flatten()])
        edge_density = np.sum(edges > 30) / len(edges)
        
        return {
            'r_mean': r_mean,
            'g_mean': g_mean,
            'b_mean': b_mean,
            'r_std': r_std,
            'g_std': g_std,
            'b_std': b_std,
            'texture_energy': texture_energy,
            'texture_entropy': texture_entropy,
            'edge_density': edge_density,
            'brightness': np.mean(gray),
            'contrast': np.std(gray)
        }
    
    def generate_training_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Generate comprehensive training dataset.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (dna_features, image_features, labels, disease_names)
        """
        logger.info(f"Generating {num_samples} training samples...")
        
        dna_features_list = []
        image_features_list = []
        labels = []
        disease_names = []
        
        for i in range(num_samples):
            # Randomly select disease and crop type
            disease = np.random.choice(self.disease_classes)
            crop_type = np.random.choice(self.crop_types)
            
            # Generate DNA sequence and features
            dna_seq = self.generate_dna_sequence(
                length=np.random.randint(500, 2000),
                disease=disease
            )
            dna_features = self.generate_dna_features(dna_seq)
            
            # Generate image features
            image_features = self.generate_image_features(disease, crop_type)
            
            # Store features
            dna_features_list.append(list(dna_features.values()))
            image_features_list.append(list(image_features.values()))
            labels.append(self.disease_classes.index(disease))
            disease_names.append(disease)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        return (np.array(dna_features_list), 
                np.array(image_features_list), 
                np.array(labels), 
                disease_names)
    
    def save_training_data(self, dna_features: np.ndarray, image_features: np.ndarray, 
                          labels: np.ndarray, disease_names: List[str], 
                          output_dir: str = "data/training"):
        """
        Save training data to files.
        
        Args:
            dna_features: DNA feature matrix
            image_features: Image feature matrix
            labels: Label vector
            disease_names: List of disease names
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, "dna_features.npy"), dna_features)
        np.save(os.path.join(output_dir, "image_features.npy"), image_features)
        np.save(os.path.join(output_dir, "labels.npy"), labels)
        
        # Save metadata
        metadata = {
            'num_samples': len(labels),
            'dna_feature_dim': dna_features.shape[1],
            'image_feature_dim': image_features.shape[1],
            'disease_classes': self.disease_classes,
            'crop_types': self.crop_types
        }
        
        import json
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training data saved to {output_dir}")
    
    def load_training_data(self, data_dir: str = "data/training") -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load training data from files.
        
        Args:
            data_dir: Data directory
            
        Returns:
            Tuple of (dna_features, image_features, labels, metadata)
        """
        dna_features = np.load(os.path.join(data_dir, "dna_features.npy"))
        image_features = np.load(os.path.join(data_dir, "image_features.npy"))
        labels = np.load(os.path.join(data_dir, "labels.npy"))
        
        with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded training data from {data_dir}")
        return dna_features, image_features, labels, metadata
