"""
DNA Data Augmentation for crop disease identification.
Provides various augmentation techniques for DNA sequences.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import random
import logging

logger = logging.getLogger(__name__)

class DNAAugmentation:
    """
    DNA sequence augmentation for crop disease identification.
    """
    
    def __init__(self):
        self.nucleotide_bases = ['A', 'T', 'G', 'C']
        self.mutation_rates = {
            'substitution': 0.01,
            'insertion': 0.005,
            'deletion': 0.005,
            'inversion': 0.002
        }
    
    def augment_sequence(self, sequence: SeqRecord, augmentation_type: str = 'standard') -> SeqRecord:
        """
        Augment DNA sequence.
        
        Args:
            sequence: Input DNA sequence
            augmentation_type: Type of augmentation ('light', 'standard', 'severe')
            
        Returns:
            Augmented sequence
        """
        if augmentation_type == 'light':
            return self._light_augmentation(sequence)
        elif augmentation_type == 'severe':
            return self._severe_augmentation(sequence)
        else:
            return self._standard_augmentation(sequence)
    
    def _light_augmentation(self, sequence: SeqRecord) -> SeqRecord:
        """
        Apply light augmentation to sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Lightly augmented sequence
        """
        seq_str = str(sequence.seq)
        
        # Apply 1-2 random mutations
        num_mutations = random.randint(1, 2)
        augmented_seq = self._apply_mutations(seq_str, num_mutations)
        
        return SeqRecord(
            Seq(augmented_seq),
            id=f"{sequence.id}_light_aug",
            description=f"Light augmentation of {sequence.description}"
        )
    
    def _standard_augmentation(self, sequence: SeqRecord) -> SeqRecord:
        """
        Apply standard augmentation to sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Standard augmented sequence
        """
        seq_str = str(sequence.seq)
        
        # Apply 3-5 random mutations
        num_mutations = random.randint(3, 5)
        augmented_seq = self._apply_mutations(seq_str, num_mutations)
        
        return SeqRecord(
            Seq(augmented_seq),
            id=f"{sequence.id}_standard_aug",
            description=f"Standard augmentation of {sequence.description}"
        )
    
    def _severe_augmentation(self, sequence: SeqRecord) -> SeqRecord:
        """
        Apply severe augmentation to sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Severely augmented sequence
        """
        seq_str = str(sequence.seq)
        
        # Apply 6-10 random mutations
        num_mutations = random.randint(6, 10)
        augmented_seq = self._apply_mutations(seq_str, num_mutations)
        
        return SeqRecord(
            Seq(augmented_seq),
            id=f"{sequence.id}_severe_aug",
            description=f"Severe augmentation of {sequence.description}"
        )
    
    def _apply_mutations(self, sequence: str, num_mutations: int) -> str:
        """
        Apply random mutations to sequence.
        
        Args:
            sequence: Input sequence string
            num_mutations: Number of mutations to apply
            
        Returns:
            Mutated sequence
        """
        seq_list = list(sequence)
        seq_length = len(seq_list)
        
        for _ in range(num_mutations):
            if seq_length == 0:
                break
            
            # Choose random position
            pos = random.randint(0, seq_length - 1)
            
            # Choose mutation type
            mutation_type = random.choices(
                list(self.mutation_rates.keys()),
                weights=list(self.mutation_rates.values())
            )[0]
            
            if mutation_type == 'substitution':
                # Substitute nucleotide
                original = seq_list[pos]
                new_base = random.choice([b for b in self.nucleotide_bases if b != original])
                seq_list[pos] = new_base
            
            elif mutation_type == 'insertion':
                # Insert random nucleotide
                new_base = random.choice(self.nucleotide_bases)
                seq_list.insert(pos, new_base)
                seq_length += 1
            
            elif mutation_type == 'deletion':
                # Delete nucleotide
                if seq_length > 1:
                    del seq_list[pos]
                    seq_length -= 1
            
            elif mutation_type == 'inversion':
                # Invert small segment
                segment_length = random.randint(2, 5)
                if pos + segment_length <= seq_length:
                    segment = seq_list[pos:pos + segment_length]
                    seq_list[pos:pos + segment_length] = segment[::-1]
        
        return ''.join(seq_list)
    
    def create_sequence_variants(self, sequence: SeqRecord, num_variants: int = 5) -> List[SeqRecord]:
        """
        Create multiple variants of a sequence.
        
        Args:
            sequence: Input sequence
            num_variants: Number of variants to create
            
        Returns:
            List of sequence variants
        """
        variants = []
        
        for i in range(num_variants):
            # Randomly choose augmentation type
            aug_type = random.choices(
                ['light', 'standard', 'severe'],
                weights=[0.3, 0.5, 0.2]
            )[0]
            
            variant = self.augment_sequence(sequence, aug_type)
            variant.id = f"{sequence.id}_variant_{i+1}"
            variants.append(variant)
        
        return variants
    
    def augment_sequence_features(self, features: Dict, noise_level: float = 0.1) -> Dict:
        """
        Augment sequence features with noise.
        
        Args:
            features: Dictionary of sequence features
            noise_level: Level of noise to add
            
        Returns:
            Augmented features dictionary
        """
        augmented_features = {}
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                # Add Gaussian noise to numerical features
                noise = np.random.normal(0, noise_level * abs(value))
                augmented_features[key] = value + noise
            elif isinstance(value, dict):
                # Recursively augment nested dictionaries
                augmented_features[key] = self.augment_sequence_features(value, noise_level)
            else:
                # Keep non-numerical features unchanged
                augmented_features[key] = value
        
        return augmented_features
    
    def create_synthetic_sequences(self, reference_sequence: SeqRecord, num_sequences: int = 10) -> List[SeqRecord]:
        """
        Create synthetic sequences based on reference.
        
        Args:
            reference_sequence: Reference sequence
            num_sequences: Number of synthetic sequences to create
            
        Returns:
            List of synthetic sequences
        """
        synthetic_sequences = []
        ref_seq = str(reference_sequence.seq)
        
        for i in range(num_sequences):
            # Create synthetic sequence with controlled mutations
            synthetic_seq = self._create_synthetic_sequence(ref_seq, mutation_rate=0.05)
            
            synthetic_record = SeqRecord(
                Seq(synthetic_seq),
                id=f"synthetic_{reference_sequence.id}_{i+1}",
                description=f"Synthetic sequence based on {reference_sequence.description}"
            )
            
            synthetic_sequences.append(synthetic_record)
        
        return synthetic_sequences
    
    def _create_synthetic_sequence(self, reference: str, mutation_rate: float = 0.05) -> str:
        """
        Create synthetic sequence from reference.
        
        Args:
            reference: Reference sequence
            mutation_rate: Rate of mutations
            
        Returns:
            Synthetic sequence
        """
        seq_list = list(reference)
        num_mutations = int(len(seq_list) * mutation_rate)
        
        # Apply random mutations
        for _ in range(num_mutations):
            if len(seq_list) == 0:
                break
            
            pos = random.randint(0, len(seq_list) - 1)
            mutation_type = random.choice(['substitution', 'insertion', 'deletion'])
            
            if mutation_type == 'substitution':
                original = seq_list[pos]
                new_base = random.choice([b for b in self.nucleotide_bases if b != original])
                seq_list[pos] = new_base
            
            elif mutation_type == 'insertion':
                new_base = random.choice(self.nucleotide_bases)
                seq_list.insert(pos, new_base)
            
            elif mutation_type == 'deletion' and len(seq_list) > 1:
                del seq_list[pos]
        
        return ''.join(seq_list)
    
    def augment_genetic_markers(self, markers: Dict, noise_level: float = 0.1) -> Dict:
        """
        Augment genetic marker data.
        
        Args:
            markers: Dictionary of genetic markers
            noise_level: Level of noise to add
            
        Returns:
            Augmented markers dictionary
        """
        augmented_markers = {}
        
        for marker_type, marker_data in markers.items():
            if isinstance(marker_data, dict):
                augmented_markers[marker_type] = {}
                
                for marker_name, marker_info in marker_data.items():
                    if isinstance(marker_info, dict):
                        augmented_markers[marker_type][marker_name] = {}
                        
                        for key, value in marker_info.items():
                            if isinstance(value, (int, float)):
                                # Add noise to numerical values
                                noise = np.random.normal(0, noise_level * abs(value))
                                augmented_markers[marker_type][marker_name][key] = value + noise
                            else:
                                augmented_markers[marker_type][marker_name][key] = value
                    else:
                        augmented_markers[marker_type][marker_name] = marker_info
            else:
                augmented_markers[marker_type] = marker_data
        
        return augmented_markers
    
    def create_sequence_ensemble(self, sequences: List[SeqRecord], ensemble_size: int = 10) -> List[SeqRecord]:
        """
        Create ensemble of sequences for robust analysis.
        
        Args:
            sequences: List of input sequences
            ensemble_size: Size of ensemble to create
            
        Returns:
            List of ensemble sequences
        """
        ensemble_sequences = []
        
        for sequence in sequences:
            # Create multiple variants of each sequence
            variants = self.create_sequence_variants(sequence, num_variants=ensemble_size)
            ensemble_sequences.extend(variants)
        
        return ensemble_sequences
    
    def augment_sequence_alignment(self, alignment_data: Dict, noise_level: float = 0.1) -> Dict:
        """
        Augment sequence alignment data.
        
        Args:
            alignment_data: Dictionary of alignment data
            noise_level: Level of noise to add
            
        Returns:
            Augmented alignment data
        """
        augmented_alignment = {}
        
        for key, value in alignment_data.items():
            if key in ['score', 'identity', 'coverage'] and isinstance(value, (int, float)):
                # Add noise to alignment scores
                noise = np.random.normal(0, noise_level * abs(value))
                augmented_alignment[key] = max(0, value + noise)  # Ensure non-negative
            else:
                augmented_alignment[key] = value
        
        return augmented_alignment
