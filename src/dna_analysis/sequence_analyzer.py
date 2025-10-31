"""
DNA Sequence Analyzer for crop disease identification.
Handles DNA sequence processing, alignment, and analysis.
"""

import numpy as np
import pandas as pd
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DNASequenceAnalyzer:
    """
    Analyzes DNA sequences for disease identification.
    """
    
    def __init__(self):
        self.aligner = Align.PairwiseAligner()
        self.aligner.mode = 'global'
        self.aligner.match_score = 2
        self.aligner.mismatch_score = -1
        self.aligner.open_gap_score = -2
        self.aligner.extend_gap_score = -0.5
        
    def load_sequence(self, sequence_path: str) -> SeqRecord:
        """
        Load DNA sequence from file.
        
        Args:
            sequence_path: Path to sequence file (FASTA, FASTQ, etc.)
            
        Returns:
            SeqRecord object
        """
        try:
            sequence = SeqIO.read(sequence_path, "fasta")
            logger.info(f"Loaded sequence: {sequence.id}")
            return sequence
        except Exception as e:
            logger.error(f"Error loading sequence: {e}")
            raise
    
    def preprocess_sequence(self, sequence: SeqRecord) -> SeqRecord:
        """
        Preprocess DNA sequence for analysis.
        
        Args:
            sequence: Input sequence record
            
        Returns:
            Preprocessed sequence record
        """
        # Convert to uppercase
        processed_seq = sequence.seq.upper()
        
        # Remove ambiguous bases (optional)
        # processed_seq = processed_seq.replace('N', '')
        
        # Create new sequence record
        processed_record = SeqRecord(
            processed_seq,
            id=sequence.id,
            description=sequence.description
        )
        
        return processed_record
    
    def calculate_gc_content(self, sequence: Seq) -> float:
        """
        Calculate GC content of sequence.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            GC content as percentage
        """
        gc_count = sequence.count('G') + sequence.count('C')
        total_bases = len(sequence)
        
        if total_bases == 0:
            return 0.0
            
        return (gc_count / total_bases) * 100
    
    def find_motifs(self, sequence: Seq, motif_patterns: List[str]) -> Dict[str, List[int]]:
        """
        Find specific motifs in DNA sequence.
        
        Args:
            sequence: DNA sequence to search
            motif_patterns: List of motif patterns to find
            
        Returns:
            Dictionary mapping motifs to their positions
        """
        motifs_found = {}
        
        for motif in motif_patterns:
            positions = []
            start = 0
            
            while True:
                pos = sequence.find(motif, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            
            motifs_found[motif] = positions
        
        return motifs_found
    
    def align_sequences(self, query_seq: Seq, reference_seq: Seq) -> Dict:
        """
        Align query sequence with reference sequence.
        
        Args:
            query_seq: Query DNA sequence
            reference_seq: Reference DNA sequence
            
        Returns:
            Alignment results dictionary
        """
        try:
            alignments = self.aligner.align(query_seq, reference_seq)
            best_alignment = alignments[0]
            
            alignment_info = {
                'score': best_alignment.score,
                'query_start': best_alignment.query_start,
                'query_end': best_alignment.query_end,
                'target_start': best_alignment.target_start,
                'target_end': best_alignment.target_end,
                'identity': self._calculate_identity(best_alignment),
                'coverage': self._calculate_coverage(best_alignment, len(query_seq))
            }
            
            return alignment_info
            
        except Exception as e:
            logger.error(f"Alignment error: {e}")
            return None
    
    def _calculate_identity(self, alignment) -> float:
        """Calculate sequence identity from alignment."""
        matches = 0
        total = 0
        
        for i in range(len(alignment.query)):
            if alignment.query[i] != '-' and alignment.target[i] != '-':
                total += 1
                if alignment.query[i] == alignment.target[i]:
                    matches += 1
        
        return (matches / total * 100) if total > 0 else 0.0
    
    def _calculate_coverage(self, alignment, query_length: int) -> float:
        """Calculate query coverage from alignment."""
        aligned_length = alignment.query_end - alignment.query_start
        return (aligned_length / query_length * 100) if query_length > 0 else 0.0
    
    def extract_features(self, sequence: SeqRecord) -> Dict:
        """
        Extract features from DNA sequence for ML models.
        
        Args:
            sequence: DNA sequence record
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'length': len(sequence.seq),
            'gc_content': self.calculate_gc_content(sequence.seq),
            'at_content': 100 - self.calculate_gc_content(sequence.seq),
            'n_content': (sequence.seq.count('N') / len(sequence.seq)) * 100,
            'complexity': self._calculate_complexity(sequence.seq),
            'dinucleotide_freq': self._calculate_dinucleotide_freq(sequence.seq),
            'trinucleotide_freq': self._calculate_trinucleotide_freq(sequence.seq)
        }
        
        return features
    
    def _calculate_complexity(self, sequence: Seq) -> float:
        """Calculate sequence complexity."""
        unique_kmers = set()
        k = 3  # k-mer size
        
        for i in range(len(sequence) - k + 1):
            unique_kmers.add(str(sequence[i:i+k]))
        
        max_possible = 4 ** k
        return len(unique_kmers) / max_possible
    
    def _calculate_dinucleotide_freq(self, sequence: Seq) -> Dict[str, float]:
        """Calculate dinucleotide frequencies."""
        dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC',
                        'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        
        freqs = {}
        total_dinuc = len(sequence) - 1
        
        for dinuc in dinucleotides:
            count = sequence.count(dinuc)
            freqs[dinuc] = count / total_dinuc if total_dinuc > 0 else 0.0
        
        return freqs
    
    def _calculate_trinucleotide_freq(self, sequence: Seq) -> Dict[str, float]:
        """Calculate trinucleotide frequencies."""
        trinucleotides = ['AAA', 'AAT', 'AAG', 'AAC', 'ATA', 'ATT', 'ATG', 'ATC',
                         'AGA', 'AGT', 'AGG', 'AGC', 'ACA', 'ACT', 'ACG', 'ACC',
                         'TAA', 'TAT', 'TAG', 'TAC', 'TTA', 'TTT', 'TTG', 'TTC',
                         'TGA', 'TGT', 'TGG', 'TGC', 'TCA', 'TCT', 'TCG', 'TCC',
                         'GAA', 'GAT', 'GAG', 'GAC', 'GTA', 'GTT', 'GTG', 'GTC',
                         'GGA', 'GGT', 'GGG', 'GGC', 'GCA', 'GCT', 'GCG', 'GCC',
                         'CAA', 'CAT', 'CAG', 'CAC', 'CTA', 'CTT', 'CTG', 'CTC',
                         'CGA', 'CGT', 'CGG', 'CGC', 'CCA', 'CCT', 'CCG', 'CCC']
        
        freqs = {}
        total_trinuc = len(sequence) - 2
        
        for trinuc in trinucleotides:
            count = sequence.count(trinuc)
            freqs[trinuc] = count / total_trinuc if total_trinuc > 0 else 0.0
        
        return freqs
