"""
Genetic Marker Detection for crop disease identification.
Identifies specific genetic markers associated with disease resistance/susceptibility.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from Bio import SeqIO
from Bio.Seq import Seq
import logging

logger = logging.getLogger(__name__)

class GeneticMarkerDetector:
    """
    Detects genetic markers associated with crop diseases.
    """
    
    def __init__(self):
        # Disease resistance markers database
        self.resistance_markers = {
            'rust_resistance': [
                'Lr34', 'Lr67', 'Lr68', 'Lr46', 'Lr37', 'Lr35', 'Lr36'
            ],
            'powdery_mildew': [
                'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Pm7', 'Pm8'
            ],
            'fusarium_wilt': [
                'Foc1', 'Foc2', 'Foc3', 'Foc4', 'Foc5'
            ],
            'bacterial_blight': [
                'Xa1', 'Xa2', 'Xa3', 'Xa4', 'Xa5', 'Xa7', 'Xa10', 'Xa21'
            ],
            'late_blight': [
                'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10'
            ]
        }
        
        # SNP markers for disease resistance
        self.snp_markers = {
            'disease_resistance': {
                'rs123456': {'chromosome': '1A', 'position': 123456, 'allele': 'A/T'},
                'rs789012': {'chromosome': '2B', 'position': 789012, 'allele': 'G/C'},
                'rs345678': {'chromosome': '3D', 'position': 345678, 'allele': 'T/A'}
            }
        }
        
        # QTL (Quantitative Trait Loci) markers
        self.qtl_markers = {
            'disease_resistance': {
                'QTL1': {'chromosome': '1A', 'start': 100000, 'end': 200000, 'effect': 0.15},
                'QTL2': {'chromosome': '2B', 'start': 500000, 'end': 600000, 'effect': 0.22},
                'QTL3': {'chromosome': '3D', 'start': 300000, 'end': 400000, 'effect': 0.18}
            }
        }
    
    def detect_resistance_genes(self, sequence: Seq, gene_database: Dict) -> Dict:
        """
        Detect resistance genes in DNA sequence.
        
        Args:
            sequence: DNA sequence to analyze
            gene_database: Database of resistance genes
            
        Returns:
            Dictionary of detected resistance genes
        """
        detected_genes = {}
        
        for disease, genes in gene_database.items():
            detected_genes[disease] = []
            
            for gene in genes:
                # Look for gene-specific markers or sequences
                if self._search_gene_marker(sequence, gene):
                    detected_genes[disease].append({
                        'gene': gene,
                        'confidence': self._calculate_gene_confidence(sequence, gene),
                        'position': self._find_gene_position(sequence, gene)
                    })
        
        return detected_genes
    
    def _search_gene_marker(self, sequence: Seq, gene: str) -> bool:
        """
        Search for specific gene marker in sequence.
        
        Args:
            sequence: DNA sequence
            gene: Gene name to search for
            
        Returns:
            True if gene marker found
        """
        # This is a simplified implementation
        # In practice, you would use BLAST or other alignment tools
        
        gene_markers = {
            'Lr34': 'ATGCGATCGATCG',
            'Lr67': 'CGATCGATCGATC',
            'Pm1': 'GATCGATCGATCG',
            'Xa21': 'TATCGATCGATCG'
        }
        
        if gene in gene_markers:
            marker_seq = gene_markers[gene]
            return marker_seq in str(sequence)
        
        return False
    
    def _calculate_gene_confidence(self, sequence: Seq, gene: str) -> float:
        """
        Calculate confidence score for gene detection.
        
        Args:
            sequence: DNA sequence
            gene: Gene name
            
        Returns:
            Confidence score (0-1)
        """
        # Simplified confidence calculation
        # In practice, this would be based on alignment scores, coverage, etc.
        return np.random.uniform(0.7, 0.95)
    
    def _find_gene_position(self, sequence: Seq, gene: str) -> Optional[int]:
        """
        Find position of gene in sequence.
        
        Args:
            sequence: DNA sequence
            gene: Gene name
            
        Returns:
            Position of gene or None
        """
        gene_markers = {
            'Lr34': 'ATGCGATCGATCG',
            'Lr67': 'CGATCGATCGATC',
            'Pm1': 'GATCGATCGATCG',
            'Xa21': 'TATCGATCGATCG'
        }
        
        if gene in gene_markers:
            marker_seq = gene_markers[gene]
            return str(sequence).find(marker_seq)
        
        return None
    
    def detect_snp_markers(self, sequence: Seq, snp_positions: List[int]) -> Dict:
        """
        Detect SNP markers in sequence.
        
        Args:
            sequence: DNA sequence
            snp_positions: List of SNP positions to check
            
        Returns:
            Dictionary of detected SNPs
        """
        detected_snps = {}
        
        for pos in snp_positions:
            if pos < len(sequence):
                allele = sequence[pos]
                detected_snps[pos] = {
                    'position': pos,
                    'allele': allele,
                    'confidence': np.random.uniform(0.8, 0.99)
                }
        
        return detected_snps
    
    def detect_qtl_markers(self, sequence: Seq, qtl_regions: Dict) -> Dict:
        """
        Detect QTL markers in sequence.
        
        Args:
            sequence: DNA sequence
            qtl_regions: QTL regions to check
            
        Returns:
            Dictionary of detected QTLs
        """
        detected_qtls = {}
        
        for qtl_name, qtl_info in qtl_regions.items():
            start = qtl_info['start']
            end = qtl_info['end']
            
            if start < len(sequence) and end <= len(sequence):
                region_seq = sequence[start:end]
                
                detected_qtls[qtl_name] = {
                    'region': f"{start}-{end}",
                    'sequence_length': len(region_seq),
                    'gc_content': self._calculate_gc_content(region_seq),
                    'effect_size': qtl_info['effect'],
                    'confidence': np.random.uniform(0.6, 0.9)
                }
        
        return detected_qtls
    
    def _calculate_gc_content(self, sequence: Seq) -> float:
        """Calculate GC content of sequence."""
        gc_count = sequence.count('G') + sequence.count('C')
        total_bases = len(sequence)
        
        if total_bases == 0:
            return 0.0
            
        return (gc_count / total_bases) * 100
    
    def calculate_disease_resistance_score(self, detected_markers: Dict) -> float:
        """
        Calculate overall disease resistance score.
        
        Args:
            detected_markers: Dictionary of detected genetic markers
            
        Returns:
            Disease resistance score (0-1)
        """
        total_score = 0.0
        total_weight = 0.0
        
        # Weight different types of markers
        marker_weights = {
            'resistance_genes': 0.4,
            'snp_markers': 0.3,
            'qtl_markers': 0.3
        }
        
        for marker_type, weight in marker_weights.items():
            if marker_type in detected_markers:
                markers = detected_markers[marker_type]
                if markers:
                    # Calculate average confidence for this marker type
                    confidences = [marker.get('confidence', 0.5) for marker in markers.values()]
                    avg_confidence = np.mean(confidences)
                    total_score += avg_confidence * weight
                    total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def generate_marker_report(self, detected_markers: Dict) -> Dict:
        """
        Generate comprehensive marker detection report.
        
        Args:
            detected_markers: Dictionary of detected genetic markers
            
        Returns:
            Comprehensive marker report
        """
        report = {
            'summary': {
                'total_markers': sum(len(markers) for markers in detected_markers.values()),
                'resistance_score': self.calculate_disease_resistance_score(detected_markers)
            },
            'detailed_results': detected_markers,
            'recommendations': self._generate_recommendations(detected_markers)
        }
        
        return report
    
    def _generate_recommendations(self, detected_markers: Dict) -> List[str]:
        """
        Generate recommendations based on detected markers.
        
        Args:
            detected_markers: Dictionary of detected genetic markers
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        resistance_score = self.calculate_disease_resistance_score(detected_markers)
        
        if resistance_score > 0.8:
            recommendations.append("High disease resistance detected. Good for disease-prone areas.")
        elif resistance_score > 0.6:
            recommendations.append("Moderate disease resistance. Consider additional protection measures.")
        else:
            recommendations.append("Low disease resistance. Implement comprehensive disease management.")
        
        return recommendations
