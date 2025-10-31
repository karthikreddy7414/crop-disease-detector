"""
Models Module for DNA-based Crop Disease Identification.

This module contains the multimodal fusion models that combine
DNA analysis and computer vision for comprehensive disease identification.
"""

from .fusion_model_simple import MultimodalFusionModel
from .ensemble_model import EnsembleDiseaseClassifier

__all__ = [
    "MultimodalFusionModel",
    "EnsembleDiseaseClassifier"
]
