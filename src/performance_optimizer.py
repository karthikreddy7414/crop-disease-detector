"""
Performance optimization utilities for DNA Crop Disease Identification System.
"""

import time
import functools
import joblib
import os
from typing import Dict, Any, Optional
import numpy as np

class ModelCache:
    """Cache for loaded models to improve performance."""
    
    def __init__(self):
        self._cache = {}
        self._load_times = {}
    
    def get_model(self, model_path: str) -> Optional[Any]:
        """Get model from cache or load it."""
        if model_path in self._cache:
            return self._cache[model_path]
        
        if os.path.exists(model_path):
            start_time = time.time()
            try:
                model = joblib.load(model_path)
                self._cache[model_path] = model
                self._load_times[model_path] = time.time() - start_time
                return model
            except Exception as e:
                print(f"Failed to load model {model_path}: {e}")
                return None
        return None
    
    def preload_models(self, model_paths: list):
        """Preload all models for faster access."""
        for path in model_paths:
            self.get_model(path)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_models': len(self._cache),
            'load_times': self._load_times,
            'total_cache_size': sum(len(str(model)) for model in self._cache.values())
        }

# Global model cache instance
model_cache = ModelCache()

def performance_monitor(func):
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance if execution time is significant
            if execution_time > 0.1:  # Log if > 100ms
                print(f"Performance: {func.__name__} took {execution_time:.3f}s")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Error in {func.__name__} after {execution_time:.3f}s: {e}")
            raise
    return wrapper

def optimize_image_processing(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Optimize image processing for faster analysis."""
    import cv2
    
    # Resize image if too large
    if image.shape[0] > 512 or image.shape[1] > 512:
        image = cv2.resize(image, target_size)
    
    # Convert to appropriate format
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Ensure RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def optimize_dna_sequence(sequence: str, max_length: int = 1000) -> str:
    """Optimize DNA sequence processing."""
    # Remove whitespace and convert to uppercase
    sequence = sequence.strip().upper()
    
    # Truncate if too long
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    
    # Remove invalid characters
    valid_chars = set('ATCG')
    sequence = ''.join(c for c in sequence if c in valid_chars)
    
    return sequence

def batch_process_predictions(features_list: list, model) -> list:
    """Process multiple predictions in batch for better performance."""
    if not features_list:
        return []
    
    try:
        # Convert to numpy array for batch processing
        features_array = np.array(features_list)
        
        # Make batch predictions
        predictions = model.predict(features_array)
        probabilities = model.predict_proba(features_array)
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'prediction': pred,
                'confidence': float(max(prob)),
                'probabilities': dict(zip(model.classes_, prob))
            })
        
        return results
    except Exception as e:
        print(f"Batch processing error: {e}")
        return []

def optimize_memory_usage():
    """Optimize memory usage by clearing unused variables."""
    import gc
    gc.collect()

def get_system_performance_metrics() -> Dict[str, Any]:
    """Get current system performance metrics."""
    import psutil
    
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'disk_percent': psutil.disk_usage('/').percent,
        'cache_stats': model_cache.get_cache_stats()
    }

def initialize_performance_optimizations():
    """Initialize all performance optimizations."""
    # Preload common models
    model_paths = [
        'models/trained_models/dna_classifier.pkl',
        'models/trained_models/image_classifier.pkl',
        'models/trained_models/fusion_model.pkl'
    ]
    
    model_cache.preload_models(model_paths)
    
    # Set numpy threading for better performance
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    print("Performance optimizations initialized")

# Initialize optimizations when module is imported
try:
    initialize_performance_optimizations()
except Exception as e:
    print(f"Performance optimization initialization failed: {e}")
