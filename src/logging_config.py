"""
Enhanced logging configuration for DNA Crop Disease Identification System.
"""

import logging
import sys
from datetime import datetime
import os

def setup_logging():
    """Setup comprehensive logging for the system."""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Setup root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler for all logs
            logging.FileHandler(f'logs/system_{datetime.now().strftime("%Y%m%d")}.log'),
            # Error file handler
            logging.FileHandler(f'logs/errors_{datetime.now().strftime("%Y%m%d")}.log'),
        ]
    )
    
    # Set error file handler to only log errors
    error_handler = logging.FileHandler(f'logs/errors_{datetime.now().strftime("%Y%m%d")}.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Add error handler to root logger
    logging.getLogger().addHandler(error_handler)
    
    # Create specific loggers for different components
    dna_logger = logging.getLogger('dna_analysis')
    image_logger = logging.getLogger('image_processing')
    api_logger = logging.getLogger('api')
    model_logger = logging.getLogger('models')
    
    # Set log levels
    dna_logger.setLevel(logging.INFO)
    image_logger.setLevel(logging.INFO)
    api_logger.setLevel(logging.INFO)
    model_logger.setLevel(logging.INFO)
    
    return {
        'dna': dna_logger,
        'image': image_logger,
        'api': api_logger,
        'model': model_logger
    }

def log_api_request(endpoint: str, method: str, status_code: int, response_time: float):
    """Log API request details."""
    api_logger = logging.getLogger('api')
    api_logger.info(f"API Request - {method} {endpoint} - Status: {status_code} - Time: {response_time:.3f}s")

def log_model_prediction(model_name: str, prediction: str, confidence: float, features_count: int):
    """Log model prediction details."""
    model_logger = logging.getLogger('models')
    model_logger.info(f"Model Prediction - {model_name} - Result: {prediction} - Confidence: {confidence:.3f} - Features: {features_count}")

def log_error(component: str, error: Exception, context: str = ""):
    """Log error details."""
    error_logger = logging.getLogger(component)
    error_logger.error(f"Error in {component} - {str(error)} - Context: {context}")

def log_system_metrics():
    """Log system performance metrics."""
    import psutil
    import os
    
    # Get system metrics
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    
    # Log metrics
    metrics_logger = logging.getLogger('system')
    metrics_logger.info(f"System Metrics - CPU: {cpu_percent}% - Memory: {memory_percent}% - Disk: {disk_percent}%")

# Initialize logging when module is imported
loggers = setup_logging()
