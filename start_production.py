#!/usr/bin/env python3
"""
Production startup script for DNA Crop Disease Identification System.
"""

import os
import sys
import subprocess
import time
import signal
import logging
from pathlib import Path

def setup_logging():
    """Setup logging for startup script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('startup.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'fastapi', 'uvicorn', 'opencv-python', 'scikit-learn', 
        'biopython', 'pandas', 'Pillow', 'scikit-image', 
        'albumentations', 'numpy', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logging.error(f"Missing required packages: {missing_packages}")
        logging.info("Install missing packages with: pip install -r requirements.txt")
        return False
    
    logging.info("All dependencies are installed")
    return True

def check_models():
    """Check if trained models exist."""
    model_path = Path("models/trained_models")
    required_models = [
        "dna_classifier.pkl",
        "image_classifier.pkl", 
        "fusion_model.pkl",
        "ensemble_model.pkl"
    ]
    
    missing_models = []
    for model in required_models:
        if not (model_path / model).exists():
            missing_models.append(model)
    
    if missing_models:
        logging.warning(f"Missing trained models: {missing_models}")
        logging.info("Train models with: python train_models.py")
        return False
    
    logging.info("All trained models are available")
    return True

def create_directories():
    """Create necessary directories."""
    directories = ['logs', 'static', 'models/trained_models']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def check_port_availability(port):
    """Check if port is available."""
    import socket
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def start_server():
    """Start the FastAPI server."""
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    workers = int(os.getenv('WORKERS', 4))
    
    # Check port availability
    if not check_port_availability(port):
        logging.error(f"Port {port} is already in use")
        return False
    
    # Build uvicorn command
    cmd = [
        sys.executable, '-m', 'uvicorn',
        'src.main:app',
        '--host', host,
        '--port', str(port),
        '--workers', str(workers)
    ]
    
    # Add reload for development
    if os.getenv('ENVIRONMENT', 'production') == 'development':
        cmd.append('--reload')
        logging.info("Starting in development mode with auto-reload")
    else:
        logging.info("Starting in production mode")
    
    logging.info(f"Starting server on {host}:{port} with {workers} workers")
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Start the server
        process = subprocess.Popen(cmd)
        
        # Wait for server to start
        time.sleep(3)
        
        if process.poll() is None:
            logging.info("Server started successfully")
            return process
        else:
            logging.error("Server failed to start")
            return None
            
    except Exception as e:
        logging.error(f"Error starting server: {e}")
        return None

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main startup function."""
    setup_logging()
    
    logging.info("Starting DNA Crop Disease Identification System")
    logging.info("=" * 50)
    
    # Pre-flight checks
    logging.info("Performing pre-flight checks...")
    
    if not check_dependencies():
        sys.exit(1)
    
    if not check_models():
        logging.warning("Some models are missing, but continuing...")
    
    create_directories()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server
    server_process = start_server()
    
    if server_process:
        logging.info("System is ready!")
        logging.info("Web interface: http://localhost:8000/")
        logging.info("API documentation: http://localhost:8000/docs")
        logging.info("Health check: http://localhost:8000/health")
        logging.info("Press Ctrl+C to stop the server")
        
        try:
            # Keep the script running
            server_process.wait()
        except KeyboardInterrupt:
            logging.info("Shutdown requested by user")
        finally:
            logging.info("Shutting down server...")
            server_process.terminate()
            server_process.wait()
            logging.info("Server stopped")
    else:
        logging.error("Failed to start server")
        sys.exit(1)

if __name__ == "__main__":
    main()
