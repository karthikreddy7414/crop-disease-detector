# Production deployment configuration for DNA Crop Disease Identification System

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000
WORKERS = 4
RELOAD = False  # Set to False for production

# Model Configuration
MODEL_PATH = "models/trained_models/"
MODEL_CACHE_SIZE = 100
MODEL_PRELOAD = True

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/system.log"
ERROR_LOG_FILE = "logs/errors.log"
LOG_ROTATION_SIZE = "10MB"
LOG_BACKUP_COUNT = 5

# Performance Configuration
MAX_REQUEST_SIZE = "16MB"
TIMEOUT_SECONDS = 30
KEEPALIVE_SECONDS = 2

# Security Configuration
CORS_ORIGINS = ["*"]  # Configure specific origins for production
CORS_CREDENTIALS = True
CORS_METHODS = ["GET", "POST", "PUT", "DELETE"]
CORS_HEADERS = ["*"]

# Database Configuration (if using database)
DATABASE_URL = "sqlite:///./crop_disease.db"
DATABASE_POOL_SIZE = 10
DATABASE_MAX_OVERFLOW = 20

# Monitoring Configuration
ENABLE_METRICS = True
METRICS_INTERVAL = 60  # seconds
HEALTH_CHECK_INTERVAL = 30  # seconds

# File Upload Configuration
MAX_FILE_SIZE = "10MB"
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
ALLOWED_DNA_FORMATS = ["text/plain", "application/octet-stream"]

# Cache Configuration
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 1000

# Error Handling
SHOW_ERROR_DETAILS = False  # Set to False for production
CUSTOM_ERROR_PAGES = True

# SSL Configuration (for HTTPS)
SSL_CERT_PATH = None  # Path to SSL certificate
SSL_KEY_PATH = None   # Path to SSL private key

# Environment-specific settings
ENVIRONMENT = "production"  # development, staging, production
DEBUG = False
TESTING = False

# Backup Configuration
BACKUP_ENABLED = True
BACKUP_INTERVAL = 24  # hours
BACKUP_RETENTION_DAYS = 30

# Alert Configuration
ALERT_EMAIL = None  # Email for system alerts
ALERT_WEBHOOK = None  # Webhook URL for alerts
ALERT_THRESHOLDS = {
    "cpu_percent": 80,
    "memory_percent": 85,
    "disk_percent": 90,
    "error_rate_percent": 5
}
