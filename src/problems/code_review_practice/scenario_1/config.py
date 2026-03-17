import os
import json
import sys
import hashlib


# Application settings
APP_NAME = "DocProcessor"
VERSION = "2.1.0"
DEBUG = True

# Database configuration
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "docprocessor"
DB_USER = "admin"
DB_PASSWORD = "super_secret_password_123!"

# API settings
SECRET_KEY = "a1b2c3d4e5f6789xyzinsecurekey"
TOKEN_EXPIRY = 3600
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

# Rate limiting
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 100

# Processing settings
SUPPORTED_FORMATS = [".txt", ".md", ".csv", ".json"]
MAX_DOCUMENT_LENGTH = 1_000_000
WORKER_COUNT = os.cpu_count()

# Logging
LOG_LEVEL = "DEBUG"
LOG_FILE = "/var/log/docprocessor/app.log"


def get_db_connection_string():
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def load_config_from_file(path):
    """Load additional config from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data
