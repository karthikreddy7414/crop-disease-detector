"""
Database configuration and initialization.
"""

import sqlite3
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def init_db():
    """Initialize database."""
    try:
        # Create database connection
        conn = sqlite3.connect('dna_crop.db')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dna_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                sequence_data TEXT NOT NULL,
                sequence_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_path TEXT NOT NULL,
                image_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                dna_sequence_id INTEGER,
                image_id INTEGER,
                predicted_disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                dna_confidence REAL,
                image_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (dna_sequence_id) REFERENCES dna_sequences (id),
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
