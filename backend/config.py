"""Database connection configuration for the Smart Scheduler MySQL backend."""

import os

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),       # MySQL server hostname
    "user":     os.getenv("DB_USER",     "root"),            # MySQL username
    "password": os.getenv("DB_PASSWORD", "1234"),            # MySQL password
    "database": os.getenv("DB_NAME",     "smart_scheduler"), # Target database name
}