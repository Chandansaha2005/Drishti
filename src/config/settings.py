"""
Configuration settings for DRISTI system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration for DRISTI application."""
    
    # ========================================================================
    # PROJECT METADATA
    # ========================================================================
    PROJECT_NAME = "DRISTI - Lost Person Detection"
    VERSION = "2.0.0"
    DESCRIPTION = "Face recognition system for finding lost persons in CCTV footage"
    
    # ========================================================================
    # SERVER SETTINGS
    # ========================================================================
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # ========================================================================
    # DIRECTORIES
    # ========================================================================
    # Base directories
    BASE_DIR = Path(__file__).parent.parent.parent  # Root project directory
    DATA_DIR = BASE_DIR / "data"
    
    # Core directories
    UPLOAD_DIR = DATA_DIR / "uploads"
    RESULTS_DIR = DATA_DIR / "results"
    VIDEO_DIR = BASE_DIR / "CCTVS"  # CCTV videos
    LOG_DIR = BASE_DIR / "logs"
    
    # ========================================================================
    # FACE RECOGNITION SETTINGS
    # ========================================================================
    SIMILARITY_THRESHOLD = 0.60  # Confidence threshold for matches
    DETECTION_CONFIDENCE = 0.50  # MediaPipe detection confidence
    FRAME_SKIP = 5  # Process every Nth frame
    
    # ========================================================================
    # DATABASE SETTINGS (for future use)
    # ========================================================================
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dristi.db")
    
    # ========================================================================
    # SECURITY SETTINGS (for future use)
    # ========================================================================
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    @staticmethod
    def create_directories():
        """Create all necessary directories."""
        dirs = [
            Settings.UPLOAD_DIR,
            Settings.RESULTS_DIR,
            Settings.VIDEO_DIR,
            Settings.LOG_DIR,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)


# Initialize settings
settings = Settings()
Settings.create_directories()
