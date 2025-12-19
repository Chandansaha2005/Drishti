import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Settings
    API_V1_PREFIX = "/api/v1"
    PROJECT_NAME = "DRISTI Face Recognition API"
    VERSION = "1.0.0"
    
    # Server Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Database Settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dristi.db")
    
    # File Storage
    UPLOAD_DIR = "../data/uploads"
    RESULTS_DIR = "../data/results"
    MODELS_DIR = "../models"
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # Face Recognition Settings
    SIMILARITY_THRESHOLD = 0.75
    DETECTION_CONFIDENCE = 0.7
    FRAME_SKIP = 3
    
    # Create directories
    @staticmethod
    def create_directories():
        dirs = [
            Settings.UPLOAD_DIR,
            Settings.RESULTS_DIR,
            f"{Settings.RESULTS_DIR}/evidence",
            f"{Settings.RESULTS_DIR}/reports",
            f"{Settings.RESULTS_DIR}/snapshots",
            f"{Settings.RESULTS_DIR}/alerts",
            Settings.MODELS_DIR,
            "logs"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

settings = Settings()
settings.create_directories()