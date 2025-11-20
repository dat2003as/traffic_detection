import os

from pathlib import Path
from typing  import List

from pydantic import field_validator
from .design_pattern import singleton
from dotenv import load_dotenv

load_dotenv()


@singleton
class AppSettings:
    APP_NAME: str = "Traffic Violation Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3002", "http://localhost:5173","http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # ML Model
    MODEL_PATH: str = "ml_models/yolov8_best.pt"
    MODEL_CONFIDENCE: float = 0.5
    MODEL_IOU_THRESHOLD: float = 0.45
    MODEL_DEVICE: str = "cuda"  # cuda or cpu
    MODEL_IMG_SIZE: int = 640
    
    # File Storage
    UPLOAD_DIR: str = "storage/uploads"
    PROCESSED_DIR: str = "storage/processed"
    TEMP_DIR: str = "storage/temp"
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov"]
    
    # Database (Optional)
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    
    # Redis (Optional)
    REDIS_URL: str = os.getenv("REDIS_URL")
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    # Celery (Optional)
    MODEL_PATH:str = os.getenv("MODEL_PATH")
    MODEL_CONFIDENCE: float= 0.5
    MODEL_IOU_THRESHOLD: float = 0.45
    MODEL_DEVICE:str = os.getenv("MODEL_DEVICE")   
    MODEL_IMG_SIZE: int =640
    

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Feature Flags
    ENABLE_VIDEO_PROCESSING: bool = True
    ENABLE_BATCH_DETECTION: bool = True
    ENABLE_REALTIME_DETECTION: bool = True
    ENABLE_ANALYTICS: bool = True
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("ALLOWED_IMAGE_EXTENSIONS", "ALLOWED_VIDEO_EXTENSIONS", mode="before")  # ← THAY ĐỔI
    @classmethod
    def parse_extensions(cls, v):
        """Parse file extensions from string or list."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.UPLOAD_DIR,
            self.PROCESSED_DIR,
            self.TEMP_DIR,
            "logs",
            "ml_models",
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    
settings = AppSettings()
settings.create_directories()