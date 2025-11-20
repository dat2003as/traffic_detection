"""
SQLAlchemy models for database tables
"""
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Text
from sqlalchemy.sql import func
from datetime import datetime

from app.db.database import Base


class DetectionResult(Base):
    """Store detection results."""
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)  # 'image' or 'video'
    file_path = Column(String(500))
    
    # Detection summary
    total_detections = Column(Integer, default=0)
    total_violations = Column(Integer, default=0)
    
    # Detailed results (JSON)
    detections = Column(JSON)
    violations = Column(JSON)
    
    # Performance
    processing_time_ms = Column(Float)
    model_confidence = Column(Float)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    ip_address = Column(String(50))
    user_agent = Column(Text)


class VideoTask(Base):
    """Store video processing tasks."""
    __tablename__ = "video_tasks"
    
    id = Column(String(36), primary_key=True)  # UUID
    video_id = Column(String(36), index=True)
    video_path = Column(String(500))
    
    # Status
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)
    current_frame = Column(Integer, default=0)
    total_frames = Column(Integer, default=0)
    
    # Results
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

