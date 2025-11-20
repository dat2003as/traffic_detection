"""
Detection-related Pydantic models
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from .common import BoundingBox, Point, ImageSize


class ViolationSeverity(str, Enum):
    """Violation severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Detection(BaseModel):
    """Single object detection."""
    id: int
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    center: Point


class Violation(BaseModel):
    """Detected violation."""
    type: str = Field(..., description="Type of violation")
    severity: ViolationSeverity
    description: str
    bbox: BoundingBox
    confidence: float
    details: Dict[str, Any] = Field(default_factory=dict)


class DetectionSummary(BaseModel):
    """Summary of detection results."""
    total_detections: int
    total_violations: int
    by_class: Dict[str, int]
    violations_by_type: Dict[str, int]


class ImageDetectionRequest(BaseModel):
    """Request model for image detection."""
    conf_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    return_annotated: bool = True


class ImageDetectionResponse(BaseModel):
    """Response model for image detection."""
    image_size: ImageSize
    detections: List[Detection]
    violations: List[Violation]
    summary: DetectionSummary
    processing_time_ms: float
    annotated_image_url: Optional[str] = None


class BatchDetectionRequest(BaseModel):
    """Request for batch image detection."""
    conf_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    return_annotated: bool = False


class BatchDetectionResponse(BaseModel):
    """Response for batch detection."""
    total_images: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    processing_time_ms: float