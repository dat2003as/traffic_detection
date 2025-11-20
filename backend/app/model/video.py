"""
Video processing Pydantic models
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from .detection import Detection, Violation, DetectionSummary


class VideoInfo(BaseModel):
    """Video file information."""
    path: str
    width: int
    height: int
    fps: int
    total_frames: int
    duration_seconds: float
    processed_frames: int


class VideoProcessingRequest(BaseModel):
    """Request model for video processing."""
    conf_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    process_every_n_frames: int = Field(1, ge=1, le=10)
    save_annotated: bool = True


class VideoFrame(BaseModel):
    """Detection results for a single video frame."""
    frame_number: int
    timestamp: float
    detections: List[Detection]
    violations: List[Violation]


class ViolationTimeline(BaseModel):
    """Violation occurrence timeline."""
    timestamp: float
    frame_number: int
    violation_count: int
    violations: List[Violation]


class VideoDetectionResponse(BaseModel):
    """Response model for video detection."""
    video_info: VideoInfo
    output_path: Optional[str]
    summary: DetectionSummary
    violations_timeline: List[ViolationTimeline]
    processing_time_ms: float


class VideoProcessingStatus(BaseModel):
    """Status of video processing task."""
    task_id: str
    status: str
    progress: float = Field(..., ge=0.0, le=1.0)
    current_frame: int
    total_frames: int
    message: Optional[str] = None
    result: Optional[VideoDetectionResponse] = None