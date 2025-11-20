"""
Pydantic models for request/response validation
"""
from .common import BoundingBox, Point, ImageSize
from .detection import (
    Detection,
    Violation,
    ViolationSeverity,
    DetectionSummary,
    ImageDetectionRequest,
    ImageDetectionResponse,
    BatchDetectionRequest,
    BatchDetectionResponse
)
from .video import (
    VideoInfo,
    VideoProcessingRequest,
    VideoFrame,
    ViolationTimeline,
    VideoDetectionResponse,
    VideoProcessingStatus
)
from .statistics import (
    StatisticsSummary,
    TimeSeriesData,
    ViolationTrend,
    HeatmapData
)
from .health import (
    HealthStatus,
    HealthCheckResponse,
    MetricsResponse
)
from .error import ErrorResponse
from .db import (
                VideoTask,
                DetectionResult
)
__all__ = [
    'BoundingBox', 'Point', 'ImageSize',
    'Detection', 'Violation', 'ViolationSeverity', 'DetectionSummary',
    'ImageDetectionRequest', 'ImageDetectionResponse',
    'BatchDetectionRequest', 'BatchDetectionResponse',
    'VideoInfo', 'VideoProcessingRequest', 'VideoFrame',
    'ViolationTimeline', 'VideoDetectionResponse', 'VideoProcessingStatus',
    'StatisticsSummary', 'TimeSeriesData', 'ViolationTrend', 'HeatmapData',
    'HealthStatus', 'HealthCheckResponse', 'MetricsResponse',
    'ErrorResponse',
    'VideoTask', 'DetectionResult'
]