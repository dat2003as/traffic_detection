"""
Health check Pydantic models
"""
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
from enum import Enum


class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    model_loaded: bool
    model_info: Dict[str, Any]
    system_info: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Performance metrics response."""
    total_requests: int
    total_detections: int
    average_inference_time_ms: float
    requests_per_minute: float
    errors: int