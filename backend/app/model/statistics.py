"""
Statistics Pydantic models
"""
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime


class StatisticsSummary(BaseModel):
    """Overall statistics summary."""
    total_processed_images: int
    total_processed_videos: int
    total_detections: int
    total_violations: int
    violations_by_type: Dict[str, int]
    violations_by_severity: Dict[str, int]


class TimeSeriesData(BaseModel):
    """Time-series data point."""
    timestamp: datetime
    value: int


class ViolationTrend(BaseModel):
    """Violation trend data."""
    violation_type: str
    data_points: List[TimeSeriesData]


class HeatmapData(BaseModel):
    """Heatmap data for visualization."""
    x: List[float]
    y: List[float]
    intensity: List[float]