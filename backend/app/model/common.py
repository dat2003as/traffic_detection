"""
Common Pydantic models
"""
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class Point(BaseModel):
    """2D point coordinates."""
    x: float
    y: float


class ImageSize(BaseModel):
    """Image dimensions."""
    width: int
    height: int