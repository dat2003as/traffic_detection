from fastapi import APIRouter, Depends, Request
from typing import Dict, Any
import time
import psutil
import torch
import logging

from app.core.model import ModelManager
from app.dependencies import get_model_manager
from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Track application start time
app_start_time = time.time()

# Request metrics
request_metrics = {
    "total_requests": 0,
    "total_detections": 0,
    "total_errors": 0,
    "inference_times": []
}


@router.get("", summary="Health check")
async def health_check(
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Check if the API is healthy and ready to serve requests.
    
    **Returns:**
    - Health status and system information
    """
    uptime = time.time() - app_start_time
    
    # Check model status
    model_info = model_manager.get_model_info() if model_manager.model_loaded else {}
    
    # Get system info
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    # GPU info
    if torch.cuda.is_available():
        system_info["gpu_available"] = True
        system_info["gpu_name"] = torch.cuda.get_device_name(0)
        system_info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024**2
        system_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(0) / 1024**2
    else:
        system_info["gpu_available"] = False
    
    # Determine health status
    status_value = "healthy"
    if not model_manager.model_loaded:
        status_value = "unhealthy"
    elif system_info["memory_percent"] > 90 or system_info["disk_percent"] > 90:
        status_value = "degraded"
    
    return {
        "status": status_value,
        "timestamp": time.time(),
        "uptime_seconds": uptime,
        "model_loaded": model_manager.model_loaded,
        "model_info": model_info,
        "system_info": system_info,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


@router.get("/model", summary="Check model status")
async def check_model_status(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Get detailed information about the loaded model.
    
    **Returns:**
    - Model configuration and status
    """
    if not model_manager.model_loaded:
        return {
            "loaded": False,
            "error": "Model not loaded"
        }
    
    model_info = model_manager.get_model_info()
    
    return {
        "loaded": True,
        "model_path": settings.MODEL_PATH,
        "device": model_manager.device,
        "info": model_info,
        "config": {
            "confidence_threshold": settings.MODEL_CONFIDENCE,
            "iou_threshold": settings.MODEL_IOU_THRESHOLD,
            "image_size": settings.MODEL_IMG_SIZE
        }
    }


@router.get("/metrics", summary="Get performance metrics")
async def get_metrics():
    """
    Get API performance metrics.
    
    **Returns:**
    - Request counts, timing statistics, etc.
    """
    avg_inference_time = (
        sum(request_metrics["inference_times"]) / len(request_metrics["inference_times"])
        if request_metrics["inference_times"]
        else 0
    )
    
    uptime = time.time() - app_start_time
    requests_per_minute = (
        request_metrics["total_requests"] / (uptime / 60)
        if uptime > 0
        else 0
    )
    
    return {
        "total_requests": request_metrics["total_requests"],
        "total_detections": request_metrics["total_detections"],
        "total_errors": request_metrics["total_errors"],
        "average_inference_time_ms": avg_inference_time,
        "requests_per_minute": requests_per_minute,
        "uptime_seconds": uptime
    }


@router.post("/warmup", summary="Warmup the model")
async def warmup_model(
    iterations: int = 3,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Warmup the model with dummy predictions.
    Useful for GPU initialization.
    
    **Parameters:**
    - **iterations**: Number of warmup iterations
    
    **Returns:**
    - Warmup confirmation
    """
    try:
        await model_manager.warmup(num_iterations=iterations)
        return {
            "status": "success",
            "message": f"Model warmed up with {iterations} iterations"
        }
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return {
            "status": "failed",
            "message": str(e)
        }


@router.get("/ping", summary="Simple ping endpoint")
async def ping():
    """
    Simple ping endpoint for basic connectivity check.
    
    **Returns:**
    - Pong response
    """
    return {"message": "pong", "timestamp": time.time()}