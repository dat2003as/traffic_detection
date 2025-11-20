"""
Shared dependencies for FastAPI endpoints.
These can be injected into route handlers using Depends().
"""

from fastapi import Request, HTTPException, status
from typing import Optional
import logging

from fastapi.params import Depends

from app.core.model import ModelManager
from app.core.detector import Detector

logger = logging.getLogger(__name__)


def get_model_manager(request: Request) -> ModelManager:
    """
    Get the ModelManager instance from app state.
    
    Usage in endpoint:
        async def endpoint(model: ModelManager = Depends(get_model_manager)):
            ...
    """
    if not hasattr(request.app.state, 'model_manager'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    model_manager = request.app.state.model_manager
    
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready yet."
        )
    
    return model_manager

def get_detector(model_manager: ModelManager = Depends(get_model_manager)) -> Detector:
    """
    Get a Detector instance with the loaded model.
    
    THIS IS THE KEY CONNECTION! ← QUAN TRỌNG
    
    Usage in endpoint:
        async def endpoint(detector: Detector = Depends(get_detector)):
            ...
    """
    # Create Detector with the loaded ModelManager
    return Detector(model_manager)

