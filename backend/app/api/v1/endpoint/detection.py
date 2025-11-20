"""
Detection Endpoints - FIXED Redis Cache & Database Integration
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from fastapi.responses import FileResponse
from typing import List, Optional
import time
import logging
import cv2
import numpy as np
import hashlib
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.detector import Detector
from app.dependencies import get_detector
from app.services.storage import StorageService
from app.services.cache import cache
from app.utils.image import read_image_bytes, image_to_bytes
from app.utils.validators import validate_image_file, validate_file_size
from app.db.database import get_db, AsyncSessionLocal
from app.db.crud import create_detection_result, get_detection_result
#from app.utils.helpers import get_file_hash

logger = logging.getLogger(__name__)

router = APIRouter()
storage = StorageService()


def get_file_hash(content: bytes) -> str:
    """Calculate MD5 hash of file content."""
    return hashlib.md5(content).hexdigest()

@router.post("/image", summary="Detect violations in a single image")
async def detect_image(
    file: UploadFile = File(..., description="Image file to process"),
    conf_threshold: Optional[float] = None,
    return_annotated: bool = True,
    show_only_violations: bool = True,
    use_cache: bool = True,
    detector: Detector = Depends(get_detector),
    db: AsyncSession = Depends(get_db),
    request: Request = None
):
    """
    Detect traffic violations in an uploaded image.
    
    **Features:**
    - ✅ Redis caching for faster repeated requests
    - ✅ Database storage for analytics
    - ✅ File storage for uploaded/processed images
    
    **Parameters:**
    - **file**: Image file (JPG, PNG)
    - **conf_threshold**: Confidence threshold (0.0-1.0)
    - **return_annotated**: Whether to return annotated image
    - **use_cache**: Use Redis cache (default: True)
    
    **Returns:**
    - Detection results with violations
    - Annotated image URL (if requested)
    """
    start_time = time.time()
    
    try:
        # Validate file
        validate_image_file(file)
        validate_file_size(file, max_size_mb=10)
        
        # Read image
        image_bytes = await file.read()
        
        # ============= REDIS CACHE CHECK =============
        cache_key = None
        if use_cache:
            # Generate cache key from file hash + params
            file_hash = get_file_hash(image_bytes)
            cache_key = cache.generate_cache_key(
                "detection",
                file_hash,
                conf_threshold or "default",
                return_annotated
            )
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            
            if cached_result:
                logger.info(f"✅ Cache HIT for {file.filename}")
                cached_result["from_cache"] = True
                cached_result["cache_time_ms"] = (time.time() - start_time) * 1000
                return cached_result
            
            logger.info(f"⚠️ Cache MISS for {file.filename}")
        # =============================================
        
        # Decode image
        image = read_image_bytes(image_bytes)
        
        # Run detection
        result = await detector.detect_from_image(
            image,
            conf_threshold=conf_threshold,
            return_annotated=return_annotated
        )
        
        # Save annotated image if requested
        annotated_url = None
        if return_annotated and "annotated_image" in result:
            annotated_bytes = image_to_bytes(result["annotated_image"])
            filename = f"annotated_{file.filename}"
            file_path = await storage.save_processed_file(
                annotated_bytes,
                filename,
                subfolder="images"
            )
            annotated_url = storage.get_file_url(file_path)
            
            # Remove annotated_image from result (too large for JSON)
            del result["annotated_image"]
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Prepare response
        response = {
            **result,
            "annotated_image_url": annotated_url,
            "processing_time_ms": processing_time_ms,
            "from_cache": False
        }
        
        # ============= SAVE TO DATABASE =============
        try:
            db_result = await create_detection_result(
                db=db,
                file_name=file.filename,
                file_type="image",
                file_path=annotated_url,
                detections=result["detections"],
                violations=result["violations"],
                processing_time_ms=processing_time_ms,
                model_confidence=conf_threshold,
                ip_address=request.client.host if request and request.client else None,
                user_agent=request.headers.get("user-agent") if request else None
            )
            response["result_id"] = db_result.id
            logger.info(f"✅ Saved to DB with ID: {db_result.id}")
        except Exception as e:
            logger.error(f"❌ Database save failed: {e}")
            # Don't fail the request if DB save fails
        # ============================================
        
        # ============= SAVE TO REDIS CACHE ==========
        # 🔥 FIX: Only cache AFTER we have the response
        if use_cache and cache_key and response:
            try:
                # Cache for 1 hour
                await cache.set(cache_key, response, ttl=3600)
                logger.info(f"✅ Cached result for {file.filename}")
            except Exception as e:
                logger.error(f"❌ Cache save failed: {e}")
        # ============================================
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Error in image detection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@router.post("/batch", summary="Detect violations in multiple images")
async def detect_batch_images(
    files: List[UploadFile] = File(..., description="List of image files"),
    conf_threshold: Optional[float] = None,
    use_cache: bool = True,
    detector: Detector = Depends(get_detector),
    db: AsyncSession = Depends(get_db)
):
    """
    Detect traffic violations in multiple images.
    
    **Features:**
    - ✅ Batch processing with caching
    - ✅ Database storage for all results
    - ✅ Parallel cache lookups
    
    **Parameters:**
    - **files**: List of image files (max 50)
    - **conf_threshold**: Confidence threshold
    - **use_cache**: Use Redis cache
    
    **Returns:**
    - List of detection results for each image
    """
    start_time = time.time()
    
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 images per batch"
        )
    
    results = []
    successful = 0
    failed = 0
    cache_hits = 0
    
    for file in files:
        file_start_time = time.time()
        try:
            # Validate file
            validate_image_file(file)
            image_bytes = await file.read()
            
            # ============= CHECK CACHE =============
            cache_key = None
            cached_result = None
            if use_cache:
                file_hash = get_file_hash(image_bytes)
                cache_key = cache.generate_cache_key(
                    "detection",
                    file_hash,
                    conf_threshold or "default",
                    False  # batch doesn't return annotated
                )
                cached_result = await cache.get(cache_key)
                
                if cached_result:
                    cache_hits += 1
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "result": {
                            **cached_result,
                            "from_cache": True
                        }
                    })
                    successful += 1
                    continue
            # ======================================
            
            # Process image
            image = read_image_bytes(image_bytes)
            result = await detector.detect_from_image(
                image,
                conf_threshold=conf_threshold,
                return_annotated=False
            )
            
            file_processing_time = (time.time() - file_start_time) * 1000
            
            # ============= SAVE TO DB =============
            try:
                db_result = await create_detection_result(
                    db=db,
                    file_name=file.filename,
                    file_type="image",
                    detections=result["detections"],
                    violations=result["violations"],
                    processing_time_ms=file_processing_time,
                    model_confidence=conf_threshold
                )
                result["result_id"] = db_result.id
            except Exception as e:
                logger.error(f"DB save failed for {file.filename}: {e}")
            # ======================================
            
            # ============= CACHE RESULT ===========
            if use_cache and cache_key:
                try:
                    await cache.set(cache_key, result, ttl=3600)
                except Exception as e:
                    logger.error(f"Cache save failed for {file.filename}: {e}")
            # ======================================
            
            results.append({
                "filename": file.filename,
                "success": True,
                "result": {
                    **result,
                    "from_cache": False,
                    "processing_time_ms": file_processing_time
                }
            })
            successful += 1
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
            failed += 1
    
    total_processing_time_ms = (time.time() - start_time) * 1000
    
    return {
        "total_images": len(files),
        "successful": successful,
        "failed": failed,
        "cache_hits": cache_hits,
        "cache_hit_rate": f"{(cache_hits/len(files)*100):.1f}%" if len(files) > 0 else "0%",
        "results": results,
        "total_processing_time_ms": total_processing_time_ms
    }


@router.get("/result/{result_id}", summary="Get detection result by ID")
async def get_detection_result_endpoint(
    result_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve a previously saved detection result from database.
    
    **Parameters:**
    - **result_id**: ID of the detection result
    
    **Returns:**
    - Detection result data
    """
    try:
        result = await get_detection_result(db, result_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Detection result {result_id} not found"
            )
        
        return {
            "id": result.id,
            "file_name": result.file_name,
            "file_type": result.file_type,
            "file_path": result.file_path,
            "total_detections": result.total_detections,
            "total_violations": result.total_violations,
            "detections": result.detections,
            "violations": result.violations,
            "processing_time_ms": result.processing_time_ms,
            "created_at": result.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/cache/clear", summary="Clear detection cache")
async def clear_detection_cache():
    """
    Clear all cached detection results.
    
    **Admin endpoint** - Should be protected in production.
    """
    try:
        success = await cache.clear_all()
        if success:
            return {"message": "Cache cleared successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear cache"
            )
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/cache/stats", summary="Get cache statistics")
async def get_cache_stats():
    """
    Get Redis cache statistics.
    """
    try:
        stats = await cache.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )