"""
Video Processing Endpoints - USING CRUD FUNCTIONS
"""
import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, FileResponse
from typing import Optional, AsyncGenerator
import asyncio
import json
import uuid
import logging
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.detector import Detector
from app.dependencies import get_detector
from app.services.storage import StorageService
from app.utils.validators import validate_video_file, validate_file_size
from app.utils.video import get_video_info
from app.db.database import get_db
# ============= IMPORT CRUD FUNCTIONS =============
from app.db.crud import (
    create_video_task,
    get_video_task,
    update_video_task,
    start_video_task,
    update_video_progress,
    complete_video_task,
    fail_video_task
)
import cv2
from app.db.database import AsyncSessionLocal
from app.utils.helpers import _count_violations_by_type, _find_video_path, _get_output_path, _sse_event

# ================================================

logger = logging.getLogger(__name__)

router = APIRouter()
storage = StorageService()


@router.post("/upload", summary="Upload video for processing")
async def upload_video(
    file: UploadFile = File(..., description="Video file to upload"),
    db: AsyncSession = Depends(get_db)
):
    """Upload a video file for processing."""
    try:
        validate_video_file(file)
        validate_file_size(file, max_size_mb=200)
        
        video_id = str(uuid.uuid4())
        
        file_path = await storage.save_upload_file(
            file,
            subfolder="videos",
            custom_filename=f"{video_id}_{file.filename}"
        )
        
        try:
            video_metadata = get_video_info(file_path)
            logger.info(f"✅ Video info: {video_metadata['width']}x{video_metadata['height']} @ {video_metadata['fps']}fps")
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            video_metadata = {"error": str(e)}
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "file_path": file_path,
            "status": "uploaded",
            "video_info": video_metadata
        }
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/process-stream", summary="Process video with real-time streaming")
async def process_video_stream(
    video_id: str,
    conf_threshold: Optional[float] = None,
    process_every_n_frames: int = 3,
    save_annotated: bool = True,
    detector: Detector = Depends(get_detector),
    db: AsyncSession = Depends(get_db)
):
    """Process video and stream results in real-time using SSE."""
    
    video_path = _find_video_path(video_id)
    
    try:
        video_info = get_video_info(video_path)
        total_frames = video_info['total_frames']
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read video: {e}"
        )
    
    # ============= USE CRUD =============
    task_id = str(uuid.uuid4())
    await create_video_task(
        db=db,
        task_id=task_id,
        video_id=video_id,
        video_path=video_path,
        total_frames=total_frames
    )
    # ===================================
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events stream."""
        try:
            yield _sse_event({
                'event': 'started',
                'task_id': task_id,
                'video_info': video_info
            })
            
            output_path = _get_output_path(task_id) if save_annotated else None
            
            async for update in _process_video_streaming(
                video_path=video_path,
                output_path=output_path,
                detector=detector,
                conf_threshold=conf_threshold,
                process_every_n_frames=process_every_n_frames,
                task_id=task_id,
                video_info=video_info
            ):
                yield _sse_event(update)
                await asyncio.sleep(0)
            
            yield _sse_event({'event': 'complete', 'task_id': task_id})
            
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield _sse_event({'event': 'error', 'error': str(e)})
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/status/{task_id}", summary="Get video processing status")
async def get_processing_status(
    task_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get the current status of a video processing task."""
    try:
        # ============= USE CRUD =============
        task = await get_video_task(db, task_id)
        # ===================================
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        
        return {
            "task_id": task.id,
            "video_id": task.video_id,
            "status": task.status,
            "progress": task.progress,
            "current_frame": task.current_frame,
            "total_frames": task.total_frames,
            "result": task.result,
            "error_message": task.error_message,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{task_id}/download", summary="Download processed video")
async def download_processed_video(task_id: str):
    """Download the processed video with annotations."""
    video_path = _get_output_path(task_id)
    
    if not Path(video_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processed video not found. Processing may still be in progress."
        )
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"detected_{task_id}.mp4"
    )

async def _process_video_streaming(
    video_path: str,
    output_path: Optional[str],
    detector: Detector,
    conf_threshold: Optional[float],
    process_every_n_frames: int,
    task_id: str,
    video_info: dict
) -> AsyncGenerator[dict, None]:
    """Process video and yield real-time updates."""
    # ============================================================
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = video_info['fps']
        width = video_info['width']
        height = video_info['height']
        total_frames = video_info['total_frames']
        
        out = None
        if output_path:
            # 1. Chuyển đổi sang đường dẫn tuyệt đối để OpenCV ghi file chính xác
            abs_output_path = os.path.abspath(output_path)
            
            # Đảm bảo thư mục cha tồn tại
            os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
            
            logger.info(f"📹 Attempting to write video to: {abs_output_path}")

            # 2. Thử khởi tạo với Codec H.264 (avc1) - Chuẩn Web
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(abs_output_path, fourcc, fps, (width, height))

            # 3. Kiểm tra xem có mở được không
            if not out.isOpened():
                logger.warning("⚠️ Codec 'avc1' failed! Trying fallback to 'mp4v'...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(abs_output_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    logger.error("❌ CRITICAL: Could not open VideoWriter with any codec!")
        
        frame_count = 0
        processed_count = 0
        all_violations = []
        last_annotated_frame = None
        
        # ============= USE CRUD: START TASK =============
        async with AsyncSessionLocal() as db:
            await start_video_task(db, task_id)
        # ================================================
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % process_every_n_frames == 0:
                result = await detector.detect_from_image(
                    frame,
                    conf_threshold=conf_threshold,
                    return_annotated=output_path is not None
                )
                
                result["frame_number"] = frame_count
                result["timestamp"] = round(frame_count / fps, 2)
                
                if out is not None and "annotated_image" in result:
                    last_annotated_frame = result["annotated_image"]
                    out.write(last_annotated_frame)
                    del result["annotated_image"]
                
                all_violations.extend(result["violations"])
                processed_count += 1
                
                yield {
                    "event": "detection",
                    "frame": frame_count,
                    "timestamp": result["timestamp"],
                    "detections": result["detections"],
                    "violations": result["violations"],
                    "summary": result["summary"]
                }
                
            else:
                if out is not None and last_annotated_frame is not None:
                    out.write(last_annotated_frame)
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                progress = frame_count / total_frames
                
                yield {
                    "event": "progress",
                    "progress": round(progress, 3),
                    "current_frame": frame_count,
                    "total_frames": total_frames,
                    "processed_frames": processed_count
                }
                
                # ============= USE CRUD: UPDATE PROGRESS =============
                async with AsyncSessionLocal() as db:
                    await update_video_progress(
                        db,
                        task_id,
                        progress=progress,
                        current_frame=frame_count
                    )
                # =====================================================
        
        cap.release()
        if out is not None:
            out.release()
            logger.info(f"✅ Video output saved: {output_path}")
        
        summary = {
            "total_detections": sum(len(r.get("detections", [])) for r in []),
            "total_violations": len(all_violations),
            "violations_by_type": _count_violations_by_type(all_violations),
            "video_info": video_info,
            "processed_frames": processed_count
        }
        
        yield {
            "event": "summary",
            "summary": summary,
            "output_path": output_path,
            "download_url": f"/api/v1/video/{task_id}/download" if output_path else None
        }
        
        # ============= USE CRUD: COMPLETE TASK =============
        async with AsyncSessionLocal() as db:
            await complete_video_task(db, task_id, summary)
        # ===================================================
        
        logger.info(f"✅ Video processing complete: {task_id}")
        logger.info(f"   - Processed: {processed_count}/{total_frames} frames")
        logger.info(f"   - Violations: {len(all_violations)}")
        
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        
        yield {
            "event": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }
        
        # ============= USE CRUD: FAIL TASK =============
        async with AsyncSessionLocal() as db:
            await fail_video_task(db, task_id, str(e))
        # ===============================================
        
        raise
    
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()

