"""
CRUD operations for database - CLEANED & OPTIMIZED
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from app.model.db import DetectionResult, VideoTask


# ============= DETECTION RESULTS =============

async def create_detection_result(
    db: AsyncSession,
    file_name: str,
    file_type: str,
    detections: List[Dict],
    violations: List[Dict],
    processing_time_ms: float,
    **kwargs
) -> DetectionResult:
    """Save a detection result to database."""
    result = DetectionResult(
        file_name=file_name,
        file_type=file_type,
        total_detections=len(detections),
        total_violations=len(violations),
        detections=detections,
        violations=violations,
        processing_time_ms=processing_time_ms,
        **kwargs
    )
    
    db.add(result)
    await db.commit()
    await db.refresh(result)
    
    return result


async def get_detection_result(db: AsyncSession, result_id: int) -> Optional[DetectionResult]:
    """Get a detection result by ID."""
    result = await db.execute(
        select(DetectionResult).where(DetectionResult.id == result_id)
    )
    return result.scalar_one_or_none()


async def get_recent_detections(
    db: AsyncSession,
    limit: int = 10,
    file_type: Optional[str] = None
) -> List[DetectionResult]:
    """Get recent detection results."""
    query = select(DetectionResult).order_by(DetectionResult.created_at.desc())
    
    if file_type:
        query = query.where(DetectionResult.file_type == file_type)
    
    query = query.limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


# ============= STATISTICS =============

async def get_statistics_summary(db: AsyncSession) -> Dict[str, Any]:
    """Get overall statistics summary."""
    # Total processed
    total_processed = await db.scalar(
        select(func.count(DetectionResult.id))
    )
    
    # Total detections
    total_detections = await db.scalar(
        select(func.sum(DetectionResult.total_detections))
    ) or 0
    
    # Total violations
    total_violations = await db.scalar(
        select(func.sum(DetectionResult.total_violations))
    ) or 0
    
    # By file type
    by_type = await db.execute(
        select(
            DetectionResult.file_type,
            func.count(DetectionResult.id)
        ).group_by(DetectionResult.file_type)
    )
    
    return {
        "total_processed": total_processed,
        "total_detections": total_detections,
        "total_violations": total_violations,
        "by_type": {row[0]: row[1] for row in by_type}
    }


async def get_violations_by_type(db: AsyncSession) -> Dict[str, int]:
    """Count violations by type from all results."""
    results = await db.execute(
        select(DetectionResult.violations)
    )
    
    violation_counts = {}
    for (violations,) in results:
        if violations:
            for v in violations:
                v_type = v.get('type', 'unknown')
                violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
    
    return violation_counts


async def get_timeline_data(
    db: AsyncSession,
    hours: int = 24
) -> List[Dict]:
    """Get detection timeline for past N hours."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    results = await db.execute(
        select(
            func.date_trunc('hour', DetectionResult.created_at).label('hour'),
            func.count(DetectionResult.id).label('count'),
            func.sum(DetectionResult.total_violations).label('violations')
        ).where(
            DetectionResult.created_at >= cutoff
        ).group_by('hour').order_by('hour')
    )
    
    return [
        {
            'timestamp': row.hour.isoformat(),
            'count': row.count,
            'violations': row.violations or 0
        }
        for row in results
    ]


# ============= VIDEO TASKS =============

async def create_video_task(
    db: AsyncSession,
    task_id: str,
    video_id: str,
    video_path: str,
    total_frames: int
) -> VideoTask:
    """Create a new video processing task."""
    task = VideoTask(
        id=task_id,
        video_id=video_id,
        video_path=video_path,
        total_frames=total_frames,
        status='pending'
    )
    
    db.add(task)
    await db.commit()
    await db.refresh(task)
    
    return task


async def get_video_task(db: AsyncSession, task_id: str) -> Optional[VideoTask]:
    """Get video task by ID."""
    result = await db.execute(
        select(VideoTask).where(VideoTask.id == task_id)
    )
    return result.scalar_one_or_none()


async def update_video_task(
    db: AsyncSession,
    task_id: str,
    **updates
) -> Optional[VideoTask]:
    """
    Update video task with arbitrary fields.
    
    Unified function to replace:
    - update_video_task_status
    - update_video_task_progress
    - complete_video_task
    - fail_video_task
    
    Usage:
        # Update status
        await update_video_task(db, task_id, status="processing")
        
        # Update progress
        await update_video_task(db, task_id, progress=0.5, current_frame=100)
        
        # Complete task
        await update_video_task(
            db, task_id,
            status="completed",
            progress=1.0,
            result={...},
            completed_at=datetime.utcnow()
        )
        
        # Fail task
        await update_video_task(
            db, task_id,
            status="failed",
            error_message="Error message",
            completed_at=datetime.utcnow()
        )
    """
    result = await db.execute(
        select(VideoTask).where(VideoTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    
    if not task:
        return None
    
    # Apply updates
    for key, value in updates.items():
        if hasattr(task, key):
            setattr(task, key, value)
    
    # Auto-set started_at when status becomes "processing"
    if updates.get('status') == 'processing' and not task.started_at:
        task.started_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(task)
    
    return task


# ============= CONVENIENCE FUNCTIONS =============
# These are wrappers around update_video_task for better API

async def start_video_task(db: AsyncSession, task_id: str) -> Optional[VideoTask]:
    """Mark video task as started (processing)."""
    return await update_video_task(
        db,
        task_id,
        status="processing",
        started_at=datetime.utcnow()
    )


async def update_video_progress(
    db: AsyncSession,
    task_id: str,
    progress: float,
    current_frame: int
) -> Optional[VideoTask]:
    """Update video task progress."""
    return await update_video_task(
        db,
        task_id,
        progress=progress,
        current_frame=current_frame
    )


async def complete_video_task(
    db: AsyncSession,
    task_id: str,
    result: dict
) -> Optional[VideoTask]:
    """Mark video task as completed."""
    return await update_video_task(
        db,
        task_id,
        status="completed",
        progress=1.0,
        result=result,
        completed_at=datetime.utcnow()
    )


async def fail_video_task(
    db: AsyncSession,
    task_id: str,
    error_message: str
) -> Optional[VideoTask]:
    """Mark video task as failed."""
    return await update_video_task(
        db,
        task_id,
        status="failed",
        error_message=error_message,
        completed_at=datetime.utcnow()
    )