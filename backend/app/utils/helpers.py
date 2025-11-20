"""
General Helper Functions
"""
import glob
import os
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json
import cv2
from fastapi import status

from fastapi import HTTPException

from app.settings import settings

def generate_unique_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def generate_filename(prefix: str = "", extension: str = "") -> str:
    """Generate a unique filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = generate_unique_id()[:8]
    
    filename = f"{prefix}_{timestamp}_{unique_id}{extension}" if prefix else f"{timestamp}_{unique_id}{extension}"
    return filename


def get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def ensure_directory(directory: str) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_bytes(bytes_size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def clean_old_files(directory: str, max_age_hours: int = 24) -> int:
    """
    Delete files older than specified hours.
    Returns number of files deleted.
    """
    from datetime import timedelta
    
    path = Path(directory)
    if not path.exists():
        return 0
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    deleted_count = 0
    
    for file_path in path.glob('*'):
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_time:
                file_path.unlink()
                deleted_count += 1
    
    return deleted_count


def _count_violations_by_type(violations: list) -> dict:
    """Count violations grouped by type."""
    counts = {}
    for v in violations:
        v_type = v.get("type", "unknown")
        counts[v_type] = counts.get(v_type, 0) + 1
    return counts

def _find_video_path(video_id: str) -> str:
    # Tìm file trong upload dir
    # Logic tìm file khớp với video_id
    upload_dir = settings.UPLOAD_DIR
    # Nếu upload_dir là tương đối, chuyển sang tuyệt đối để tìm cho chắc
    if not os.path.isabs(upload_dir):
        upload_dir = os.path.join(os.getcwd(), upload_dir)
        
    for fname in os.listdir(upload_dir):
        if fname.startswith(video_id):
            return os.path.join(upload_dir, fname)
    raise FileNotFoundError(f"Video {video_id} not found")

def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    info = {
        "path": video_path,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "duration_seconds": 0
    }
    if info["fps"] > 0:
        info["duration_seconds"] = info["total_frames"] / info["fps"]
        
    cap.release()
    return info


def _sse_event(data: dict) -> str:
    """Format data as Server-Sent Event."""
    return f"data: {json.dumps(data)}\n\n"