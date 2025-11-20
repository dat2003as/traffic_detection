"""
Video Processing Utilities
"""
import cv2
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        "path": video_path,
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    
    info["duration_seconds"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    
    cap.release()
    return info


def extract_frame(video_path: str, frame_number: int) -> Any:
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not extract frame {frame_number}")
    
    return frame


def extract_frames(
    video_path: str,
    interval: int = 1,
    max_frames: Optional[int] = None
) -> list:
    """
    Extract frames from video at specified interval.
    
    Args:
        video_path: Path to video file
        interval: Extract every N frames
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frame arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            frames.append(frame)
            
            if max_frames and len(frames) >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return frames


def create_video_from_frames(
    frames: list,
    output_path: str,
    fps: int = 30
) -> None:
    """Create video from list of frames."""
    if not frames:
        raise ValueError("No frames provided")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()