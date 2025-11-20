"""
Image Processing Utilities
"""
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Optional
from pathlib import Path


def read_image_file(file_path: str) -> np.ndarray:
    """Read image from file path."""
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")
    return image


def read_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Read image from bytes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


def image_to_bytes(image: np.ndarray, format: str = '.jpg') -> bytes:
    """Convert numpy array image to bytes."""
    is_success, buffer = cv2.imencode(format, image)
    if not is_success:
        raise ValueError("Could not encode image")
    return buffer.tobytes()


def image_to_base64(image: np.ndarray, format: str = '.jpg') -> str:
    """Convert numpy array image to base64 string."""
    image_bytes = image_to_bytes(image, format)
    return base64.b64encode(image_bytes).decode('utf-8')


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy array image."""
    image_bytes = base64.b64decode(base64_string)
    return read_image_bytes(image_bytes)


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize image to specified dimensions.
    If keep_aspect_ratio is True, only one dimension needs to be specified.
    """
    h, w = image.shape[:2]
    
    if keep_aspect_ratio:
        if width and not height:
            ratio = width / w
            height = int(h * ratio)
        elif height and not width:
            ratio = height / h
            width = int(w * ratio)
    
    if width is None or height is None:
        raise ValueError("Must specify at least one dimension")
    
    return cv2.resize(image, (width, height))


def crop_image(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop image to bounding box."""
    return image[y1:y2, x1:x2]