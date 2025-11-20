"""
High-level detection interface.
Handles image/video processing and violation detection logic.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from app.core.model import ModelManager
from app.core.violation_analyzer import ViolationAnalyzer

logger = logging.getLogger(__name__)


class Detector:
    """High-level detector for images and videos."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.violation_analyzer = ViolationAnalyzer()
    
    async def detect_from_image_path(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None,
        return_annotated: bool = True,
    ) -> Dict[str, Any]:
        """
        Detect objects in an image file.
        
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold
            return_annotated: Whether to return annotated image
            
        Returns:
            Detection results with violations
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            return await self.detect_from_image(
                image,
                conf_threshold=conf_threshold,
                return_annotated=return_annotated
            )
            
        except Exception as e:
            logger.error(f"Error detecting from image path: {e}")
            raise
    
    async def detect_from_image(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        return_annotated: bool = True,
        show_only_violations: bool = True,
    ) -> Dict[str, Any]:
        """
        Detect objects in an image array.
        
        Args:
            image: Image as numpy array (BGR format)
            conf_threshold: Confidence threshold
            return_annotated: Whether to return annotated image
            
        Returns:
            Detection results with violations
        """
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Run model inference
            detections = await self.model_manager.predict_image(
                image,
                conf_threshold=conf_threshold
            )
            
            # Analyze violations
            violations = self.violation_analyzer.analyze_frame(detections)
            
            # Create result
            result = {
                "image_size": {
                    "width": width,
                    "height": height
                },
                "detections": detections,
                "violations": violations,
                "summary": {
                    "total_detections": len(detections),
                    "total_violations": len(violations),
                    "by_class": self._count_by_class(detections),
                    "violations_by_type": self._count_violations_by_type(violations),
                }
            }
            
            # Add annotated image if requested
            if return_annotated:
                annotated = self.model_manager.annotate_image(
                    image, 
                    detections,
                    violations,  
                    show_conf=True,
                    show_only_violations=show_only_violations,  
                    font_scale=0.5
                )
                result["annotated_image"] = annotated
            # ===============================================
            
            return result
            
        except Exception as e:
            logger.error(f"Error in detect_from_image: {e}")
            raise
    
    def _count_by_class(self, detections: List[Dict]) -> Dict[str, int]:
        """Count detections by class."""
        counts = {}
        for det in detections:
            class_name = det["class_name"]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
    
    def _count_violations_by_type(self, violations: List[Dict]) -> Dict[str, int]:
        """Count violations by type."""
        counts = {}
        for viol in violations:
            viol_type = viol["type"]
            counts[viol_type] = counts.get(viol_type, 0) + 1
        return counts
    
    
    
    def _create_timeline(self, frame_results: List[Dict]) -> List[Dict]:
        """Create a timeline of violations."""
        timeline = []
        for result in frame_results:
            if result["violations"]:
                timeline.append({
                    "timestamp": result["timestamp"],
                    "frame_number": result["frame_number"],
                    "violation_count": len(result["violations"]),
                    "violations": result["violations"]
                })
        return timeline