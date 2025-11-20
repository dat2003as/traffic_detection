"""
YOLO Model Manager
Handles model loading, inference, and predictions.
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2

from app.settings import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages YOLO model loading and inference."""
    
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.device = self._get_device()
        self.model_loaded = False
        
    def _get_device(self) -> str:
        """Determine the device to use for inference."""
        if settings.MODEL_DEVICE == "cuda" and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device
    
    async def load_model(self) -> None:
        """Load the YOLO model from disk."""
        try:
            model_path = Path(settings.MODEL_PATH)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading model from: {model_path}")
            self.model = YOLO(str(model_path))
            self.model.to(self.device)
            
            self.model_loaded = True
            logger.info("✅ Model loaded successfully")
            
            # Get model info
            model_info = self.get_model_info()
            logger.info(f"Model info: {model_info}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_loaded or self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_type": self.model.model.__class__.__name__,
            "device": self.device,
            "classes": self.model.names,
            "num_classes": len(self.model.names),
            "img_size": settings.MODEL_IMG_SIZE,
        }
    
    async def predict_image(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image as numpy array (BGR format)
            conf_threshold: Confidence threshold (default from settings)
            iou_threshold: IOU threshold for NMS (default from settings)
            
        Returns:
            List of detections with bounding boxes and class info
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        conf_threshold = conf_threshold or settings.MODEL_CONFIDENCE
        iou_threshold = iou_threshold or settings.MODEL_IOU_THRESHOLD
        
        try:
            logger.info("="*60)
            logger.info("🤖 MODEL PREDICTION")
            logger.info(f"📐 Input shape: {image.shape}")
            logger.info(f"📊 Input dtype: {image.dtype}")
            logger.info(f"🎯 Conf threshold: {conf_threshold}")
            logger.info(f"🎯 IOU threshold: {iou_threshold}")
            logger.info(f"💻 Device: {self.device}")
            # Run inference
            results = self.model.predict(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False,
            )
            
            # Parse results
            detections = self._parse_results(results[0])
            
            return detections
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _parse_results(self, result) -> List[Dict[str, Any]]:
        """Parse YOLO results into structured format."""
        detections = []
        
        boxes = result.boxes
        
        for i, box in enumerate(boxes):
            detection = {
                "id": i,
                "class_id": int(box.cls[0]),
                "class_name": self.model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": {
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3]),
                },
                "bbox_normalized": {
                    "x1": float(box.xyxyn[0][0]),
                    "y1": float(box.xyxyn[0][1]),
                    "x2": float(box.xyxyn[0][2]),
                    "y2": float(box.xyxyn[0][3]),
                }
            }
            
            # Add center point
            detection["center"] = {
                "x": (detection["bbox"]["x1"] + detection["bbox"]["x2"]) / 2,
                "y": (detection["bbox"]["y1"] + detection["bbox"]["y2"]) / 2,
            }
            
            detections.append(detection)
        
        return detections
    
    
    
    def annotate_image(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        violations: List[Dict[str, Any]] = None,
        show_conf: bool = True,
        show_only_violations: bool = True,  
        font_scale: float = 0.9
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        annotated = image.copy()
        
        
        # If no violations list, create empty one
        if violations is None:
            violations = []
        if show_only_violations:
        # Chỉ vẽ những detections là violations
            for viol in violations:
                self._draw_violation(annotated, viol, show_conf, font_scale)
            
            logger.info(f"✅ Drew {len(violations)} violations only")
            return annotated
        # Get violation detection IDs
        violation_det_ids = set()
        for viol in violations:
            # Match violation bbox to detection
            for det in detections:
                if self._bbox_match(det["bbox"], viol["bbox"]):
                    violation_det_ids.add(det["id"])
                    break
        
        # Draw detections
        for det in detections:
            is_violation = det["id"] in violation_det_ids
            
            # Color mapping
            if is_violation:
                color = (255, 0, 0)  # Blue for violations
                label_prefix = "⚠️ "
            elif det["class_name"].lower() in ["dhelmet", "helmet"]:
                color = (0, 255, 0)  # Green for helmets
                label_prefix = ""
            elif det["class_name"].lower() in ["dnohelmet", "no-helmet"]:
                color = (0, 0, 255)  # Red for no-helmet
                label_prefix = "⚠️ "
            else:
                color = (255, 255, 0)  # Cyan for others
                label_prefix = ""
            
            bbox = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            
            # Draw box
            cv2.rectangle(
                annotated,
                (int(bbox["x1"]), int(bbox["y1"])),
                (int(bbox["x2"]), int(bbox["y2"])),
                color,
                2
            )
            
            # Label
            label = f"{label_prefix}{class_name}"
            if show_conf:
                label += f" {confidence:.2f}"
            
            # Draw label
            (w, h), baseline = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                2
            )
            
            y = max(int(bbox["y1"]) - 10, h + 10)
            
            # Label background
            cv2.rectangle(
                annotated,
                (int(bbox["x1"]), y - h - baseline),
                (int(bbox["x1"]) + w, y + baseline),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                annotated,
                label,
                (int(bbox["x1"]), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        logger.info(f"✅ Annotation complete")
        
        return annotated


    def _draw_violation(
        self, 
        image: np.ndarray, 
        violation: Dict, 
        show_conf: bool,
        font_scale: float
    ):
        """Draw a single violation on image."""
        bbox = violation["bbox"]
        confidence = violation.get("confidence", 0.0)
        viol_type = violation.get("type", "violation")
        
        # ============= COLOR: BLUE FOR VIOLATIONS =============
        color = (255, 0, 0)  # Blue in BGR
        # ======================================================
        
        # Draw bounding box (thick)
        cv2.rectangle(
            image,
            (int(bbox["x1"]), int(bbox["y1"])),
            (int(bbox["x2"]), int(bbox["y2"])),
            color,
            3  # Thick line
        )
        
        # ============= LABEL: "NO HELMET" =============
        if viol_type == "no_helmet":
            label = "NO HELMET"
        else:
            label = viol_type.upper()
        
        if show_conf:
            label += f" {confidence:.2f}"
        # ==============================================
        
        # Calculate label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            2
        )
        
        # Label position (above bbox)
        label_y = max(int(bbox["y1"]) - 10, label_h + 10)
        
        # Draw label background
        cv2.rectangle(
            image,
            (int(bbox["x1"]), label_y - label_h - baseline - 2),
            (int(bbox["x1"]) + label_w + 4, label_y + baseline),
            color,
            -1  # Filled
        )
        
        # Draw label text (white)
        cv2.putText(
            image,
            label,
            (int(bbox["x1"]) + 2, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White
            2
        )
    def _bbox_match(self, bbox1: Dict, bbox2: Dict, threshold: float = 5.0) -> bool:
        """Check if two bboxes are approximately the same."""
        return (
            abs(bbox1["x1"] - bbox2["x1"]) < threshold and
            abs(bbox1["y1"] - bbox2["y1"]) < threshold and
            abs(bbox1["x2"] - bbox2["x2"]) < threshold and
            abs(bbox1["y2"] - bbox2["y2"]) < threshold
        )
    async def warmup(self, num_iterations: int = 3):
        """
        Warmup the model with dummy predictions.
        Useful for GPU initialization.
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        logger.info("Warming up model...")
        
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        for i in range(num_iterations):
            await self.predict_image(dummy_image)
        
        logger.info("✅ Model warmup complete")