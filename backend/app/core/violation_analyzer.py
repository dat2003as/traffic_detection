"""
Violation Analysis Logic
Analyzes detections to identify traffic violations (no helmet, etc.)
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ViolationAnalyzer:
    """Analyzes detections to identify traffic violations."""
    
    def __init__(self):
        # Define class mappings (adjust based on your model's classes)
        self.class_mappings = {
            "helmet": "helmet",
            "no-helmet": "no_helmet",
            "person": "person",
            "motorcycle": "motorcycle",
            "motorbike": "motorcycle",
            "bike": "motorcycle",
        }
        
        # IoU threshold for checking if person is on motorcycle
        self.iou_threshold = 0.3
    
    def analyze_frame(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze detections in a frame to identify violations.
        
        Args:
            detections: List of detections from model
            
        Returns:
            List of violations found
        """
        violations = []
        
        # Group detections by class
        grouped = self._group_by_class(detections)

        logger.info(f"🔍 Analyzing violations:")
        logger.info(f"  - Total detections: {len(detections)}")
        logger.info(f"  - Grouped: {list(grouped.keys())}")
        for class_name, items in grouped.items():
            logger.info(f"    {class_name}: {len(items)}")
        # Analyze helmet violations
        helmet_violations = self._analyze_helmet_violations(grouped)
        violations.extend(helmet_violations)
        
        # Can add more violation types here:
        # - Illegal parking
        # - Wrong-way driving
        # - etc.
        
        return violations
    
    def _group_by_class(self, detections: List[Dict]) -> Dict[str, List[Dict]]:
        """Group detections by class name."""
        grouped = {}
        
        for det in detections:
            class_name = det["class_name"].lower()
            
            # ============= FIX CLASS MAPPING =============
            # Map your model's classes
            if class_name == "dhelmet":
                mapped_class = "helmet"
            elif class_name == "dnohelmet":
                mapped_class = "no_helmet"  # ← KEY!
            elif class_name in ["motorbike", "motorcycle"]:
                mapped_class = "motorcycle"
            elif class_name == "person":
                mapped_class = "person"
            else:
                mapped_class = class_name
            # ============================================
            
            logger.debug(f"Mapping: {class_name} → {mapped_class}")
            
            if mapped_class not in grouped:
                grouped[mapped_class] = []
            
            grouped[mapped_class].append(det)
        
        return grouped
    
    def _analyze_helmet_violations(self, grouped: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Detect people on motorcycles without helmets.
        
        Logic:
        1. Find all motorcycles
        2. For each motorcycle, find persons on it (using IoU)
        3. Check if those persons have helmets
        4. If no helmet detected in head region → violation
        """
        violations = []
        
        motorcycles = grouped.get("motorcycle", [])
        persons = grouped.get("person", [])
        helmets = grouped.get("helmet", [])
        no_helmets = grouped.get("no_helmet", [])
        
        # If model directly detects "no-helmet" class
        if no_helmets:
             for no_helmet in no_helmets:
                violations.append({
                    "type": "no_helmet",
                    "severity": "high",
                    "description": "Person not wearing helmet",
                    "bbox": no_helmet["bbox"],
                    "confidence": no_helmet["confidence"],
                    "details": {
                        "detection_id": no_helmet["id"],
                        "class_name": no_helmet["class_name"]
                    }
                })
        
        # Alternative: Analyze person-motorcycle-helmet relationships
        else:
            for motorcycle in motorcycles:
                # Find persons on this motorcycle
                persons_on_bike = self._find_overlapping_objects(
                    motorcycle["bbox"],
                    persons,
                    self.iou_threshold
                )
                
                for person in persons_on_bike:
                    # Check if person has helmet
                    has_helmet = self._check_helmet(person, helmets)
                    
                    if not has_helmet:
                        violations.append({
                            "type": "no_helmet",
                            "severity": "high",
                            "description": "Person on motorcycle without helmet",
                            "bbox": person["bbox"],
                            "confidence": person["confidence"],
                            "details": {
                                "person_id": person["id"],
                                "motorcycle_id": motorcycle["id"],
                                "motorcycle_bbox": motorcycle["bbox"],
                            }
                        })
        
        return violations
    
    def _find_overlapping_objects(
        self,
        ref_bbox: Dict[str, float],
        objects: List[Dict],
        iou_threshold: float
    ) -> List[Dict]:
        """Find objects that overlap with reference bbox."""
        overlapping = []
        
        for obj in objects:
            iou = self._calculate_iou(ref_bbox, obj["bbox"])
            if iou > iou_threshold:
                overlapping.append(obj)
        
        return overlapping
    
    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        # Get coordinates
        x1_min, y1_min = bbox1["x1"], bbox1["y1"]
        x1_max, y1_max = bbox1["x2"], bbox1["y2"]
        x2_min, y2_min = bbox2["x1"], bbox2["y1"]
        x2_max, y2_max = bbox2["x2"], bbox2["y2"]
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height
        
        # Calculate union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou
    
    def _check_helmet(self, person: Dict, helmets: List[Dict]) -> bool:
        """Check if a person has a helmet."""
        # Define head region (top 30% of person bbox)
        person_bbox = person["bbox"]
        head_height = (person_bbox["y2"] - person_bbox["y1"]) * 0.3
        
        head_region = {
            "x1": person_bbox["x1"],
            "y1": person_bbox["y1"],
            "x2": person_bbox["x2"],
            "y2": person_bbox["y1"] + head_height,
        }
        
        # Check if any helmet overlaps with head region
        for helmet in helmets:
            iou = self._calculate_iou(head_region, helmet["bbox"])
            if iou > 0.2:  # Lower threshold for head region
                return True
        
        return False
    
    def analyze_parking_violation(
        self,
        detections: List[Dict],
        no_parking_zones: List[List[tuple]]
    ) -> List[Dict]:
        """
        Detect vehicles parked in no-parking zones.
        
        Args:
            detections: List of detections
            no_parking_zones: List of polygons defining no-parking zones
            
        Returns:
            List of parking violations
        """
        violations = []
        
        # Get vehicles
        vehicles = [d for d in detections if d["class_name"] in ["car", "motorcycle", "truck"]]
        
        for vehicle in vehicles:
            center = vehicle["center"]
            
            # Check if vehicle center is in any no-parking zone
            for zone_idx, zone in enumerate(no_parking_zones):
                if self._point_in_polygon((center["x"], center["y"]), zone):
                    violations.append({
                        "type": "illegal_parking",
                        "severity": "medium",
                        "description": f"Vehicle parked in no-parking zone {zone_idx + 1}",
                        "bbox": vehicle["bbox"],
                        "confidence": vehicle["confidence"],
                        "details": {
                            "vehicle_id": vehicle["id"],
                            "vehicle_type": vehicle["class_name"],
                            "zone_id": zone_idx,
                        }
                    })
                    break
        
        return violations
    
    def _point_in_polygon(self, point: tuple, polygon: List[tuple]) -> bool:
        """Check if a point is inside a polygon using ray casting."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside