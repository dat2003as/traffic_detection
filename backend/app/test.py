import cv2
from app.dependencies import get_detector
import asyncio

async def main():
    detector = get_detector()
    image = cv2.imread("D:\traffic_violation_detection\image.png")
    result = await detector.detect_from_image(
        image,
        conf_threshold=0.1,
        return_annotated=True
    )
    print(f"Detections: {len(result['detections'])}")

asyncio.run(main())
