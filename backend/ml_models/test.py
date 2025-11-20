# Test script: test_model.py

from ultralytics import YOLO
import torch

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
print("\n📦 Loading model...")
model = YOLO('ml_models/yolov8_best.pt')

# Get model info
print(f"✅ Model loaded successfully!")
print(f"Classes: {model.names}")
print(f"Device: {model.device}")

# Test inference on dummy image
import numpy as np
dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

print("\n🧪 Testing inference...")
results = model(dummy_img, verbose=False)
print(f"✅ Inference works!")
print(f"Detected: {len(results[0].boxes)} objects")