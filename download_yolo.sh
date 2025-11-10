#!/usr/bin/env python3
"""
Download and convert YOLOv8n model for Raspberry Pi 5
Uses ultralytics to download .pt and convert to ONNX format
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("  YOLOv8n Model Download & Conversion")
print("=" * 70)
print()

# Create models directory
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
print(f"📁 Model directory: {models_dir.absolute()}")
print()

# Check if ultralytics is installed
try:
    from ultralytics import YOLO
    print("✅ ultralytics package found")
except ImportError:
    print("❌ ultralytics not installed")
    print()
    print("Installing ultralytics...")
    print()
    os.system(f"{sys.executable} -m pip install ultralytics")
    print()
    try:
        from ultralytics import YOLO
        print("✅ ultralytics installed successfully")
    except ImportError:
        print("❌ Failed to install ultralytics")
        print()
        print("Please install manually:")
        print(f"  {sys.executable} -m pip install ultralytics")
        sys.exit(1)

print()
print("-" * 70)

# Check if ONNX model already exists
onnx_path = models_dir / "yolov8n.onnx"
if onnx_path.exists():
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"⚠️  ONNX model already exists: {onnx_path} ({size_mb:.1f} MB)")
    response = input("   Re-download and convert? (y/N): ")
    if response.lower() != 'y':
        print("✅ Using existing model")
        sys.exit(0)
    onnx_path.unlink()

print()
print("Step 1: Downloading YOLOv8n PyTorch model (.pt)")
print("-" * 70)

try:
    # This will automatically download yolov8n.pt if not present
    model = YOLO('yolov8n.pt')
    print("✅ YOLOv8n.pt downloaded successfully")
except Exception as e:
    print(f"❌ Failed to download model: {e}")
    sys.exit(1)

print()
print("Step 2: Converting to ONNX format")
print("-" * 70)
print("This may take 1-2 minutes...")
print()

try:
    # Export to ONNX with optimizations for inference
    model.export(
        format='onnx',
        imgsz=640,
        simplify=True,
        dynamic=False  # Static shape is faster on RPi
    )
    print()
    print("✅ Model converted to ONNX successfully")
except Exception as e:
    print(f"❌ Failed to convert model: {e}")
    sys.exit(1)

# Move the generated ONNX file to models directory
generated_onnx = Path("yolov8n.onnx")
if generated_onnx.exists():
    generated_onnx.rename(onnx_path)
    print(f"✅ Model moved to: {onnx_path}")

# Verify final file
if onnx_path.exists():
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print()
    print("=" * 70)
    print(f"✅ YOLOv8n ONNX model ready! ({size_mb:.1f} MB)")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Edit edge_tracker.py and set BACKEND_API_URL to your server IP")
    print("  2. Run: python3 edge_tracker.py")
    print()
else:
    print()
    print("=" * 70)
    print("❌ Model file not found after conversion")
    print("=" * 70)
    print()
    sys.exit(1)

# Clean up the .pt file if user wants
print()
response = input("Delete the .pt file to save space? (y/N): ")
if response.lower() == 'y':
    pt_file = Path("yolov8n.pt")
    if pt_file.exists():
        pt_file.unlink()
        print("✅ yolov8n.pt deleted")
    else:
        print("⚠️  yolov8n.pt not found")
