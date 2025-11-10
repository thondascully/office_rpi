# rpi_tracker/detector.py
"""
YOLO Detector Module
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple

class PersonDetector:
    """YOLOv8n person detector"""
    
    def __init__(self, model_path: Path, detection_scale: float, confidence: float):
        self.detection_scale = detection_scale
        self.confidence = confidence
        
        print(f"🧠 Loading YOLOv8n from {model_path}...")
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        print("✅ YOLOv8n loaded")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO"""
        resized = cv2.resize(image, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        return batched
    
    def postprocess(self, output: np.ndarray, orig_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, float]]:
        """Postprocess YOLO output"""
        detections = []
        output = output[0]
        output = np.transpose(output, (1, 0))
        
        orig_h, orig_w = orig_shape
        scale_x = orig_w / 640
        scale_y = orig_h / 640
        
        PERSON_CLASS_ID = 0
        
        for detection in output:
            cx, cy, w, h = detection[:4]
            class_scores = detection[4:]
            
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if class_id == PERSON_CLASS_ID and confidence >= self.confidence:
                x = int((cx - w / 2) * scale_x)
                y = int((cy - h / 2) * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                
                x = max(0, min(x, orig_w - 1))
                y = max(0, min(y, orig_h - 1))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))
                
                detections.append((x, y, w, h, float(confidence)))
        
        return detections
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect people in frame"""
        # Downscale for detection
        detection_w = int(frame.shape[1] * self.detection_scale)
        detection_h = int(frame.shape[0] * self.detection_scale)
        detection_frame = cv2.resize(frame, (detection_w, detection_h))
        
        # Run YOLO
        input_tensor = self.preprocess(detection_frame)
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        raw_detections = self.postprocess(output, detection_frame.shape[:2])
        
        # Scale back to original resolution
        detections = []
        for x, y, w, h, conf in raw_detections:
            x_full = int(x / self.detection_scale)
            y_full = int(y / self.detection_scale)
            w_full = int(w / self.detection_scale)
            h_full = int(h / self.detection_scale)
            detections.append((x_full, y_full, w_full, h_full, conf))
        
        return detections
