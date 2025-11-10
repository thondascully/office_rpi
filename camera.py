# rpi_tracker/camera.py
"""
Camera Module - Handles rpicam-vid capture
"""
import subprocess
import numpy as np
import cv2
import time
from typing import Optional

class Camera:
    """Camera capture using rpicam-vid"""
    
    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.process: Optional[subprocess.Popen] = None
        self.frame_size = width * height * 3 // 2  # YUV420
        
    def start(self):
        """Start camera process"""
        cmd = [
            'rpicam-vid',
            '--codec', 'yuv420',
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', str(self.fps),
            '--timeout', '0',
            '--nopreview',
            '-o', '-'
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )
        
        time.sleep(2)  # Warm up
        print(f"✅ Camera started: {self.width}x{self.height} @ {self.fps}fps")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame from camera"""
        if not self.process:
            return None
        
        raw_frame = self.process.stdout.read(self.frame_size)
        
        if len(raw_frame) != self.frame_size:
            return None
        
        # Convert YUV420 to BGR
        yuv_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        
        return bgr_frame
    
    def stop(self):
        """Stop camera process"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("📷 Camera stopped")
