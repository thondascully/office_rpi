# rpi_tracker/camera.py
"""
Camera Module - Handles rpicam-vid capture
"""
import subprocess
import numpy as np
import cv2
import time
import select
import os
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
        print(f"Camera initialized: {self.width}x{self.height} @ {self.fps}fps")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame from camera (non-blocking)"""
        if not self.process:
            return None
        
        # Check if process is still alive
        if self.process.poll() is not None:
            return None
        
        # Non-blocking check if data is available
        # Use select with a small timeout to avoid blocking
        ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
        if not ready:
            # No data available yet, return None to continue loop
            return None
        
        # Try to read the frame size
        raw_frame = b''
        bytes_read = 0
        timeout_count = 0
        max_timeout = 50  # 50 * 0.01 = 0.5 seconds max wait
        
        while bytes_read < self.frame_size and timeout_count < max_timeout:
            chunk = self.process.stdout.read(self.frame_size - bytes_read)
            if not chunk:
                # No more data available
                time.sleep(0.01)
                timeout_count += 1
                continue
            raw_frame += chunk
            bytes_read += len(chunk)
        
        if len(raw_frame) != self.frame_size:
            return None
        
        # Convert YUV420 to BGR
        try:
            yuv_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
            bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
            return bgr_frame
        except Exception:
            return None
    
    def is_alive(self) -> bool:
        """Check if camera process is still running"""
        if not self.process:
            return False
        return self.process.poll() is None
    
    def stop(self):
        """Stop camera process"""
        if self.process:
            self.process.terminate()
            self.process.wait()
