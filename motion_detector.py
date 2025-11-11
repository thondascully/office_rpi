# rpi_tracker/motion_detector.py
"""
Lightweight Motion Detection - Prevents overheating by reducing YOLO calls

COMPUTATIONAL COST ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Operation                    CPU Cost        Time (RPi5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOLO Detection (0.25 scale)  ~80-100%       200-300ms
Motion Check (MAE)           ~1-2%          1-2ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY:
1. Run cheap motion detection every 1 second (configurable)
2. Only trigger YOLO when motion detected
3. Result: 97-99% reduction in YOLO calls → No overheating!
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import random

class MotionDetector:
    """Ultra-lightweight motion detection for triggering YOLO"""
    
    def __init__(
        self, 
        tripwire_outer_x: int,
        tripwire_inner_x: int,
        frame_width: int,
        frame_height: int,
        sample_size: int = 500,           # Number of pixels to check
        motion_threshold: int = 15,        # MAE threshold (0-255 scale)
        min_motion_frames: int = 2,        # Require motion in N consecutive frames
        cooldown_frames: int = 90          # Skip frames after detection (prevents spam)
    ):
        """
        Args:
            tripwire_outer_x: Left boundary of detection zone
            tripwire_inner_x: Right boundary of detection zone
            frame_width: Full camera width
            frame_height: Full camera height
            sample_size: How many random pixels to check
            motion_threshold: Minimum pixel difference (higher = less sensitive)
            min_motion_frames: Consecutive frames with motion before triggering
            cooldown_frames: Frames to skip after triggering
        """
        self.zone_x1 = tripwire_outer_x
        self.zone_x2 = tripwire_inner_x
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sample_size = sample_size
        self.motion_threshold = motion_threshold
        self.min_motion_frames = min_motion_frames
        self.cooldown_frames = cooldown_frames
        
        # State
        self.prev_frame: Optional[np.ndarray] = None
        self.motion_count = 0           # Consecutive frames with motion
        self.cooldown_counter = 0       # Frames since last trigger
        
        # Pre-generate random sample coordinates (only in tripwire zone)
        self._generate_sample_points()
        
        # Motion detector initialized (logging removed for cleaner output)
    
    def _generate_sample_points(self):
        """Pre-generate random pixel coordinates in tripwire zone"""
        self.sample_coords = []
        
        for _ in range(self.sample_size):
            x = random.randint(self.zone_x1, min(self.zone_x2, self.frame_width - 1))
            y = random.randint(0, self.frame_height - 1)
            self.sample_coords.append((y, x))  # Note: numpy uses (row, col)
    
    def check_motion(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check if there's significant motion in the frame
        
        Returns:
            (should_run_yolo, mae_score)
        """
        # Handle cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, 0.0
        
        # Convert to grayscale (faster than BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame - just store and return
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0
        
        # Sample random pixels and compute MAE
        current_samples = [gray[y, x] for y, x in self.sample_coords]
        prev_samples = [self.prev_frame[y, x] for y, x in self.sample_coords]
        
        # Mean Absolute Error
        mae = np.mean(np.abs(np.array(current_samples) - np.array(prev_samples)))
        
        # Update previous frame
        self.prev_frame = gray
        
        # Check if motion exceeds threshold
        if mae > self.motion_threshold:
            self.motion_count += 1
        else:
            self.motion_count = 0  # Reset if no motion
        
        # Trigger YOLO only if motion sustained
        if self.motion_count >= self.min_motion_frames:
            self.motion_count = 0
            self.cooldown_counter = self.cooldown_frames
            return True, mae
        
        return False, mae
    
    def reset(self):
        """Reset detector state (call when mode changes)"""
        self.prev_frame = None
        self.motion_count = 0
        self.cooldown_counter = 0


class AdaptiveMotionDetector(MotionDetector):
    """
    Enhanced version that adapts to lighting changes
    
    IMPROVEMENTS:
    - Running average of MAE (detects gradual lighting changes)
    - Dynamic threshold adjustment
    - Better handling of sunrise/sunset
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mae_history = []
        self.history_size = 10  # Track last 10 checks (~10 seconds @ 1fps)
        self.adaptive_multiplier = 0.3  # Much smaller multiplier - only slight adaptation
        self.max_adaptive_threshold = self.motion_threshold * 3  # Cap at 3x base threshold
    
    def check_motion(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Check motion with adaptive threshold"""
        # Handle cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0
        
        # Sample pixels and compute MAE
        current_samples = [gray[y, x] for y, x in self.sample_coords]
        prev_samples = [self.prev_frame[y, x] for y, x in self.sample_coords]
        mae = np.mean(np.abs(np.array(current_samples) - np.array(prev_samples)))
        
        # Update history
        self.mae_history.append(mae)
        if len(self.mae_history) > self.history_size:
            self.mae_history.pop(0)
        
        # Adaptive threshold (ignores gradual changes like lighting)
        # Much more conservative - only slight adjustment above base threshold
        if len(self.mae_history) >= 5:
            avg_mae = np.mean(self.mae_history)
            # Only add a small percentage above base threshold, not above average
            adaptive_threshold = self.motion_threshold + (avg_mae * self.adaptive_multiplier)
            # Cap it to prevent it from getting too high
            adaptive_threshold = min(adaptive_threshold, self.max_adaptive_threshold)
            # Always at least the base threshold
            adaptive_threshold = max(adaptive_threshold, self.motion_threshold)
        else:
            adaptive_threshold = self.motion_threshold
        
        # Update previous frame
        self.prev_frame = gray
        
        # Check if motion exceeds adaptive threshold
        if mae > adaptive_threshold:
            self.motion_count += 1
        else:
            self.motion_count = 0
        
        # Trigger YOLO only if motion sustained
        if self.motion_count >= self.min_motion_frames:
            self.motion_count = 0
            self.cooldown_counter = self.cooldown_frames
            return True, mae
        
        return False, mae
    
    def reset(self):
        """Reset detector state"""
        super().reset()
        self.mae_history = []