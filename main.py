# rpi_tracker/main.py
"""
Edge Tracker - Headless Mode (No Display)
Ultra lightweight detection with remote dashboard viewing

FIXES:
- Registration freeze bug fixed (non-blocking input)
- Proper state transitions
- Better error handling
- Detection continues during streaming mode

Run with: python3 main.py
Debug mode with display: python3 main.py --display
"""
import sys
import time
import yaml
import cv2
import threading
import argparse
import select
from pathlib import Path
from typing import Optional
from datetime import datetime

from camera import Camera
from detector import PersonDetector
from api_client import ServerClient

# Parse arguments FIRST
parser = argparse.ArgumentParser()
parser.add_argument('--display', action='store_true', help='Enable local display (debug mode)')
args = parser.parse_args()

DISPLAY_ENABLED = args.display

# Load configuration
def load_config():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print("⚠️  config.yaml not found")
        sys.exit(1)

config = load_config()

# Configuration
SERVER_URL = config['server']['url']
RPI_ID = config['server']['rpi_id']
CAMERA_WIDTH = config['camera']['width']
CAMERA_HEIGHT = config['camera']['height']
CAMERA_FPS = config['camera']['fps']
DETECTION_SCALE = config['detection']['scale']
DETECTION_CONFIDENCE = config['detection']['confidence']
CHECK_INTERVAL = config['detection']['check_interval']
BURST_SIZE = config['capture']['burst_size']
BURST_INTERVAL = config['capture']['burst_interval']
EVENT_COOLDOWN = config['capture']['event_cooldown']
STREAM_FPS = config['camera']['stream_fps']
STREAM_QUALITY = config['camera']['stream_quality']
MODEL_PATH = Path(config['model_path'])

# Get tripwire config from server on startup
def get_tripwire_config():
    try:
        import requests
        response = requests.get(f"{SERVER_URL}/api/rpi/config/{RPI_ID}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['tripwires']['outer_x'], data['tripwires']['inner_x']
    except:
        pass
    # Fallback to local config
    return config['tripwires']['outer_x'], config['tripwires']['inner_x']

TRIPWIRE_OUTER_X, TRIPWIRE_INNER_X = get_tripwire_config()

# FIX: Non-blocking input handler
class InputHandler:
    def __init__(self):
        self.command = None
        self.lock = threading.Lock()
        self.running = True
    
    def start(self):
        thread = threading.Thread(target=self._input_loop, daemon=True)
        thread.start()
    
    def _input_loop(self):
        """Non-blocking input loop"""
        while self.running:
            try:
                # Non-blocking check for input
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    cmd = sys.stdin.readline().strip()
                    with self.lock:
                        self.command = cmd
            except:
                time.sleep(0.1)
    
    def get_command(self) -> Optional[str]:
        with self.lock:
            cmd = self.command
            self.command = None
            return cmd
    
    def stop(self):
        self.running = False

# Debug display functions (only used if --display flag)
def draw_calibration_overlay(frame, outer_x, inner_x):
    """Draw tripwires for calibration only"""
    overlay = frame.copy()
    cv2.line(overlay, (outer_x, 0), (outer_x, frame.shape[0]), (0, 255, 255), 3)
    cv2.line(overlay, (inner_x, 0), (inner_x, frame.shape[0]), (0, 0, 255), 3)
    
    # Add labels
    cv2.putText(overlay, "CALIBRATION MODE", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(overlay, f"Outer: {outer_x}px", (outer_x + 10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(overlay, f"Inner: {inner_x}px", (inner_x + 10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return overlay

def draw_status_overlay(frame, mode, count=0):
    """Minimal status overlay"""
    if mode == "detecting":
        color = (255, 255, 0)
        text = "DETECTING"
    elif mode == "burst":
        color = (0, 255, 0)
        text = f"CAPTURING {count}/{BURST_SIZE}"
    elif mode == "registration":
        color = (255, 0, 255)
        text = f"REGISTERING {count}/{BURST_SIZE}"
    elif mode == "streaming":
        color = (0, 255, 255)
        text = "STREAMING TO DASHBOARD"
    else:
        color = (100, 100, 100)
        text = "IDLE"
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def main():
    print("=" * 70)
    if DISPLAY_ENABLED:
        print("  Edge Tracker - Debug Mode (Display Enabled)")
    else:
        print("  Edge Tracker - Headless Mode (No Display)")
    print("=" * 70)
    print(f"  Server: {SERVER_URL}")
    print(f"  RPi ID: {RPI_ID}")
    print(f"  Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"  Detection: Every {CHECK_INTERVAL}s at {int(DETECTION_SCALE*100)}% scale")
    print(f"  Stream: {STREAM_FPS}fps @ {STREAM_QUALITY}% quality")
    print(f"  Tripwires: Outer={TRIPWIRE_OUTER_X}, Inner={TRIPWIRE_INNER_X}")
    print("=" * 70)
    print()
    
    # Initialize components
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    camera = Camera(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    detector = PersonDetector(MODEL_PATH, DETECTION_SCALE, DETECTION_CONFIDENCE)
    server = ServerClient(SERVER_URL, RPI_ID)
    input_handler = InputHandler()
    input_handler.start()
    
    camera.start()
    
    print("\n🎛️  Commands:")
    print("  r - Register person")
    print("  q - Quit")
    print()
    
    # State
    mode = "idle"
    last_check_time = 0
    burst_images = []
    burst_count = 0
    last_burst_time = 0
    last_event_time = 0
    last_heartbeat = 0
    last_command_check = 0
    last_stream_time = 0
    
    registration_mode = False
    registration_images = []
    registration_count = 0
    last_registration_capture = 0
    registration_name_pending = False
    
    streaming_mode = False
    calibration_mode = False
    
    uptime_start = time.time()
    
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("⚠️  Camera stream ended")
                break
            
            current_time = time.time()
            uptime = int(current_time - uptime_start)
            
            # Send heartbeat
            if current_time - last_heartbeat >= config['server']['heartbeat_interval']:
                server.send_heartbeat(mode, uptime)
                last_heartbeat = current_time
            
            # Check for dashboard commands
            if current_time - last_command_check >= config['server']['command_poll_interval']:
                cmd_data = server.check_commands()
                cmd = cmd_data.get('command') if cmd_data else None
                
                if cmd == 'register' and not registration_mode:
                    print("\n📸 Registration triggered from dashboard!")
                    registration_mode = True
                    registration_images = []
                    registration_count = 0
                    last_registration_capture = 0
                    registration_name_pending = False
                    mode = "registration"
                    
                elif cmd == 'start_stream':
                    print("\n📹 Streaming started from dashboard")
                    streaming_mode = True
                    if mode == "idle":
                        mode = "streaming"
                    last_stream_time = 0
                    
                elif cmd == 'stop_stream':
                    print("\n⏸️  Streaming stopped")
                    streaming_mode = False
                    if not registration_mode and not calibration_mode:
                        mode = "idle"
                    
                elif cmd == 'calibrate':
                    print("\n⚙️  Calibration mode activated!")
                    calibration_mode = True
                    streaming_mode = True
                    mode = "calibration"
                    last_stream_time = 0
                    
                elif cmd == 'stop_calibrate':
                    print("\n✅ Calibration mode exited")
                    calibration_mode = False
                    streaming_mode = False
                    mode = "idle"
                
                last_command_check = current_time
            
            # STREAMING MODE - Send frames to dashboard
            if streaming_mode and (current_time - last_stream_time >= 1.0/STREAM_FPS):
                # Prepare frame for streaming
                if calibration_mode:
                    stream_frame = draw_calibration_overlay(frame.copy(), TRIPWIRE_OUTER_X, TRIPWIRE_INNER_X)
                else:
                    stream_frame = frame
                
                # Send to server
                server.send_stream_frame(stream_frame, STREAM_QUALITY)
                last_stream_time = current_time
            
            # REGISTRATION MODE - Fully automatic, no prompting
            if registration_mode:
                # Capture burst
                if current_time - last_registration_capture >= BURST_INTERVAL:
                    registration_images.append(frame.copy())
                    registration_count += 1
                    last_registration_capture = current_time
                    print(f"  📷 Captured {registration_count}/{BURST_SIZE}")
                
                if registration_count >= BURST_SIZE:
                    # Automatically register as unlabeled (no name prompt)
                    print("  📤 Uploading to server (unlabeled)...")
                    result = server.register_person(None, registration_images)
                    
                    if result.get('status') == 'success':
                        person_id = result.get('person_id')
                        print(f"✅ Registered: {person_id}")
                        print("   ⚠️  Unlabeled - assign name via dashboard")
                    else:
                        print(f"❌ Registration failed: {result}")
                    
                    # Cleanup and return to previous mode
                    registration_mode = False
                    registration_images = []
                    registration_count = 0
                    mode = "streaming" if streaming_mode else "idle"
                    print("\n🎛️  Ready for next command")
            
            # BURST CAPTURE MODE
            elif mode == "burst":
                if current_time - last_burst_time >= BURST_INTERVAL:
                    burst_images.append(frame.copy())
                    burst_count += 1
                    last_burst_time = current_time
                    print(f"  📷 {burst_count}/{BURST_SIZE}")
                
                if burst_count >= BURST_SIZE:
                    # Send to server
                    print("  📤 Sending to server...")
                    result = server.send_event("enter", burst_images)
                    
                    if result.get('status') == 'success':
                        person_id = result.get('person_id')
                        name = result.get('name', 'Unknown')
                        similarity = result.get('similarity', 0)
                        print(f"✅ Recognized: {name} ({person_id}) - {similarity:.2%}")
                    elif result.get('status') == 'unknown_registered':
                        person_id = result.get('person_id')
                        print(f"⚠️  Unknown person registered: {person_id}")
                    else:
                        print(f"❌ Event failed: {result}")
                    
                    burst_images = []
                    burst_count = 0
                    mode = "streaming" if streaming_mode else "idle"
                    last_event_time = current_time
            
            # DETECTION MODE - Always runs unless in calibration mode or registration
            # KEY FIX: Removed the "not streaming_mode" condition
            if not calibration_mode and not registration_mode and mode != "burst":
                # Run detection at intervals
                if current_time - last_check_time >= CHECK_INTERVAL:
                    detections = detector.detect(frame)
                    last_check_time = current_time
                    
                    # Check if anyone in zone
                    people_in_zone = False
                    for x, y, w, h, conf in detections:
                        center_x = x + w // 2
                        if TRIPWIRE_OUTER_X < center_x < TRIPWIRE_INNER_X:
                            people_in_zone = True
                            break
                    
                    if people_in_zone and (current_time - last_event_time >= EVENT_COOLDOWN):
                        # Start burst capture
                        print(f"\n👤 Person detected at {datetime.now().strftime('%H:%M:%S')}")
                        mode = "burst"
                        burst_images = [frame.copy()]
                        burst_count = 1
                        last_burst_time = current_time
            
            # Display (only if --display flag enabled)
            if DISPLAY_ENABLED:
                if calibration_mode:
                    display_frame = draw_calibration_overlay(frame.copy(), TRIPWIRE_OUTER_X, TRIPWIRE_INNER_X)
                else:
                    display_frame = draw_status_overlay(frame.copy(), mode, 
                                                       burst_count if mode == "burst" else registration_count)
                
                # Small display
                display_w = int(CAMERA_WIDTH * 0.25)
                display_h = int(CAMERA_HEIGHT * 0.25)
                small_frame = cv2.resize(display_frame, (display_w, display_h), 
                                        interpolation=cv2.INTER_NEAREST)
                cv2.imshow('Edge Tracker', small_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and not registration_mode:
                    print("\n📝 Starting registration...")
                    registration_mode = True
                    registration_images = []
                    registration_count = 0
                    last_registration_capture = 0
                    registration_name_pending = False
                    mode = "registration"
            
            # Terminal input (non-blocking)
            if not registration_name_pending:  # Don't process commands while waiting for name
                cmd = input_handler.get_command()
                if cmd == 'r' and not registration_mode:
                    print("\n📝 Starting registration...")
                    registration_mode = True
                    registration_images = []
                    registration_count = 0
                    last_registration_capture = 0
                    registration_name_pending = False
                    mode = "registration"
                elif cmd == 'q':
                    break
    
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        input_handler.stop()
        camera.stop()
        if DISPLAY_ENABLED:
            cv2.destroyAllWindows()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
