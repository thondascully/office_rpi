# Raspberry Pi 5 Setup Guide - Edge Tracker

## 📋 What You Need

- Raspberry Pi 5 (4GB or 8GB RAM recommended)
- Raspberry Pi Camera Module (v2, v3, or HQ camera)
- MicroSD card with Raspberry Pi OS (64-bit recommended)
- Internet connection (WiFi or Ethernet)
- Your MacBook server running on the same network

## 🚀 Quick Setup Steps

### 1. Connect to Your RPi5

```bash
# SSH into your RPi5
ssh pi@raspberrypi.local

# Or use the IP address
ssh pi@192.168.1.XXX
```

### 2. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip (should already be installed)
sudo apt install python3-pip python3-venv -y

# Install camera libraries
sudo apt install python3-picamera2 -y

# Install OpenCV dependencies
sudo apt install libopencv-dev python3-opencv -y

# Install other system libraries
sudo apt install libatlas-base-dev libhdf5-dev -y
```

### 3. Create Project Directory

```bash
# Create project directory
mkdir -p ~/rpi_tracker_project
cd ~/rpi_tracker_project

# Create models directory
mkdir -p models
```

### 4. Copy Files to RPi5

From your MacBook, copy the edge_tracker.py file:

```bash
# On your MacBook
scp edge_tracker.py pi@raspberrypi.local:~/rpi_tracker_project/
scp rpi_requirements.txt pi@raspberrypi.local:~/rpi_tracker_project/requirements.txt
```

### 5. Install Python Dependencies

```bash
# On your RPi5
cd ~/rpi_tracker_project

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Download YOLOv8n Model

```bash
# Download YOLOv8n ONNX model (6 MB)
cd ~/rpi_tracker_project
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx -O models/yolov8n.onnx

# Verify download
ls -lh models/yolov8n.onnx
# Should show ~6 MB
```

### 7. Configure Server Connection

Edit the edge_tracker.py file to point to your MacBook:

```bash
nano edge_tracker.py
```

Find this line near the top:
```python
BACKEND_API_URL = "http://192.168.1.100:8000"  # ← CHANGE THIS!
```

Replace with your MacBook's IP address. To find your MacBook's IP:

**On MacBook:**
```bash
# For WiFi
ipconfig getifaddr en0

# For Ethernet
ipconfig getifaddr en1

# Or use ifconfig and look for inet address
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Example: If your MacBook's IP is `192.168.1.150`, change to:
```python
BACKEND_API_URL = "http://192.168.1.150:8000"
```

Save and exit (Ctrl+X, Y, Enter)

### 8. Test Camera Connection

```bash
# Quick camera test
rpicam-hello
```

You should see a preview window. Press Ctrl+C to close.

If you get an error, make sure:
- Camera cable is properly connected
- Camera is enabled in `sudo raspi-config` → Interface Options → Camera

### 9. Run the Edge Tracker!

```bash
cd ~/rpi_tracker_project
source venv/bin/activate  # If using virtual environment
python3 edge_tracker.py
```

You should see:
```
======================================================================
  Edge Tracker - Raspberry Pi 5
  Office Occupancy Map - 'Sensor' Component
======================================================================

🧠 Loading YOLOv8n model from models/yolov8n.onnx...
✅ YOLOv8n loaded. Input shape: [1, 3, 640, 640]
📷 Initializing camera...
✅ Camera ready

======================================================================
  SYSTEM READY
======================================================================
Server: http://192.168.1.150:8000
Tripwires: OUTER=200px, INNER=440px

Commands:
  r      - Register new member (capture 10 frames)
  q      - Quit
======================================================================
```

## 🎯 How to Use

### Register a New Member

1. Have the person stand in front of the camera
2. Press `r` (or type `r` and press Enter)
3. The system will capture 10 frames
4. Enter a member_id (e.g., `alice_smith`)
5. The frames are sent to the server for registration

### Test Entry/Exit Detection

1. Have a registered person walk past the camera
2. They should cross both tripwires (yellow and red lines in the display)
3. When they cross the inner tripwire, an event is sent to the server
4. Check the server logs to see the recognition result

## 🔧 Configuration & Tuning

### Adjust Tripwire Positions

In `edge_tracker.py`, modify these values:

```python
TRIPWIRE_OUTER_X = 200  # Outer boundary (pixels from left)
TRIPWIRE_INNER_X = 440  # Inner boundary (pixels from left)
```

- Set them based on your door location
- Watch the video preview to see where people walk
- Adjust so outer tripwire is before the door, inner is after

### Adjust Detection Confidence

```python
MIN_DETECTION_CONFIDENCE = 0.5  # Range: 0.0 to 1.0
```

- **Too many false detections?** Increase to `0.6` or `0.7`
- **Missing people?** Decrease to `0.4`

### Adjust Buffer Size

```python
BUFFER_SIZE = 10  # Number of images to collect
```

- More images = more reliable face recognition, but slower
- Fewer images = faster, but less reliable

### Camera Settings

```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
```

## 🐛 Troubleshooting

### Problem: "Cannot connect to server"

**Check 1**: Is your MacBook server running?
```bash
# On MacBook
curl http://localhost:8000/state
```

**Check 2**: Can RPi5 reach MacBook?
```bash
# On RPi5
ping 192.168.1.150  # Replace with your MacBook IP
curl http://192.168.1.150:8000/state
```

**Check 3**: Firewall blocking?
- On MacBook, check System Settings → Network → Firewall
- Temporarily disable to test

### Problem: "No module named 'picamera2'"

**Solution:**
```bash
sudo apt install python3-picamera2 -y
```

### Problem: Camera not detected

**Check connection:**
```bash
# List cameras
rpicam-hello --list-cameras

# Should show something like:
# 0 : imx219 [3280x2464] (/base/soc/i2c0mux/i2c@1/imx219@10)
```

**Enable camera:**
```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
sudo reboot
```

### Problem: YOLOv8n running slow (< 5 FPS)

**Causes:**
- RPi5 CPU is doing all the work
- This is expected - YOLOv8n should run at 15-30 FPS on RPi5

**Solutions:**
1. Lower resolution:
   ```python
   CAMERA_WIDTH = 320
   CAMERA_HEIGHT = 240
   ```

2. Skip frames:
   ```python
   # Only process every 2nd frame
   frame_count = 0
   if frame_count % 2 == 0:
       detections = detector.detect(frame)
   frame_count += 1
   ```

### Problem: "Person not recognized" on server

**Causes:**
- Person not registered
- Poor image quality (blurry, dark, angle)

**Solutions:**
1. Check if person is registered:
   ```bash
   curl http://192.168.1.150:8000/debug/members
   ```

2. Re-register with better lighting/position

3. Adjust similarity threshold on server

### Problem: Window not displaying

**If running headless (no monitor):**

You can disable the display by commenting out:
```python
# cv2.imshow('Edge Tracker', frame)
# key = cv2.waitKey(1) & 0xFF
```

And use terminal-only mode.

## 📊 Performance Expectations

**Raspberry Pi 5 (typical):**
- YOLOv8n inference: ~30-50ms (20-33 FPS)
- Overall pipeline: 15-30 FPS
- Memory usage: ~500-800 MB

**CPU temperature:**
- Normal: 40-60°C
- Under load: 60-75°C
- If > 80°C, consider adding heatsink/fan

Check temperature:
```bash
vcgencmd measure_temp
```

## 🔄 Running on Boot (Optional)

To make the tracker start automatically on boot:

### Create systemd service:

```bash
sudo nano /etc/systemd/system/edge-tracker.service
```

Add:
```ini
[Unit]
Description=Office Map Edge Tracker
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rpi_tracker_project
ExecStart=/home/pi/rpi_tracker_project/venv/bin/python3 /home/pi/rpi_tracker_project/edge_tracker.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable edge-tracker.service
sudo systemctl start edge-tracker.service

# Check status
sudo systemctl status edge-tracker.service

# View logs
sudo journalctl -u edge-tracker.service -f
```

## 🎥 Camera Positioning Tips

1. **Mount camera at door height** (5-6 feet)
2. **Angle slightly downward** to capture faces
3. **Ensure good lighting** - face the camera away from windows
4. **Test the tripwire positions** by walking through and watching the overlay
5. **Adjust OUTER and INNER lines** so people cross both when entering

## 📁 Your RPi5 Directory Structure

```
~/rpi_tracker_project/
├── edge_tracker.py           ← Main script
├── requirements.txt          ← Python dependencies
├── venv/                     ← Virtual environment (if using)
└── models/
    └── yolov8n.onnx         ← YOLOv8n model (6 MB)
```

## 🔗 Network Architecture

```
[RPi5 Camera]
      ↓
[RPi5: YOLOv8n Detection & Tracking]
      ↓
   (WiFi/Ethernet)
      ↓
[MacBook: ArcFace Face Recognition]
      ↓
[MacBook: State Management]
      ↓
[Web Browser: Office Map Display]
```

## 📚 Next Steps

1. ✅ RPi5 set up and running
2. Register your first member (press `r`)
3. Test walk-through detection
4. Adjust tripwire positions if needed
5. Add more members
6. Consider auto-start on boot
7. Add more RPi5 cameras for multiple entrances!

## 🆘 Still Having Issues?

Check these logs for detailed error messages:

**On RPi5:**
```bash
# If running directly
# Errors show in terminal

# If running as service
sudo journalctl -u edge-tracker.service -n 50
```

**On MacBook server:**
```bash
# Server logs show recognition details
# Look for similarity scores and matching results
```

The system logs everything, so you can debug by reading the output!
