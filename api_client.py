# rpi_tracker/api_client.py
"""
API Client - Lightweight communication with server
"""
import cv2
import requests
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime

class ServerClient:
    """Minimal client for server communication"""
    
    def __init__(self, server_url: str, rpi_id: str):
        self.server_url = server_url.rstrip('/')
        self.rpi_id = rpi_id
        self.session = requests.Session()  # Reuse connections
    
    def send_heartbeat(self, status: str, uptime: int) -> bool:
        """Send heartbeat to server"""
        try:
            response = self.session.post(
                f"{self.server_url}/api/rpi/heartbeat",
                json={
                    'rpi_id': self.rpi_id,
                    'status': status,
                    'uptime': uptime
                },
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            # Silent fail on heartbeat
            return False
    
    def check_commands(self) -> Optional[Dict]:
        """Check for commands from dashboard"""
        try:
            response = self.session.get(
                f"{self.server_url}/api/rpi/commands/{self.rpi_id}",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def send_event(self, direction: str, images: List[np.ndarray]) -> dict:
        """Send detection event to server"""
        files = []
        for i, img in enumerate(images):
            # Compress with lower quality for network efficiency
            _, buffer = cv2.imencode('.jpg', img, [
                cv2.IMWRITE_JPEG_QUALITY, 85,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            files.append(('images', (f'frame_{i}.jpg', buffer.tobytes(), 'image/jpeg')))
        
        try:
            response = self.session.post(
                f"{self.server_url}/api/event",
                data={
                    'direction': direction,
                    'rpi_id': self.rpi_id,
                    'timestamp': datetime.now().isoformat()
                },
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error"}
        except Exception as e:
            return {"status": "error"}
    
    def register_person(self, name: Optional[str], images: List[np.ndarray]) -> dict:
        """Register new person"""
        files = []
        for i, img in enumerate(images):
            # High quality for registration
            _, buffer = cv2.imencode('.jpg', img, [
                cv2.IMWRITE_JPEG_QUALITY, 95
            ])
            files.append(('images', (f'reg_{i}.jpg', buffer.tobytes(), 'image/jpeg')))
        
        data = {'rpi_id': self.rpi_id}
        if name:
            data['name'] = name
        
        try:
            response = self.session.post(
                f"{self.server_url}/api/register",
                data=data,
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error"}
        except Exception as e:
            return {"status": "error"}
    
    def send_stream_frame(self, frame: np.ndarray, quality: int = 70) -> bool:
        """Send single frame for live streaming to dashboard"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, quality
            ])
            
            response = self.session.post(
                f"{self.server_url}/api/rpi/stream/{self.rpi_id}",
                data=buffer.tobytes(),
                headers={'Content-Type': 'image/jpeg'},
                timeout=2  # Fast timeout for streaming
            )
            return response.status_code == 200
        except:
            return False
