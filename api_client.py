# rpi_tracker/api_client.py
"""
API Client - Lightweight communication with server
"""
import cv2
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime

class ServerClient:
    """Minimal client for server communication"""
    
    def __init__(self, server_url: str, rpi_id: str):
        self.server_url = server_url.rstrip('/')
        self.rpi_id = rpi_id
        self.session = requests.Session()  # Reuse connections
        
        # Configure connection pooling to prevent overwhelming the server
        # Limit to 2 connections per host to avoid connection exhaustion
        adapter = HTTPAdapter(
            pool_connections=1,  # Number of connection pools to cache
            pool_maxsize=2,      # Max connections per pool (prevents overwhelming server)
            max_retries=Retry(
                total=2,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET", "POST"]
            )
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
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
                try:
                    return response.json()
                except Exception as e:
                    print(f"ERROR: Failed to parse server response: {e}")
                    return {"status": "error", "message": f"JSON parse error: {e}"}
            else:
                print(f"ERROR: Server returned status {response.status_code}")
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except requests.exceptions.Timeout:
            print("ERROR: Request to server timed out")
            return {"status": "error", "message": "Request timeout"}
        except requests.exceptions.ConnectionError as e:
            print(f"ERROR: Connection to server failed: {e}")
            return {"status": "error", "message": "Connection failed"}
        except Exception as e:
            print(f"ERROR: Unexpected error in send_event: {e}")
            return {"status": "error", "message": str(e)}
    
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
    
    def update_person_name(self, person_id: str, name: str) -> dict:
        """Update a person's name (for unlabeled persons that get labeled)"""
        try:
            response = self.session.post(
                f"{self.server_url}/api/person/{person_id}/update",
                json={
                    'name': name,
                    'rpi_id': self.rpi_id
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
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
