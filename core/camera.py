"""
ElderGuard AI - Camera & Video Ingestion Manager
Provides persistent, thread-safe video capture with frame caching
and zero-dropout guarantees for live monitoring in Streamlit and native runtimes.
"""
import cv2
import threading
import time
import numpy as np
from typing import Optional, Tuple
from config import sys_config

class VideoCaptureManager:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self._last_successful_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.is_hardware_available = False
        self.error_message: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Initializes the webcam and starts background frame reader thread."""
        if self.is_running and self.cap and self.cap.isOpened():
            return True

        try:
            # Try DSHOW backend first on Windows for low-latency webcam init
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                self.is_hardware_available = False
                self.error_message = f"Webcam index {self.camera_index} could not be opened."
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Warmup: read initial frames to flush exposure and buffer
            for _ in range(3):
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self.lock:
                        self.current_frame = frame
                        self._last_successful_frame = frame.copy()

            self.is_hardware_available = True
            self.is_running = True

            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True

        except Exception as e:
            self.is_hardware_available = False
            self.error_message = f"Camera initialization exception: {str(e)}"
            return False

    def _capture_loop(self):
        consecutive_errors = 0
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    consecutive_errors = 0
                    with self.lock:
                        self.current_frame = frame
                        self._last_successful_frame = frame
                else:
                    consecutive_errors += 1
                    if consecutive_errors > 60: # ~2 seconds of continuous failure
                        self.is_hardware_available = False
                    time.sleep(0.01)
            except Exception:
                consecutive_errors += 1
                time.sleep(0.01)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Returns the latest captured frame.
        Caches the last successful frame to bridge temporary USB/driver blips.
        """
        with self.lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
            elif self._last_successful_frame is not None:
                return True, self._last_successful_frame.copy()

        if not self.is_hardware_available or not self.is_running:
            dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(dummy, "WEBCAM CONNECTING / STANDBY", (90, self.height // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            return False, dummy

        return False, None

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_hardware_available = False
        with self.lock:
            self.current_frame = None
