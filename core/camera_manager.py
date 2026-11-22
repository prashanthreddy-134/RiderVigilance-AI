import cv2
import time
import threading
import numpy as np

class CameraManager:
    """
    Asynchronous Threaded Camera Manager for Rider Vigilance AI.
    Captures live webcam frames in a dedicated background thread with automatic
    camera index discovery, warm-up synchronization, and robust error fallback.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        self.thread = None
        self.last_error = None

    def find_working_camera_index(self):
        """Discovers active webcam index (0, 1, 2...)."""
        for idx in range(4):
            temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not temp_cap.isOpened():
                temp_cap = cv2.VideoCapture(idx)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                temp_cap.release()
                if ret and frame is not None:
                    return idx
        return 0

    def start(self):
        """Starts background webcam capture thread and waits for initial frame."""
        if self.is_running:
            return
        
        self.last_error = None
        with self.lock:
            self.latest_frame = None
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        print(f"[CameraManager] Started webcam capture thread on index {self.camera_index}")

        # Wait for a real frame (up to 5 seconds), not merely an opened device.
        wait_start = time.time()
        while time.time() - wait_start < 5.0 and self.is_running:
            with self.lock:
                ready = self.latest_frame is not None
            if ready:
                print("[CameraManager] Camera stream ready!")
                return
            time.sleep(0.05)
        if self.is_running:
            self.last_error = self.last_error or "Camera opened but no frame arrived within 5 seconds."
            print(f"[CameraManager] WARNING: {self.last_error}")

    def stop(self):
        """Stops webcam capture and releases camera hardware cleanly."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.latest_frame = None
        print("[CameraManager] Stopped webcam capture and released device.")

    def _capture_worker(self):
        """Background worker thread continuously reading camera frames."""
        idx = self.camera_index
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            idx = self.find_working_camera_index()
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            self.last_error = f"Unable to open any camera (tried index {self.camera_index} and fallbacks)"
            print(f"[CameraManager] ERROR: {self.last_error}")
            self.is_running = False
            return

        self.camera_index = idx

        # Configure low-latency 640x480 standard resolution & minimal buffer
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        print(f"[CameraManager] Successfully opened live webcam on index {self.camera_index} (optimized for low latency)")

        while self.is_running:
            try:
                ret, frame = self.cap.read()
            except Exception as exc:
                self.last_error = f"Camera read exception: {exc}"
                print(f"[CameraManager] ERROR: {self.last_error}")
                time.sleep(0.1)
                continue
            if ret and frame is not None:
                # Standardize frame to 640 width if webcam hardware overrides settings with HD/4K
                if frame.shape[1] > 640:
                    h, w = frame.shape[:2]
                    target_w = 640
                    target_h = int(h * (640.0 / w))
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                with self.lock:
                    self.latest_frame = frame
                self.last_error = None
            else:
                self.last_error = "Failed reading frame from camera"
                time.sleep(0.02)
            time.sleep(0.005)

        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None

    def get_frame(self):
        """Returns latest pre-fetched webcam frame or placeholder if initializing."""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        
        # Draw placeholder frame if camera is starting up
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:, :] = (20, 24, 32)
        cv2.putText(placeholder, "INITIALIZING LIVE WEBCAM FEED...", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        return placeholder
