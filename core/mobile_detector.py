import os
import cv2
import threading
import numpy as np

class MobileDetector:
    """
    YOLOv8-based High-Performance Mobile Phone Detector for Rider Safety.
    Uses OpenCV DNN with YOLOv8 ONNX model to detect mobile phones (COCO Class ID: 67).
    Supports non-blocking asynchronous detection to preserve 30+ FPS video pipeline.
    """
    COCO_CELL_PHONE_CLASS_ID = 67

    def __init__(self, model_path="models/yolov8n.onnx", conf_threshold=0.30, nms_threshold=0.45, async_mode=True):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.async_mode = async_mode

        # Async background worker state
        self.lock = threading.Lock()
        self.latest_detected = False
        self.latest_detections = []
        self.pending_frame = None
        self.is_running = True
        self.frame_ready_event = threading.Event()

        self.load_model()

        if self.async_mode:
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception:
                    pass
                print(f"[MobileDetector] Successfully loaded YOLOv8 ONNX model from {self.model_path}")
            except Exception as e:
                print(f"[MobileDetector] Failed loading YOLO model ({e})")
                self.net = None
        else:
            print(f"[MobileDetector] YOLO model file not found at {self.model_path}")

    def _worker_loop(self):
        """Dedicated background thread continuously running YOLO without blocking main CV pipeline."""
        while self.is_running:
            self.frame_ready_event.wait(timeout=0.1)
            self.frame_ready_event.clear()

            with self.lock:
                if self.pending_frame is None:
                    continue
                frame_to_process = self.pending_frame
                self.pending_frame = None

            detected, detections = self._detect_sync(frame_to_process)

            with self.lock:
                self.latest_detected = detected
                self.latest_detections = detections

    def detect(self, frame):
        """
        Detects cell phone in the frame.
        When async_mode is active, offloads heavy YOLO forward pass to a background worker
        and returns latest detections immediately (<0.01 ms), preserving real-time 30+ FPS.
        """
        if self.net is None or frame is None:
            return False, []

        if not self.async_mode:
            return self._detect_sync(frame)

        # Non-blocking async dispatch: pass newest frame to worker
        with self.lock:
            self.pending_frame = frame
            self.frame_ready_event.set()
            return self.latest_detected, self.latest_detections

    def _detect_sync(self, frame):
        """Synchronous vectorized YOLOv8 detection."""
        if self.net is None or frame is None:
            return False, []

        h, w = frame.shape[:2]

        try:
            # YOLOv8 input preprocessing (640x640)
            blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward()  # Shape: (1, 84, 8400)

            # Transpose to (8400, 84)
            outputs = outputs[0].transpose()

            # Fast vectorized extraction of cell phone scores
            cell_phone_scores = outputs[:, 4 + self.COCO_CELL_PHONE_CLASS_ID]
            mask = cell_phone_scores >= self.conf_threshold

            if not np.any(mask):
                return False, []

            filtered_outputs = outputs[mask]
            filtered_scores = cell_phone_scores[mask]

            x_factor = w / 640.0
            y_factor = h / 640.0

            boxes = []
            confidences = []
            for row, score in zip(filtered_outputs, filtered_scores):
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]

                left = int((cx - 0.5 * bw) * x_factor)
                top = int((cy - 0.5 * bh) * y_factor)
                width = int(bw * x_factor)
                height = int(bh * y_factor)

                boxes.append([left, top, width, height])
                confidences.append(float(score))

            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    detections.append({
                        "box": boxes[i],
                        "confidence": round(confidences[i], 3)
                    })

            return len(detections) > 0, detections

        except Exception as e:
            print(f"[MobileDetector] Error during YOLO detection: {e}")
            return False, []

    def stop(self):
        self.is_running = False
        self.frame_ready_event.set()

