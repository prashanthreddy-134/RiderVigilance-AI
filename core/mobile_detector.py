
import os
import cv2
import threading
import numpy as np


class MobileDetector:
    """
    YOLOv8 ONNX mobile phone detector for RiderVigilance AI.

    Uses OpenCV DNN with CPU inference and an asynchronous worker
    to avoid blocking the main video-processing pipeline.
    """

    COCO_CELL_PHONE_CLASS_ID = 67

    def __init__(
        self,
        model_path="models/yolov8n.onnx",
        conf_threshold=0.30,
        nms_threshold=0.45,
        async_mode=True
    ):
        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.async_mode = async_mode

        self.net = None

        self.lock = threading.Lock()
        self.latest_detected = False
        self.latest_detections = []

        self.pending_frame = None
        self.is_running = True
        self.frame_ready_event = threading.Event()

        self.load_model()

        self.worker_thread = None

        if self.async_mode and self.net is not None:
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="MobileDetectorWorker"
            )
            self.worker_thread.start()

    def load_model(self):
        """Load the YOLOv8 ONNX model."""

        if not os.path.isfile(self.model_path):
            print(
                f"[MobileDetector] Model file not found: "
                f"{self.model_path}"
            )
            self.net = None
            return

        try:
            self.net = cv2.dnn.readNetFromONNX(
                self.model_path
            )

            self.net.setPreferableBackend(
                cv2.dnn.DNN_BACKEND_OPENCV
            )

            self.net.setPreferableTarget(
                cv2.dnn.DNN_TARGET_CPU
            )

            print(
                "[MobileDetector] YOLOv8 ONNX model loaded: "
                f"{self.model_path}"
            )

        except Exception as error:
            print(
                f"[MobileDetector] Model loading failed: {error}"
            )
            self.net = None

    def _worker_loop(self):
        """Process the newest available frame in the background."""

        while self.is_running:

            self.frame_ready_event.wait(timeout=0.2)
            self.frame_ready_event.clear()

            if not self.is_running:
                break

            with self.lock:
                if self.pending_frame is None:
                    continue

                frame_to_process = self.pending_frame
                self.pending_frame = None

            detected, detections = self._detect_sync(
                frame_to_process
            )

            with self.lock:
                self.latest_detected = detected
                self.latest_detections = detections

    def detect(self, frame):
        """
        Return the latest phone detection.

        In asynchronous mode, the newest frame is sent to the
        background worker and the latest completed result is returned.
        """

        if self.net is None or frame is None:
            return False, []

        if not self.async_mode:
            return self._detect_sync(frame)

        if not self.is_running:
            return False, []

        # Copy the frame so the caller can safely reuse its buffer.
        frame_copy = frame.copy()

        with self.lock:
            self.pending_frame = frame_copy

            latest_result = (
                self.latest_detected,
                list(self.latest_detections)
            )

            self.frame_ready_event.set()

        return latest_result

    def _detect_sync(self, frame):
        """Run YOLOv8 inference and apply phone-class filtering."""

        if self.net is None or frame is None:
            return False, []

        height, width = frame.shape[:2]

        try:
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0 / 255.0,
                size=(640, 640),
                swapRB=True,
                crop=False
            )

            self.net.setInput(blob)
            outputs = self.net.forward()

            if outputs is None or outputs.size == 0:
                return False, []

            # Expected YOLOv8 COCO output: (1, 84, 8400).
            if outputs.ndim != 3 or outputs.shape[0] != 1:
                print(
                    "[MobileDetector] Unexpected output shape: "
                    f"{outputs.shape}"
                )
                return False, []

            predictions = outputs[0]

            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.transpose()

            expected_features = 4 + 80

            if predictions.shape[1] != expected_features:
                print(
                    "[MobileDetector] Unexpected prediction layout: "
                    f"{predictions.shape}"
                )
                return False, []

            phone_scores = predictions[
                :,
                4 + self.COCO_CELL_PHONE_CLASS_ID
            ]

            valid_mask = phone_scores >= self.conf_threshold

            if not np.any(valid_mask):
                return False, []

            selected = predictions[valid_mask]
            selected_scores = phone_scores[valid_mask]

            x_factor = width / 640.0
            y_factor = height / 640.0

            boxes = []
            confidences = []

            for row, score in zip(
                selected,
                selected_scores
            ):

                cx, cy, box_width, box_height = row[:4]

                left = int(
                    (cx - box_width / 2.0) * x_factor
                )

                top = int(
                    (cy - box_height / 2.0) * y_factor
                )

                box_width = int(
                    box_width * x_factor
                )

                box_height = int(
                    box_height * y_factor
                )

                left = max(0, min(left, width - 1))
                top = max(0, min(top, height - 1))

                box_width = max(
                    1,
                    min(box_width, width - left)
                )

                box_height = max(
                    1,
                    min(box_height, height - top)
                )

                boxes.append([
                    left,
                    top,
                    box_width,
                    box_height
                ])

                confidences.append(float(score))

            indices = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                self.conf_threshold,
                self.nms_threshold
            )

            detections = []

            if len(indices) > 0:

                for index in np.array(indices).flatten():

                    detections.append({
                        "box": boxes[int(index)],
                        "confidence": round(
                            confidences[int(index)],
                            3
                        )
                    })

            return (
                len(detections) > 0,
                detections
            )

        except Exception as error:
            print(
                f"[MobileDetector] YOLO inference error: {error}"
            )
            return False, []

    def stop(self):
        """Stop the background detection worker safely."""

        self.is_running = False
        self.frame_ready_event.set()

        if (
            self.worker_thread is not None
            and self.worker_thread.is_alive()
            and threading.current_thread() is not self.worker_thread
        ):
            self.worker_thread.join(timeout=2.0)

        with self.lock:
            self.pending_frame = None
            self.latest_detected = False
            self.latest_detections = []
