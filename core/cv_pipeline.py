import os
import cv2
import time
import math
import numpy as np
from collections import deque
from core.fatigue_classifier import FatigueClassifier
from core.logger import SessionLogger
from core.mobile_detector import MobileDetector

# Try importing MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"[CVPipeline] MediaPipe import warning: {e}")

class ComputerVisionPipeline:
    """
    Computer Vision & Facial Analytics Pipeline for Rider Fatigue Monitoring.
    Calculates Left, Right, and Average EAR and continuous Mouth Aspect Ratio (MAR).
    Supports dual temporal detection:
      1. Eye Closure Drowsiness (Continuous Closure >= 2.0s)
      2. Temporal Yawn Detection (Sustained MAR > Threshold for >= 1.5s)
      3. YOLO Real-Time Mobile Phone Detection ("DO NOT USE MOBILE WHILE DRIVING")
    All events trigger visual banners and reusable audio alarm events.
    """

    LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380] # p1, p2, p3, p4, p5, p6
    RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 144, 153] # p1, p2, p3, p4, p5, p6
    MOUTH_LANDMARKS = [61, 291, 13, 14, 82, 87]        # left, right, upper, lower, upper2, lower2

    def __init__(self, use_simulation=False, model_path="models/face_landmarker.task", yolo_model_path="models/yolov8n.onnx"):
        self.use_simulation = use_simulation
        self.classifier = FatigueClassifier()
        self.logger = SessionLogger()
        self.mobile_detector = MobileDetector(model_path=yolo_model_path)
        self.model_path = model_path
        
        self.landmarker = None
        self.init_landmarker()

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Configurable Eye & Yawn Thresholds
        self.eye_closed_threshold = 0.260
        self.yawn_mar_threshold = 0.500
        self.yawn_min_duration = 1.50      # Minimum seconds open mouth needed to confirm yawn

        # Dynamic Rider Calibration
        self.calibration_frames = 0
        self.calibration_max_frames = 25
        self.is_calibrated = False
        self.ear_baseline_samples = []
        self.mar_baseline_samples = []
        self.ear_baseline = 0.320
        self.mar_baseline = 0.160

        # Frame and Time Trackers
        self.frame_count = 0
        self.start_time = time.time()
        
        self.smoothed_left_ear = 0.32
        self.smoothed_right_ear = 0.32
        self.smoothed_avg_ear = 0.32
        self.smoothed_mar = 0.16
        
        self.ear_history = deque(maxlen=30)
        self.mar_history = deque(maxlen=30)
        self.blink_timestamps = deque(maxlen=100)
        self.yawn_timestamps = deque(maxlen=50)
        
        # 1. Eye Closure Temporal Tracking
        self.eye_closed_start_time = None
        self.current_closure_duration = 0.0
        self.eye_alarm_triggered = False

        # 2. Yawn Temporal Tracking
        self.yawn_start_time = None
        self.current_yawn_duration = 0.0
        self.yawn_alarm_triggered = False

        # 3. Repeated Short Eye Closures (Early Sleepiness) Config & Trackers
        self.short_closure_min = 0.95       # Minimum seconds (~1s) for a short closure event
        self.short_closure_max = 2.00       # Maximum seconds for a short closure event (< 2.0s)
        self.sleepiness_event_count = 3    # Threshold count (3 or 4 short closures within window)
        self.sleepiness_window = 60.0      # Rolling time window in seconds
        self.short_closure_events = deque(maxlen=50) # Timestamps of recorded short closure events
        self.sleepiness_alarm_triggered = False
        
        self.is_eye_closed_prev = False
        self.is_yawning_prev = False
        
        self.sim_scenario = "normal"
        self.sim_t = 0.0

    def init_landmarker(self):
        """Initializes MediaPipe FaceLandmarker."""
        if MEDIAPIPE_AVAILABLE and os.path.exists(self.model_path):
            try:
                base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                print(f"[CVPipeline] Successfully loaded MediaPipe FaceLandmarker from {self.model_path}")
            except Exception as e:
                print(f"[CVPipeline] Failed loading MediaPipe FaceLandmarker: {e}")
                self.landmarker = None
        else:
            print(f"[CVPipeline] MediaPipe model path missing or MediaPipe unavailable: {self.model_path}")

    def reset_calibration(self):
        """Resets baseline calibration."""
        self.calibration_frames = 0
        self.is_calibrated = False
        self.ear_baseline_samples = []
        self.mar_baseline_samples = []
        print("[CVPipeline] Baseline calibration reset.")

    def set_eye_closed_threshold(self, value: float):
        self.eye_closed_threshold = max(0.12, min(0.35, float(value)))
        print(f"[CVPipeline] Eye Closed Threshold updated to: {self.eye_closed_threshold:.3f}")

    def set_yawn_mar_threshold(self, value: float):
        self.yawn_mar_threshold = max(0.35, min(0.75, float(value)))
        print(f"[CVPipeline] Yawn MAR Threshold updated to: {self.yawn_mar_threshold:.3f}")

    def _euclidean_dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def calculate_single_ear(self, landmarks_xy, indices):
        p1 = landmarks_xy[indices[0]]
        p2 = landmarks_xy[indices[1]]
        p3 = landmarks_xy[indices[2]]
        p4 = landmarks_xy[indices[3]]
        p5 = landmarks_xy[indices[4]]
        p6 = landmarks_xy[indices[5]]

        v1 = self._euclidean_dist(p2, p6)
        v2 = self._euclidean_dist(p3, p5)
        horiz = self._euclidean_dist(p1, p4)

        if horiz < 1e-6:
            return 0.0

        ear = (v1 + v2) / (2.0 * horiz)
        return float(ear)

    def calculate_mar(self, landmarks_xy):
        left_corner = landmarks_xy[self.MOUTH_LANDMARKS[0]]
        right_corner = landmarks_xy[self.MOUTH_LANDMARKS[1]]
        top_lip = landmarks_xy[self.MOUTH_LANDMARKS[2]]
        bottom_lip = landmarks_xy[self.MOUTH_LANDMARKS[3]]

        horiz = self._euclidean_dist(left_corner, right_corner)
        vert = self._euclidean_dist(top_lip, bottom_lip)

        if horiz < 1e-6:
            return 0.0

        mar = vert / horiz
        return float(mar)

    def update_auto_calibration(self, avg_ear, mar):
        if not self.is_calibrated:
            self.calibration_frames += 1
            self.ear_baseline_samples.append(avg_ear)
            self.mar_baseline_samples.append(mar)

            if self.calibration_frames >= self.calibration_max_frames:
                self.ear_baseline = max(0.28, float(np.percentile(self.ear_baseline_samples, 75)))
                self.mar_baseline = max(0.14, float(np.percentile(self.mar_baseline_samples, 30)))
                self.eye_closed_threshold = round(self.ear_baseline * 0.78, 3)
                self.is_calibrated = True
                print(f"[CVPipeline] Calibration Complete! Open EAR={self.ear_baseline:.3f}, Adaptive Eye Threshold={self.eye_closed_threshold:.3f}")

    def generate_simulated_frame(self, scenario="normal"):
        self.sim_t += 0.1
        t = self.sim_t
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (24, 28, 36)
        
        for y in range(0, 480, 40):
            cv2.line(frame, (0, y), (640, y), (32, 38, 48), 1)
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 480), (32, 38, 48), 1)

        face_center = (320 + int(10 * math.sin(t * 0.5)), 240 + int(5 * math.cos(t * 0.3)))
        
        if scenario == "normal":
            blink_dip = 0.18 if (int(t * 1.5) % 8 == 0) else 0.0
            left_ear = max(0.12, 0.32 - blink_dip + 0.01 * math.sin(t * 2))
            right_ear = max(0.12, 0.32 - blink_dip + 0.01 * math.cos(t * 2))
            mar = 0.14 + 0.03 * math.cos(t * 1.2)

        elif scenario == "mild_fatigue":
            blink_dip = 0.16 if (int(t * 2.0) % 5 == 0) else 0.0
            left_ear = max(0.13, 0.25 - blink_dip + 0.02 * math.sin(t * 3))
            right_ear = max(0.13, 0.25 - blink_dip + 0.02 * math.cos(t * 3))
            mar = 0.18 + 0.05 * math.sin(t * 0.8)

        elif scenario == "yawning":
            left_ear = 0.28 + 0.02 * math.sin(t * 2)
            right_ear = 0.28 + 0.02 * math.cos(t * 2)
            mar = 0.62 if (t % 10.0 > 2.0 and t % 10.0 < 7.0) else 0.18

        elif scenario == "microsleep_drowsy":
            left_ear = 0.13 + 0.01 * math.sin(t)
            right_ear = 0.13 + 0.01 * math.cos(t)
            mar = 0.25 + 0.05 * math.sin(t * 1.5)

        elif scenario == "mobile_use":
            left_ear = 0.31 + 0.01 * math.sin(t * 2)
            right_ear = 0.31 + 0.01 * math.cos(t * 2)
            mar = 0.16 + 0.02 * math.sin(t * 1.5)
            # Render simulated smartphone held by rider
            cv2.rectangle(frame, (430, 180), (510, 320), (30, 30, 35), -1)
            cv2.rectangle(frame, (430, 180), (510, 320), (0, 0, 255), 2)
            cv2.rectangle(frame, (437, 195), (503, 305), (220, 235, 250), -1)
            cv2.putText(frame, "MOBILE", (442, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 2)

        elif scenario == "repeated_closures":
            # Simulate rider repeatedly closing eyes for ~1.4s (smoothed) then reopening for ~1.4s
            cycle = t % 3.4
            eye_closed = (cycle < 2.0)
            eye_dip = 0.20 if eye_closed else 0.0
            left_ear = max(0.12, 0.32 - eye_dip)
            right_ear = max(0.12, 0.32 - eye_dip)
            mar = 0.15 + 0.02 * math.sin(t * 1.2)

        avg_ear = (left_ear + right_ear) / 2.0

        cv2.ellipse(frame, face_center, (110, 140), 0, 0, 360, (180, 200, 220), 2)
        cv2.ellipse(frame, (face_center[0], face_center[1] - 30), (145, 150), 0, 180, 360, (0, 165, 255), 3)
        cv2.putText(frame, "SIMULATED RIDER STREAM", (face_center[0] - 85, face_center[1] - 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        eye_h = int(avg_ear * 100)
        cv2.ellipse(frame, (face_center[0] - 45, face_center[1] - 30), (22, max(2, eye_h)), 0, 0, 360, (0, 255, 200), 2)
        cv2.ellipse(frame, (face_center[0] + 45, face_center[1] - 30), (22, max(2, eye_h)), 0, 0, 360, (0, 255, 200), 2)

        mouth_h = int(mar * 80)
        cv2.ellipse(frame, (face_center[0], face_center[1] + 55), (35, max(4, mouth_h)), 0, 0, 360, (255, 100, 100), 2)

        return frame, left_ear, right_ear, avg_ear, mar

    def process_frame(self, frame=None):
        """
        Processes camera/simulated frame:
        - Calculates Left, Right, Avg EAR and continuous Mouth Aspect Ratio (MAR).
        - Updates continuous eye closure timer & sustained yawn timer.
        - Resets timers cleanly when rider reopens eyes / closes mouth.
        """
        self.frame_count += 1
        now = time.time()
        alarm_sound_trigger = False

        mobile_detected = False
        mobile_detections = []

        if self.use_simulation or frame is None:
            frame, left_ear, right_ear, avg_ear, raw_mar = self.generate_simulated_frame(scenario=self.sim_scenario)
            if self.sim_scenario == "mobile_use":
                mobile_detected = True
                mobile_detections = [{"box": [430, 180, 80, 140], "confidence": 0.96}]
        else:
            # Ensure frame is capped at 640 width for high-performance inference on any laptop
            if frame.shape[1] > 640:
                h_orig, w_orig = frame.shape[:2]
                frame = cv2.resize(frame, (640, int(h_orig * (640.0 / w_orig))), interpolation=cv2.INTER_LINEAR)

            h, w, _ = frame.shape
            left_ear, right_ear, avg_ear, raw_mar = 0.32, 0.32, 0.32, 0.15

            if self.landmarker is not None:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = self.landmarker.detect(mp_image)

                    if result.face_landmarks and len(result.face_landmarks) > 0:
                        face_lms = result.face_landmarks[0]
                        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms]

                        left_ear = self.calculate_single_ear(pts, self.LEFT_EYE_LANDMARKS)
                        right_ear = self.calculate_single_ear(pts, self.RIGHT_EYE_LANDMARKS)
                        avg_ear = (left_ear + right_ear) / 2.0
                        raw_mar = self.calculate_mar(pts)

                        self.update_auto_calibration(avg_ear, raw_mar)

                        # Draw landmarks on frame
                        for idx in self.LEFT_EYE_LANDMARKS:
                            cv2.circle(frame, pts[idx], 2, (0, 255, 255), -1)
                        for idx in self.RIGHT_EYE_LANDMARKS:
                            cv2.circle(frame, pts[idx], 2, (255, 255, 0), -1)
                        for idx in self.MOUTH_LANDMARKS:
                            cv2.circle(frame, pts[idx], 2, (0, 165, 255), -1)

                except Exception as e:
                    print(f"[CVPipeline] Error during landmark processing: {e}")

            # Run YOLO Mobile Detector on camera frame
            if self.mobile_detector is not None:
                mobile_detected, mobile_detections = self.mobile_detector.detect(frame)

        # Draw bounding boxes for detected mobile phones
        if mobile_detected and mobile_detections:
            for det in mobile_detections:
                bx, by, bw_box, bh_box = det["box"]
                conf = det["confidence"]
                cv2.rectangle(frame, (bx, by), (bx + bw_box, by + bh_box), (0, 0, 255), 3)
                cv2.rectangle(frame, (bx, max(0, by - 28)), (bx + bw_box, by), (0, 0, 220), -1)
                cv2.putText(frame, f"CELL PHONE: {int(conf * 100)}%", (bx + 5, max(15, by - 8)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        # Light exponential smoothing ($\alpha = 0.70$)
        alpha = 0.70
        self.smoothed_left_ear = (alpha * left_ear) + ((1.0 - alpha) * self.smoothed_left_ear)
        self.smoothed_right_ear = (alpha * right_ear) + ((1.0 - alpha) * self.smoothed_right_ear)
        self.smoothed_avg_ear = (alpha * avg_ear) + ((1.0 - alpha) * self.smoothed_avg_ear)
        self.smoothed_mar = (alpha * raw_mar) + ((1.0 - alpha) * self.smoothed_mar)

        s_left_ear = float(self.smoothed_left_ear)
        s_right_ear = float(self.smoothed_right_ear)
        s_avg_ear = float(self.smoothed_avg_ear)
        s_mar = float(self.smoothed_mar)

        eye_threshold = float(self.eye_closed_threshold)
        yawn_threshold = float(self.yawn_mar_threshold)

        # -------------------------------------------------------------
        # 1. CONTINUOUS EYE-CLOSURE & SHORT CLOSURE LOGIC
        # -------------------------------------------------------------
        is_eye_closed_now = (s_avg_ear < eye_threshold)

        if is_eye_closed_now:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now
                print(f"[DEBUG] Eye closure started! Left EAR={s_left_ear:.3f}, Right EAR={s_right_ear:.3f}, Avg EAR={s_avg_ear:.3f}, Threshold={eye_threshold:.3f}")

            self.current_closure_duration = round(now - self.eye_closed_start_time, 2)
        else:
            if self.eye_closed_start_time is not None:
                final_dur = round(now - self.eye_closed_start_time, 2)
                print(f"[DEBUG] Eyes reopened after {final_dur:.2f}s. Closure timer reset.")

                # Check for Repeated Short Eye Closure Event (1.0s <= duration < 2.0s)
                if self.short_closure_min <= final_dur < self.short_closure_max:
                    self.short_closure_events.append(now)
                    print(f"[DEBUG] Short eye closure event recorded! Duration={final_dur:.2f}s (Total in window: {len(self.short_closure_events)})")

            self.eye_closed_start_time = None
            self.current_closure_duration = 0.0
            self.eye_alarm_triggered = False

        closure_duration = self.current_closure_duration
        is_eye_closed = (self.eye_closed_start_time is not None)

        # Rolling Window Filter for Repeated Short Eye Closures (Early Sleepiness)
        window_cutoff = now - self.sleepiness_window
        recent_short_closures = [t for t in self.short_closure_events if t >= window_cutoff]
        short_closure_count = len(recent_short_closures)
        sleepy_confirmed = (short_closure_count >= self.sleepiness_event_count)

        # -------------------------------------------------------------
        # 2. TEMPORAL YAWN TIMER LOGIC (MAR > YAWN_MAR_THRESHOLD)
        # -------------------------------------------------------------
        is_mouth_open_now = (s_mar > yawn_threshold)

        if is_mouth_open_now:
            if self.yawn_start_time is None:
                self.yawn_start_time = now
                print(f"[DEBUG] Wide mouth opening started! MAR={s_mar:.3f}, Yawn Threshold={yawn_threshold:.3f}")

            self.current_yawn_duration = round(now - self.yawn_start_time, 2)
        else:
            if self.yawn_start_time is not None:
                final_yawn_dur = round(now - self.yawn_start_time, 2)
                print(f"[DEBUG] Mouth closed after {final_yawn_dur:.2f}s. Yawn timer reset.")
            self.yawn_start_time = None
            self.current_yawn_duration = 0.0
            self.yawn_alarm_triggered = False

        yawn_duration = self.current_yawn_duration
        is_mouth_open = (self.yawn_start_time is not None)
        yawn_confirmed = (yawn_duration >= self.yawn_min_duration)

        # -------------------------------------------------------------
        # 3. ALARM TRIGGER & ACTIONABLE ALERT MESSAGE LOGIC
        # -------------------------------------------------------------
        eye_drowsy_confirmed = (closure_duration >= 2.0)
        alarm_triggered = eye_drowsy_confirmed or sleepy_confirmed or yawn_confirmed or mobile_detected
        alert_message = ""

        if mobile_detected:
            alert_message = "⚠️ DO NOT USE MOBILE WHILE DRIVING"
            alarm_sound_trigger = True
        elif eye_drowsy_confirmed:
            alert_message = "⚠️ DROWSINESS DETECTED — YOU LOOK TIRED. PLEASE DO FACE WASH / TAKE A SHORT BREAK."
            if not self.eye_alarm_triggered:
                print(f"[DEBUG] Eye closure reached {closure_duration:.2f}s! DROWSINESS ALERT + ALARM TRIGGERED.")
                self.eye_alarm_triggered = True
                alarm_sound_trigger = True

        elif sleepy_confirmed:
            alert_message = "⚠️ YOU ARE FEELING SLEEPY — PLEASE DO FACE WASH / TAKE A SHORT BREAK."
            if not self.sleepiness_alarm_triggered:
                print(f"[DEBUG] Repeated short closures reached threshold ({short_closure_count}/{self.sleepiness_event_count})! EARLY SLEEPINESS ALERT + ALARM TRIGGERED.")
                self.sleepiness_alarm_triggered = True
                alarm_sound_trigger = True

        elif yawn_confirmed:
            alert_message = "⚠️ YAWN DETECTED — YOU LOOK TIRED. PLEASE DO FACE WASH / TAKE A SHORT BREAK."
            if not self.yawn_alarm_triggered:
                print(f"[DEBUG] Sustained Yawn reached {yawn_duration:.2f}s! YAWN ALERT + ALARM TRIGGERED.")
                self.yawn_alarm_triggered = True
                alarm_sound_trigger = True

        if not sleepy_confirmed:
            self.sleepiness_alarm_triggered = False

        if is_eye_closed and not self.is_eye_closed_prev:
            self.blink_timestamps.append(now)
        self.is_eye_closed_prev = is_eye_closed

        if yawn_confirmed and not self.is_yawning_prev:
            self.yawn_timestamps.append(now)
        self.is_yawning_prev = yawn_confirmed

        one_min_ago = now - 60.0
        two_min_ago = now - 120.0
        
        recent_blinks = [t for t in self.blink_timestamps if t >= one_min_ago]
        blink_rate = float(len(recent_blinks)) * (60.0 / max(1.0, min(60.0, now - self.start_time)))
        blink_rate = float(np.clip(blink_rate, 0.0, 40.0))

        recent_yawns = [t for t in self.yawn_timestamps if t >= two_min_ago]
        yawn_rate = float(len(recent_yawns))

        prediction = self.classifier.predict(
            ear=s_avg_ear,
            mar=s_mar,
            blink_rate=blink_rate,
            yawn_rate=yawn_rate,
            closure_duration=closure_duration,
            yawn_confirmed=yawn_confirmed,
            mobile_detected=mobile_detected,
            sleepy_confirmed=sleepy_confirmed
        )

        self.logger.log_frame(
            frame_id=self.frame_count,
            ear=s_avg_ear,
            mar=s_mar,
            blink_rate=blink_rate,
            yawn_rate=yawn_rate,
            closure_duration=closure_duration,
            fatigue_score=prediction["fatigue_score"],
            state_code=prediction["state_code"],
            state_name=prediction["state_name"]
        )

        annotated_frame = self.render_hud_overlay(
            frame, s_left_ear, s_right_ear, s_avg_ear, s_mar, 
            blink_rate, closure_duration, yawn_duration, short_closure_count, prediction, alert_message
        )

        return {
            "frame": annotated_frame,
            "left_ear": round(s_left_ear, 3),
            "right_ear": round(s_right_ear, 3),
            "avg_ear": round(s_avg_ear, 3),
            "ear": round(s_avg_ear, 3),
            "ear_threshold": eye_threshold,
            "is_eye_closed": is_eye_closed,
            "closure_duration": closure_duration,
            "short_closure_count": short_closure_count,
            "short_closure_min": self.short_closure_min,
            "short_closure_max": self.short_closure_max,
            "sleepiness_event_count": self.sleepiness_event_count,
            "sleepiness_window": self.sleepiness_window,
            "sleepy_confirmed": sleepy_confirmed,
            "mar": round(s_mar, 3),
            "yawn_threshold": yawn_threshold,
            "is_mouth_open": is_mouth_open,
            "yawn_duration": yawn_duration,
            "yawn_confirmed": yawn_confirmed,
            "mobile_detected": mobile_detected,
            "mobile_confidence": mobile_detections[0]["confidence"] if mobile_detections else 0.0,
            "mobile_detections": mobile_detections,
            "alarm_triggered": alarm_triggered,
            "alarm_sound_trigger": alarm_sound_trigger,
            "alert_message": alert_message,
            "blink_rate": round(blink_rate, 1),
            "yawn_rate": round(yawn_rate, 1),
            "prediction": prediction,
            "frame_id": self.frame_count,
            "is_calibrated": self.is_calibrated
        }

    def render_hud_overlay(self, frame, left_ear, right_ear, avg_ear, mar, blink_rate, closure_duration, yawn_duration, short_closure_count, prediction, alert_message):
        h, w, _ = frame.shape
        
        state_colors = {
            0: (0, 220, 100),
            1: (0, 215, 255),
            2: (0, 120, 255),
            3: (0, 0, 255),
            4: (0, 0, 255),
            5: (0, 140, 255)
        }
        color = state_colors.get(prediction["state_code"], (255, 255, 255))

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (15, 20, 28), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (10, 10), (w - 10, 120), color, 2)

        cv2.putText(frame, "RIDER VIGILANCE HUD", (25, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 240), 1, cv2.LINE_AA)
        
        state_str = f"STATE: {prediction['state_name'].upper()}"
        cv2.putText(frame, state_str, (w - 360, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)

        thresh_str = f"EAR THRESH: < {self.eye_closed_threshold:.3f} | MAR THRESH: > {self.yawn_mar_threshold:.2f}"
        cv2.putText(frame, thresh_str, (25, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        metrics_str = f"EAR: {avg_ear:.3f} (L:{left_ear:.3f}, R:{right_ear:.3f}) | MAR: {mar:.3f} | Eye Closure: {closure_duration:.1f}s | Yawn Dur: {yawn_duration:.1f}s"
        cv2.putText(frame, metrics_str, (25, 78), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 240, 255), 1, cv2.LINE_AA)

        score = prediction["fatigue_score"]
        bar_x, bar_y, bar_w, bar_h = 25, 93, 250, 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 60, 75), -1)
        fill_w = int(bar_w * score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"Fatigue Score: {score:.2f}", (bar_x + bar_w + 12, bar_y + 11), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 240), 1)

        if alert_message:
            flash_alpha = (math.sin(time.time() * 8) + 1.0) / 2.0
            warning_overlay = frame.copy()
            cv2.rectangle(warning_overlay, (0, h - 80), (w, h), (0, 0, 220), -1)
            cv2.addWeighted(warning_overlay, 0.80 * flash_alpha, frame, 1.0 - 0.80 * flash_alpha, 0, frame)
            
            cv2.putText(frame, alert_message, (20, h - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
