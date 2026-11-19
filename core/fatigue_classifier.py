import os
import joblib
import numpy as np

class FatigueClassifier:
    """
    Rider Fatigue & Drowsiness Classifier using Random Forest & Weighted Ensemble Logic.
    
    4 Classification States:
      0: Alert (Focused)
      1: Mild Fatigue
      2: YAWN DETECTED — SEVERE FATIGUE
      3: DROWSY — EYES CLOSED
    """
    STATE_LABELS = {
        0: "Alert (Focused)",
        1: "Mild Fatigue",
        2: "Yawn Detected (Severe)",
        3: "DROWSY — EYES CLOSED",
        4: "MOBILE IN USE — DANGER",
        5: "FEELING SLEEPY (REPEATED CLOSURES)"
    }

    def __init__(self, model_path="models/random_forest_model.pkl", scaler_path="models/scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print(f"[FatigueClassifier] Successfully loaded model from {self.model_path}")
            except Exception as e:
                print(f"[FatigueClassifier] Failed loading model ({e}), falling back to weighted ensemble logic.")
                self.model = None
                self.scaler = None

    def compute_weighted_score(self, eye_closed_flag: int, yawn_flag: int, low_blink_flag: int) -> float:
        score = (0.5 * eye_closed_flag) + (0.3 * yawn_flag) + (0.2 * low_blink_flag)
        return float(np.clip(score, 0.0, 1.0))

    def predict(self, ear: float, mar: float, blink_rate: float, yawn_rate: float, closure_duration: float, 
                yawn_confirmed: bool = False, mobile_detected: bool = False, sleepy_confirmed: bool = False):
        """
        Predicts rider fatigue state and calculates fatigue score.
        Priority Overrides:
          1. Mobile usage detected -> MOBILE IN USE — DANGER (State 4, Score 0.90)
          2. Continuous eye closure >= 2.0s -> DROWSY — EYES CLOSED (State 3, Score 1.0)
          3. Sleepiness from 3-4 short closures (1-2s) -> FEELING SLEEPY (State 5, Score 0.80)
          4. Confirmed Yawn (sustained MAR >= 1.5s) -> YAWN DETECTED (State 2, Score 0.75)
        """
        eye_closed_flag = 1 if (ear < 0.26 or closure_duration >= 1.0) else 0
        yawn_flag = 1 if (mar > 0.50 or yawn_confirmed) else 0
        low_blink_flag = 1 if (blink_rate < 10 or blink_rate > 26) else 0

        fatigue_score = self.compute_weighted_score(eye_closed_flag, yawn_flag, low_blink_flag)

        # Priority 1: Mobile Phone Usage Detected
        if mobile_detected:
            return {
                "state_code": 4,
                "state_name": "MOBILE IN USE — DANGER",
                "fatigue_score": max(fatigue_score, 0.90),
                "confidence": 0.98,
                "flags": {
                    "eye_closed": eye_closed_flag,
                    "yawn": yawn_flag,
                    "low_blink": low_blink_flag,
                    "mobile_detected": True,
                    "sleepy_confirmed": False,
                    "microsleep": False
                }
            }

        # Priority 2: Continuous Eye Closure >= 2.0s
        if closure_duration >= 2.0:
            return {
                "state_code": 3,
                "state_name": "DROWSY — EYES CLOSED",
                "fatigue_score": 1.0,
                "confidence": 0.99,
                "flags": {
                    "eye_closed": 1,
                    "yawn": yawn_flag,
                    "low_blink": low_blink_flag,
                    "mobile_detected": False,
                    "sleepy_confirmed": sleepy_confirmed,
                    "microsleep": True
                }
            }

        # Priority 3: Confirmed Early Sleepiness (3-4 Repeated Short Eye Closures 1.0s-2.0s)
        if sleepy_confirmed:
            return {
                "state_code": 5,
                "state_name": "FEELING SLEEPY (REPEATED CLOSURES)",
                "fatigue_score": max(fatigue_score, 0.80),
                "confidence": 0.96,
                "flags": {
                    "eye_closed": eye_closed_flag,
                    "yawn": yawn_flag,
                    "low_blink": low_blink_flag,
                    "mobile_detected": False,
                    "sleepy_confirmed": True,
                    "microsleep": False
                }
            }

        # Priority 2: Confirmed Temporal Yawn
        if yawn_confirmed:
            return {
                "state_code": 2,
                "state_name": "YAWN DETECTED — SEVERE FATIGUE",
                "fatigue_score": max(fatigue_score, 0.75),
                "confidence": 0.95,
                "flags": {
                    "eye_closed": eye_closed_flag,
                    "yawn": 1,
                    "low_blink": low_blink_flag,
                    "microsleep": False
                }
            }

        # ML Random Forest Prediction (optimized single-pass inference)
        if self.model is not None and self.scaler is not None:
            try:
                features = np.array([[ear, mar, blink_rate, yawn_rate, closure_duration]])
                features_scaled = self.scaler.transform(features)
                probs = self.model.predict_proba(features_scaled)[0]
                state_code = int(np.argmax(probs))
                confidence = float(probs[state_code])
                
                state_name = self.STATE_LABELS.get(state_code, "Alert (Focused)")

                return {
                    "state_code": state_code,
                    "state_name": state_name,
                    "fatigue_score": round(fatigue_score, 3),
                    "confidence": round(confidence, 3),
                    "flags": {
                        "eye_closed": eye_closed_flag,
                        "yawn": yawn_flag,
                        "low_blink": low_blink_flag,
                        "microsleep": False
                    }
                }
            except Exception:
                pass

        # Rule-based Ensemble Fallback
        if fatigue_score >= 0.65 or (eye_closed_flag and yawn_flag):
            state_code = 3
        elif fatigue_score >= 0.45 or yawn_flag:
            state_code = 2
        elif fatigue_score >= 0.25 or low_blink_flag:
            state_code = 1
        else:
            state_code = 0

        state_name = self.STATE_LABELS.get(state_code, "Alert (Focused)")
        confidence = 0.88 + (0.05 if state_code == 0 or state_code == 3 else 0.0)

        return {
            "state_code": state_code,
            "state_name": state_name,
            "fatigue_score": round(fatigue_score, 3),
            "confidence": round(confidence, 3),
            "flags": {
                "eye_closed": eye_closed_flag,
                "yawn": yawn_flag,
                "low_blink": low_blink_flag,
                "microsleep": False
            }
        }
