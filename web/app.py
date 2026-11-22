import os
import cv2
import time
import logging
import threading
import atexit
import numpy as np

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from core.cv_pipeline import ComputerVisionPipeline
from core.camera_manager import CameraManager


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "5000"))

# Set HOSTED_BROWSER_CAMERA=1 on Render.
# A hosted server cannot access the visitor's physical webcam.
HOSTED_BROWSER_CAMERA = (
    os.getenv("HOSTED_BROWSER_CAMERA", "0").strip().lower()
    in {"1", "true", "yes"}
)

MAX_FRAME_WIDTH = 640
JPEG_QUALITY = 78
LOCAL_FRAME_DELAY = 0.025

ALLOWED_SCENARIOS = {
    "normal",
    "mild_fatigue",
    "yawning",
    "microsleep_drowsy",
    "repeated_closures",
    "mobile_use",
}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RiderVigilance")


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "web", "templates"),
    static_folder=os.path.join(BASE_DIR, "web", "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


# ============================================================
# INITIALIZE THE EXISTING PROJECT COMPONENTS
# ============================================================

pipeline = ComputerVisionPipeline()

camera_manager = None
if not HOSTED_BROWSER_CAMERA:
    try:
        camera_manager = CameraManager()
    except Exception:
        logger.exception("Could not initialize CameraManager")

state_lock = threading.RLock()
frame_lock = threading.RLock()

latest_frame_jpeg = None
latest_telemetry = {
    "status": "INITIALIZING",
    "state_name": "INITIALIZING",
    "detection_status": "INITIALIZING",
    "use_simulation": False,
    "sim_scenario": "normal",
    "face_detected": False,
    "ear": 0.0,
    "left_ear": 0.0,
    "right_ear": 0.0,
    "mar": 0.0,
    "fatigue_score": 0.0,
    "phone_detected": False,
    "yawn_detected": False,
    "eye_closed": False,
    "closure_duration": 0.0,
    "yawn_duration": 0.0,
    "blink_rate": 0.0,
    "yawn_rate": 0.0,
    "alarm_triggered": False,
    "alarm_sound_trigger": False,
    "alert_message": "",
    "fps": 0.0,
    "timestamp": time.time(),
}

active_scenario = "normal"
active_source = "browser" if HOSTED_BROWSER_CAMERA else "local"
last_processed_at = 0.0
smoothed_fps = 0.0

worker_started = False
worker_lock = threading.Lock()
stop_event = threading.Event()


# ============================================================
# FRAME HELPERS
# ============================================================

def resize_frame(frame):
    if frame is None:
        return None

    height, width = frame.shape[:2]
    if width <= MAX_FRAME_WIDTH:
        return frame

    new_height = int(height * MAX_FRAME_WIDTH / width)
    return cv2.resize(
        frame,
        (MAX_FRAME_WIDTH, new_height),
        interpolation=cv2.INTER_AREA,
    )


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def process_frame(frame):
    """
    Send one frame through the real ComputerVisionPipeline.

    The supplied pipeline returns a dictionary. The annotated image
    is in result["frame"]; metrics are in the remaining dictionary keys.
    """
    global latest_frame_jpeg, latest_telemetry
    global last_processed_at, smoothed_fps

    if frame is not None:
        frame = resize_frame(frame)

    try:
        result = pipeline.process_frame(frame)

        if not isinstance(result, dict):
            logger.error(
                "ComputerVisionPipeline.process_frame returned %s, expected dict",
                type(result).__name__,
            )
            return False

        annotated = result.get("frame")
        if annotated is None:
            annotated = frame

        if annotated is None:
            logger.warning("Pipeline did not return an image frame.")
            return False

        prediction = result.get("prediction") or {}
        if not isinstance(prediction, dict):
            prediction = {}

        now = time.time()
        if last_processed_at:
            elapsed = now - last_processed_at
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                smoothed_fps = (
                    current_fps
                    if smoothed_fps == 0
                    else (0.8 * smoothed_fps + 0.2 * current_fps)
                )
        last_processed_at = now

        state_name = prediction.get(
            "state_name", result.get("detection_status", "MONITORING")
        )
        telemetry = {
            # Canonical names plus compatibility aliases used by the dashboard JS.
            "status": state_name,
            "state_name": state_name,
            "use_simulation": bool(pipeline.use_simulation),
            "sim_scenario": getattr(pipeline, "sim_scenario", "normal"),
            "avg_ear": _number(result.get("avg_ear", result.get("ear", 0))),
            "is_eye_closed": bool(result.get("is_eye_closed", False)),
            "is_mouth_open": bool(result.get("is_mouth_open", False)),
            "yawn_confirmed": bool(result.get("yawn_confirmed", False)),
            "sleepy_confirmed": bool(result.get("sleepy_confirmed", False)),
            "short_closure_count": result.get("short_closure_count", 0),
            "sleepiness_event_count": result.get("sleepiness_event_count", 3),
            "mobile_detected": bool(result.get("mobile_detected", False)),
            "mobile_confidence": _number(result.get("mobile_confidence", 0)),
            "mobile_detections": result.get("mobile_detections", []),
            "detection_status": result.get(
                "detection_status", "UNKNOWN"
            ),
            "face_detected": bool(result.get("face_detected", False)),
            "ear": _number(result.get("avg_ear", result.get("ear", 0))),
            "left_ear": _number(result.get("left_ear", 0)),
            "right_ear": _number(result.get("right_ear", 0)),
            "mar": _number(result.get("mar", 0)),
            "ear_threshold": _number(result.get("ear_threshold", 0)),
            "yawn_threshold": _number(result.get("yawn_threshold", 0)),
            "fatigue_score": _number(
                prediction.get("fatigue_score", 0)
            ),
            "confidence": _number(prediction.get("confidence", 0)),
            "state_code": prediction.get("state_code", 0),
            "flags": prediction.get("flags", []),
            "closure_duration": _number(
                result.get("closure_duration", 0)
            ),
            "yawn_duration": _number(result.get("yawn_duration", 0)),
            "blink_rate": _number(result.get("blink_rate", 0)),
            "yawn_rate": _number(result.get("yawn_rate", 0)),
            "short_closure_count": result.get("short_closure_count", 0),
            "sleepy_confirmed": bool(
                result.get("sleepy_confirmed", False)
            ),
            "alarm_triggered": bool(
                result.get("alarm_triggered", False)
            ),
            "alarm_sound_trigger": bool(
                result.get("alarm_sound_trigger", False)
            ),
            "alert_message": result.get("alert_message", ""),
            "is_calibrated": bool(result.get("is_calibrated", False)),
            "fps": round(smoothed_fps, 2),
            "timestamp": now,
        }

        ok, encoded = cv2.imencode(
            ".jpg",
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            logger.error("Could not encode processed frame as JPEG.")
            return False

        with frame_lock:
            latest_frame_jpeg = encoded.tobytes()
            latest_telemetry = telemetry

        return True

    except Exception:
        logger.exception("Error while processing camera frame")
        return False


# ============================================================
# LOCAL WEBCAM WORKER
# ============================================================

def local_camera_worker():
    """
    Local mode: CameraManager captures webcam frames in its own thread.
    This worker continuously passes those frames into the CV pipeline.
    """
    if camera_manager is None:
        logger.error("Local CameraManager is unavailable.")
        return

    try:
        camera_manager.start()

        if not camera_manager.is_running:
            logger.error(
                "Webcam failed to start: %s",
                camera_manager.last_error or "No camera frame received",
            )
            return

        logger.info("Local webcam processing worker started.")

        while not stop_event.is_set():
            with state_lock:
                source = active_source

            if source == "simulation":
                # process_frame(None) makes the existing pipeline generate
                # the selected simulated scenario.
                process_frame(None)
            else:
                # Read the latest captured frame atomically. get_frame()
                # returns a placeholder when no real frame has arrived.
                with camera_manager.lock:
                    has_real_frame = camera_manager.latest_frame is not None
                    frame = (
                        camera_manager.latest_frame.copy()
                        if has_real_frame else None
                    )
                if frame is not None:
                    process_frame(frame)
                elif camera_manager.last_error:
                    logger.warning("Camera capture status: %s", camera_manager.last_error)

            time.sleep(LOCAL_FRAME_DELAY)

    except Exception:
        logger.exception("Local camera worker stopped unexpectedly.")
    finally:
        logger.info("Local camera worker exited.")


def start_local_worker():
    global worker_started

    if HOSTED_BROWSER_CAMERA:
        return

    with worker_lock:
        if worker_started:
            return

        worker_started = True
        thread = threading.Thread(
            target=local_camera_worker,
            name="RiderVigilanceCameraWorker",
            daemon=True,
        )
        thread.start()


def shutdown_camera():
    stop_event.set()

    if camera_manager is not None:
        try:
            camera_manager.stop()
        except Exception:
            logger.exception("Error stopping camera manager")

    detector = getattr(pipeline, "mobile_detector", None)
    if detector is not None and hasattr(detector, "stop"):
        try:
            detector.stop()
        except Exception:
            logger.exception("Error stopping mobile detector")


atexit.register(shutdown_camera)


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_video_stream():
    while True:
        with frame_lock:
            frame_bytes = latest_frame_jpeg

        if frame_bytes is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (20, 24, 32)
            cv2.putText(
                placeholder,
                "WAITING FOR CAMERA FRAME...",
                (115, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )
            ok, encoded = cv2.imencode(".jpg", placeholder)
            if ok:
                frame_bytes = encoded.tobytes()

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        time.sleep(0.05)


# ============================================================
# PAGE AND HEALTH ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "hosted_browser_camera": HOSTED_BROWSER_CAMERA,
        "source": active_source,
        "camera_running": bool(camera_manager and camera_manager.is_running),
        "camera_has_frame": bool(
            camera_manager is not None and camera_manager.latest_frame is not None
        ),
        "camera_error": camera_manager.last_error if camera_manager is not None else None,
        "last_frame_timestamp": latest_telemetry.get("timestamp"),
        "timestamp": time.time(),
    })


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ============================================================
# BROWSER WEBCAM UPLOAD (FOR RENDER)
# ============================================================

@app.route("/api/frame", methods=["POST"])
def receive_browser_frame():
    if not HOSTED_BROWSER_CAMERA:
        return jsonify({
            "success": False,
            "error": "Browser camera upload is disabled on this server.",
        }), 400

    raw = request.get_data(cache=False)
    if not raw:
        return jsonify({"success": False, "error": "Empty frame."}), 400

    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        return jsonify({
            "success": False,
            "error": "Could not decode uploaded image.",
        }), 400

    with state_lock:
        source = active_source

    if source == "simulation":
        success = process_frame(None)
    else:
        success = process_frame(image)

    return jsonify({
        "success": success,
        "timestamp": time.time(),
    }), (200 if success else 500)


# ============================================================
# TELEMETRY
# ============================================================

@app.route("/api/telemetry")
def telemetry():
    with frame_lock:
        data = dict(latest_telemetry)

    data.update({
        "hosted_browser_camera": HOSTED_BROWSER_CAMERA,
        "source": active_source,
        "scenario": active_scenario,
        "use_simulation": active_source == "simulation",
        "sim_scenario": active_scenario,
        "state_name": data.get("state_name") or data.get("status", "INITIALIZING"),
        "avg_ear": data.get("avg_ear", data.get("ear", 0.0)),
        "is_eye_closed": data.get("is_eye_closed", data.get("eye_closed", False)),
        "is_mouth_open": data.get("is_mouth_open", False),
        "yawn_confirmed": data.get("yawn_confirmed", data.get("yawn_detected", False)),
        "sleepy_confirmed": data.get("sleepy_confirmed", False),
        "mobile_detected": data.get("mobile_detected", data.get("phone_detected", False)),
    })

    return jsonify(data)


# ============================================================
# SCENARIO AND SOURCE CONTROLS
# ============================================================

@app.route("/api/set_scenario", methods=["POST"])
def set_scenario():
    global active_scenario, active_source

    data = request.get_json(silent=True) or {}
    scenario = data.get("scenario", "normal")

    if scenario not in ALLOWED_SCENARIOS:
        return jsonify({
            "success": False,
            "error": "Invalid scenario.",
        }), 400

    with state_lock:
        active_scenario = scenario
        active_source = "simulation"
        pipeline.sim_scenario = scenario
        pipeline.use_simulation = True

    return jsonify({
        "success": True,
        "scenario": active_scenario,
        "source": active_source,
    })


@app.route("/api/toggle_source", methods=["POST"])
def toggle_source():
    global active_source

    with state_lock:
        if active_source == "simulation":
            active_source = "browser" if HOSTED_BROWSER_CAMERA else "local"
            pipeline.use_simulation = False
        else:
            active_source = "simulation"
            pipeline.use_simulation = True

        source = active_source

    return jsonify({
        "success": True,
        "source": source,
        "browser_camera_required": (
            HOSTED_BROWSER_CAMERA and source == "browser"
        ),
        "hosted_browser_camera": HOSTED_BROWSER_CAMERA,
    })


# ============================================================
# THRESHOLD CONTROLS
# ============================================================

@app.route("/api/set_threshold", methods=["POST"])
def set_threshold():
    data = request.get_json(silent=True) or {}
    value = data.get("threshold")

    try:
        value = float(value)
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "Threshold must be numeric.",
        }), 400

    if not 0.12 <= value <= 0.35:
        return jsonify({
            "success": False,
            "error": "Threshold must be between 0.12 and 0.35.",
        }), 400

    pipeline.set_eye_closed_threshold(value)

    return jsonify({
        "success": True,
        "threshold": pipeline.eye_closed_threshold,
    })


@app.route("/api/set_yawn_threshold", methods=["POST"])
def set_yawn_threshold():
    data = request.get_json(silent=True) or {}
    value = data.get("threshold")

    try:
        value = float(value)
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "Threshold must be numeric.",
        }), 400

    if not 0.35 <= value <= 0.75:
        return jsonify({
            "success": False,
            "error": "Threshold must be between 0.35 and 0.75.",
        }), 400

    pipeline.set_yawn_mar_threshold(value)

    return jsonify({
        "success": True,
        "threshold": pipeline.yawn_mar_threshold,
    })


# ============================================================
# SUMMARY AND LOG FILES
# ============================================================

@app.route("/api/summary")
def summary():
    session_logger = getattr(pipeline, "logger", None)

    if session_logger is not None:
        for method_name in ("get_summary", "summary"):
            method = getattr(session_logger, method_name, None)
            if callable(method):
                try:
                    return jsonify(method())
                except Exception:
                    logger.exception("Could not create session summary.")

    with frame_lock:
        data = dict(latest_telemetry)

    return jsonify({
        "status": "ok",
        "latest_telemetry": data,
        "message": "Session summary method is not provided by SessionLogger.",
    })


@app.route("/api/logs")
def logs():
    import csv

    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    files = [
        name for name in os.listdir(log_dir)
        if name.lower().endswith(".csv")
        and os.path.isfile(os.path.join(log_dir, name))
    ]
    files.sort(key=lambda name: os.path.getmtime(os.path.join(log_dir, name)), reverse=True)
    current_log = files[0] if files else None
    recent_rows = []

    if current_log:
        path = os.path.join(log_dir, current_log)
        try:
            with open(path, "r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                recent_rows = list(reader)[-100:]
        except Exception:
            logger.exception("Could not read current CSV session log.")

    return jsonify({
        "current_log": current_log,
        "recent_rows": recent_rows,
        "files": [
            {
                "filename": name,
                "size": os.path.getsize(os.path.join(log_dir, name)),
                "modified": os.path.getmtime(os.path.join(log_dir, name)),
            }
            for name in files
        ],
    })


@app.route("/api/download_log/<path:filename>")
def download_log(filename):
    log_dir = os.path.join(BASE_DIR, "logs")
    return send_from_directory(
        log_dir,
        filename,
        as_attachment=True,
    )


# ============================================================
# MODEL RETRAINING
# ============================================================

@app.route("/api/retrain", methods=["POST"])
def retrain():
    try:
        from train_model import train_and_save_model

        train_and_save_model()

        return jsonify({
            "success": True,
            "message": "Model retraining completed.",
        })

    except Exception as exc:
        logger.exception("Model retraining failed.")
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "success": False,
        "error": "Frame upload exceeds the 2 MB limit.",
    }), 413


@app.errorhandler(404)
def route_not_found(_error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found.",
    }), 404


@app.errorhandler(500)
def internal_server_error(_error):
    logger.exception("Unhandled server error.")
    return jsonify({
        "success": False,
        "error": "Internal server error.",
    }), 500


# ============================================================
# STARTUP
# ============================================================

start_local_worker()

if __name__ == "__main__":
    logger.info("Starting RiderVigilance AI")
    logger.info("Hosted browser camera mode: %s", HOSTED_BROWSER_CAMERA)

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
