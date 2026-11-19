import os
import time
import cv2
import threading
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from core.cv_pipeline import ComputerVisionPipeline
from core.camera_manager import CameraManager
from train_model import train_and_save_model

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global CV Pipeline Instance & Non-blocking Camera Manager
pipeline = ComputerVisionPipeline(use_simulation=True)
camera_manager = CameraManager(camera_index=0)

frame_lock = threading.Lock()
latest_telemetry = {}

def update_telemetry(frame=None):
    global latest_telemetry
    res = pipeline.process_frame(frame)
    latest_telemetry = {
        # Eye Debugging Metrics
        "left_ear": res["left_ear"],
        "right_ear": res["right_ear"],
        "avg_ear": res["avg_ear"],
        "ear": res["avg_ear"],
        "ear_threshold": res["ear_threshold"],
        "is_eye_closed": res["is_eye_closed"],
        "closure_duration": res["closure_duration"],
        "short_closure_count": res.get("short_closure_count", 0),
        "sleepiness_event_count": res.get("sleepiness_event_count", 3),
        "sleepy_confirmed": res.get("sleepy_confirmed", False),
        # Mouth Debugging Metrics
        "mar": res["mar"],
        "yawn_threshold": res["yawn_threshold"],
        "is_mouth_open": res["is_mouth_open"],
        "yawn_duration": res["yawn_duration"],
        "yawn_confirmed": res["yawn_confirmed"],
        # YOLO Mobile Detection Metrics
        "mobile_detected": res.get("mobile_detected", False),
        "mobile_confidence": res.get("mobile_confidence", 0.0),
        "mobile_detections": res.get("mobile_detections", []),
        # Alarm & Actionable Message State
        "alarm_triggered": res["alarm_triggered"],
        "alarm_sound_trigger": res["alarm_sound_trigger"],
        "alert_message": res["alert_message"],
        # Classifier & Performance Metrics
        "blink_rate": res["blink_rate"],
        "yawn_rate": res["yawn_rate"],
        "fatigue_score": res["prediction"]["fatigue_score"],
        "state_code": res["prediction"]["state_code"],
        "state_name": res["prediction"]["state_name"],
        "confidence": res["prediction"]["confidence"],
        "flags": res["prediction"]["flags"],
        "frame_id": res["frame_id"],
        "use_simulation": pipeline.use_simulation,
        "sim_scenario": pipeline.sim_scenario,
        "is_calibrated": res["is_calibrated"]
    }
    return res

def generate_video_stream():
    """Generates MJPEG video stream frames for web feed at smooth 30 FPS."""
    while True:
        t_start = time.time()
        with frame_lock:
            if not pipeline.use_simulation:
                if not camera_manager.is_running:
                    camera_manager.start()
                frame = camera_manager.get_frame()
                res = update_telemetry(frame)
            else:
                if camera_manager.is_running:
                    camera_manager.stop()
                res = update_telemetry(None)
            
            # Encode frame to JPEG with fast compression
            ret, buffer = cv2.imencode('.jpg', res["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Precise pacing for smooth 30 FPS playback
        elapsed = time.time() - t_start
        sleep_time = max(0.005, 0.033 - elapsed)
        time.sleep(sleep_time)

@app.route('/')
def index():
    """Renders main Web Dashboard UI."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """MJPEG Video streaming route."""
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def api_telemetry():
    """Returns current real-time telemetry metrics JSON."""
    if not latest_telemetry:
        with frame_lock:
            frame = camera_manager.get_frame() if not pipeline.use_simulation else None
            update_telemetry(frame)
    return jsonify(latest_telemetry)

@app.route('/api/set_scenario', methods=['POST'])
def set_scenario():
    """Sets simulation scenario: normal, mild_fatigue, yawning, microsleep_drowsy."""
    data = request.json or {}
    scenario = data.get('scenario', 'normal')
    pipeline.sim_scenario = scenario
    pipeline.use_simulation = True
    if camera_manager.is_running:
        camera_manager.stop()
    with frame_lock:
        update_telemetry(None)
    return jsonify({"success": True, "scenario": scenario, "use_simulation": True})

@app.route('/api/toggle_source', methods=['POST'])
def toggle_source():
    """Toggles between simulated rider video stream and live webcam feed."""
    pipeline.use_simulation = not pipeline.use_simulation
    if not pipeline.use_simulation:
        camera_manager.start()
    else:
        camera_manager.stop()
    
    with frame_lock:
        frame = camera_manager.get_frame() if not pipeline.use_simulation else None
        update_telemetry(frame)
        
    return jsonify({
        "success": True, 
        "use_simulation": pipeline.use_simulation,
        "camera_running": camera_manager.is_running
    })

@app.route('/api/set_threshold', methods=['POST'])
def set_threshold():
    """Configures eye closed EAR threshold dynamically."""
    data = request.json or {}
    val = data.get('threshold', 0.26)
    pipeline.set_eye_closed_threshold(val)
    return jsonify({"success": True, "threshold": pipeline.eye_closed_threshold})

@app.route('/api/set_yawn_threshold', methods=['POST'])
def set_yawn_threshold():
    """Configures Yawn MAR threshold dynamically."""
    data = request.json or {}
    val = data.get('threshold', 0.50)
    pipeline.set_yawn_mar_threshold(val)
    return jsonify({"success": True, "threshold": pipeline.yawn_mar_threshold})

@app.route('/api/summary')
def api_summary():
    """Returns active session summary metrics."""
    summary = pipeline.logger.get_session_summary()
    return jsonify(summary)

@app.route('/api/logs')
def api_logs():
    """Returns recent CSV log entries."""
    rows = pipeline.logger.get_latest_rows(max_rows=100)
    files = pipeline.logger.list_session_files()
    return jsonify({
        "current_log": pipeline.logger.current_filename,
        "total_entries": len(rows),
        "recent_rows": rows,
        "session_files": files
    })

@app.route('/api/download_log/<filename>')
def download_log(filename):
    """Downloads a CSV session log file."""
    log_dir = os.path.abspath(pipeline.logger.log_dir)
    return send_from_directory(log_dir, filename, as_attachment=True)

@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    """Triggers ML model re-training and reloads model in classifier."""
    try:
        train_and_save_model()
        pipeline.classifier.load_model()
        return jsonify({"success": True, "message": "Model re-trained and reloaded successfully!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
