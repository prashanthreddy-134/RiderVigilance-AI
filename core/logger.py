import os
import csv
import time
from datetime import datetime
from collections import deque

class SessionLogger:
    """
    Lightweight Buffered CSV Event Logger for Rider Fatigue & Drowsiness Telemetry.
    Saves compact feature vectors with memory caching to eliminate disk I/O bottlenecks.
    """
    CSV_HEADERS = [
        "timestamp",
        "iso_time",
        "frame_id",
        "ear",
        "mar",
        "blink_rate",
        "yawn_rate",
        "closure_duration",
        "fatigue_score",
        "state_code",
        "state_name"
    ]

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_filename = None
        self.current_filepath = None
        self.recent_rows_cache = deque(maxlen=200)
        self.pending_rows = []
        self.last_flush_time = time.time()
        self.start_new_session()

    def start_new_session(self):
        self.flush()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = f"session_{session_id}.csv"
        self.current_filepath = os.path.join(self.log_dir, self.current_filename)
        self.recent_rows_cache.clear()
        self.pending_rows.clear()
        
        # Write CSV Header
        with open(self.current_filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADERS)
            
        print(f"[SessionLogger] Started new CSV log session: {self.current_filepath}")

    def flush(self):
        """Flushes buffered rows to disk in a single batched operation."""
        if not self.pending_rows or not self.current_filepath:
            return
        try:
            with open(self.current_filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.pending_rows)
            self.pending_rows.clear()
            self.last_flush_time = time.time()
        except Exception as e:
            print(f"[SessionLogger] Error flushing log rows: {e}")

    def log_frame(self, frame_id: int, ear: float, mar: float, blink_rate: float, 
                  yawn_rate: float, closure_duration: float, fatigue_score: float, 
                  state_code: int, state_name: str):
        """
        Appends a lightweight feature vector entry into memory cache and batch flushes.
        """
        now = time.time()
        iso_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        row = [
            round(now, 3),
            iso_str,
            frame_id,
            round(ear, 4),
            round(mar, 4),
            round(blink_rate, 1),
            round(yawn_rate, 1),
            round(closure_duration, 2),
            round(fatigue_score, 3),
            state_code,
            state_name
        ]

        row_dict = {
            "timestamp": str(round(now, 3)),
            "iso_time": iso_str,
            "frame_id": str(frame_id),
            "ear": str(round(ear, 4)),
            "mar": str(round(mar, 4)),
            "blink_rate": str(round(blink_rate, 1)),
            "yawn_rate": str(round(yawn_rate, 1)),
            "closure_duration": str(round(closure_duration, 2)),
            "fatigue_score": str(round(fatigue_score, 3)),
            "state_code": str(state_code),
            "state_name": state_name
        }

        self.recent_rows_cache.append(row_dict)
        self.pending_rows.append(row)

        # Batch write to disk every 20 frames or every 1.5 seconds
        if len(self.pending_rows) >= 20 or (now - self.last_flush_time >= 1.5):
            self.flush()

    def get_latest_rows(self, max_rows=50):
        """Retrieves latest rows from fast memory cache."""
        if self.recent_rows_cache:
            return list(self.recent_rows_cache)[-max_rows:]

        if not self.current_filepath or not os.path.exists(self.current_filepath):
            return []
        
        rows = []
        try:
            with open(self.current_filepath, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            print(f"[SessionLogger] Error reading log: {e}")
            return []
        
        return rows[-max_rows:]

    def get_session_summary(self):
        """Calculates aggregated summary statistics for the active session."""
        rows = self.get_latest_rows(max_rows=5000)
        if not rows:
            return {
                "total_frames": 0,
                "duration_seconds": 0,
                "avg_ear": 0.0,
                "avg_mar": 0.0,
                "avg_score": 0.0,
                "state_distribution": {"Alert (Focused)": 0, "Mild Fatigue": 0, "Severe Fatigue": 0, "Drowsy": 0},
                "log_file_size_kb": 0.0
            }

        total_frames = len(rows)
        first_time = float(rows[0]["timestamp"])
        last_time = float(rows[-1]["timestamp"])
        duration = max(1.0, last_time - first_time)
        
        ears = [float(r["ear"]) for r in rows]
        mars = [float(r["mar"]) for r in rows]
        scores = [float(r["fatigue_score"]) for r in rows]
        
        state_counts = {"Alert (Focused)": 0, "Mild Fatigue": 0, "Severe Fatigue": 0, "Drowsy": 0}
        for r in rows:
            s_name = r["state_name"]
            state_counts[s_name] = state_counts.get(s_name, 0) + 1

        file_size_kb = os.path.getsize(self.current_filepath) / 1024.0 if os.path.exists(self.current_filepath) else 0.0

        return {
            "total_frames": total_frames,
            "duration_seconds": round(duration, 1),
            "avg_ear": round(sum(ears) / total_frames, 4),
            "avg_mar": round(sum(mars) / total_frames, 4),
            "avg_score": round(sum(scores) / total_frames, 3),
            "state_distribution": state_counts,
            "log_file_size_kb": round(file_size_kb, 2)
        }

    def list_session_files(self):
        """Lists all CSV session logs stored in log directory."""
        if not os.path.exists(self.log_dir):
            return []
        files = [f for f in os.listdir(self.log_dir) if f.endswith(".csv")]
        files.sort(reverse=True)
        return files
