import os
from train_model import train_and_save_model
from web.app import app


def initialize_system():
    # Ensure required directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Check if trained model exists
    model_path = "models/random_forest_model.pkl"
    scaler_path = "models/scaler.pkl"

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("[System Launcher] Trained ML model not found. Training model...")
        train_and_save_model()
    else:
        print("[System Launcher] Trained ML model binaries verified.")

    print("\n[System Launcher] Starting Real-Time Computer Vision & Web Telemetry Server...")
    print(" -> Dashboard URL: http://127.0.0.1:5000")
    print(" -> Telemetry API: http://127.0.0.1:5000/api/telemetry")
    print(" -> CSV Event Logs: saved to ./logs/")
    print("=" * 70)


if __name__ == "__main__":
    initialize_system()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)