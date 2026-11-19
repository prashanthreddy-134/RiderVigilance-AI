import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def generate_benchmark_dataset():
    """
    Generates 23,438 frame instances matching the project review dataset breakdown:
      - Alert (Focused): 5,629 frames (Class 0)
      - Mild Fatigue: 5,802 frames (Class 1)
      - Severe Fatigue: 6,061 frames (Class 2)
      - Drowsy: 5,946 frames (Class 3)
    """
    np.random.seed(42)

    # 1. Alert (5,629 instances)
    # Open eyes (EAR ~ 0.28 - 0.35), closed mouth (MAR ~ 0.12 - 0.22), normal blinks (12-20/min), 0 yawns, 0s closure
    n_alert = 5629
    ear_alert = np.random.normal(0.30, 0.025, n_alert)
    mar_alert = np.random.normal(0.15, 0.03, n_alert)
    blink_alert = np.random.normal(16.0, 3.0, n_alert)
    yawn_alert = np.random.exponential(0.1, n_alert)
    closure_alert = np.zeros(n_alert)
    y_alert = np.zeros(n_alert, dtype=int)

    # 2. Mild Fatigue (5,802 instances)
    # Slightly lower EAR (~ 0.22 - 0.26), varying blink rate (8-11 or 24-28/min), occasional brief closure (0.2s - 0.9s)
    n_mild = 5802
    ear_mild = np.random.normal(0.24, 0.025, n_mild)
    mar_mild = np.random.normal(0.20, 0.05, n_mild)
    b_part1 = np.random.normal(8.0, 1.5, n_mild // 2)
    b_part2 = np.random.normal(27.0, 2.0, n_mild - n_mild // 2)
    blink_mild = np.concatenate([b_part1, b_part2])
    yawn_mild = np.random.exponential(0.3, n_mild)
    closure_mild = np.random.uniform(0.0, 0.9, n_mild)
    y_mild = np.ones(n_mild, dtype=int)

    # 3. Severe Fatigue (6,061 instances)
    # Yawning (MAR ~ 0.50 - 0.70), lower EAR (~ 0.19 - 0.24), higher yawn count (1.5 - 3.5), closure (0.5s - 1.8s)
    n_severe = 6061
    ear_severe = np.random.normal(0.21, 0.025, n_severe)
    mar_severe = np.random.normal(0.55, 0.08, n_severe)
    blink_severe = np.random.normal(7.0, 2.5, n_severe)
    yawn_severe = np.random.normal(2.2, 0.6, n_severe)
    closure_severe = np.random.uniform(0.5, 1.8, n_severe)
    y_severe = np.full(n_severe, 2, dtype=int)

    # 4. Drowsy (5,946 instances)
    # Low EAR (< 0.20), continuous eye closure (2.0s - 4.5s), very low blink rate (< 5/min)
    n_drowsy = 5946
    ear_drowsy = np.random.normal(0.15, 0.02, n_drowsy)
    mar_drowsy = np.random.normal(0.30, 0.10, n_drowsy)
    blink_drowsy = np.random.normal(3.0, 1.5, n_drowsy)
    yawn_drowsy = np.random.normal(1.0, 0.8, n_drowsy)
    closure_drowsy = np.random.uniform(2.0, 5.0, n_drowsy)
    y_drowsy = np.full(n_drowsy, 3, dtype=int)

    # Combine into DataFrame
    ear_all = np.clip(np.concatenate([ear_alert, ear_mild, ear_severe, ear_drowsy]), 0.10, 0.40)
    mar_all = np.clip(np.concatenate([mar_alert, mar_mild, mar_severe, mar_drowsy]), 0.05, 0.80)
    blink_all = np.clip(np.concatenate([blink_alert, blink_mild, blink_severe, blink_drowsy]), 0.0, 40.0)
    yawn_all = np.clip(np.concatenate([yawn_alert, yawn_mild, yawn_severe, yawn_drowsy]), 0.0, 5.0)
    closure_all = np.clip(np.concatenate([closure_alert, closure_mild, closure_severe, closure_drowsy]), 0.0, 6.0)
    y_all = np.concatenate([y_alert, y_mild, y_severe, y_drowsy])

    df = pd.DataFrame({
        "ear": ear_all,
        "mar": mar_all,
        "blink_rate": blink_all,
        "yawn_rate": yawn_all,
        "closure_duration": closure_all,
        "target": y_all
    })

    return df

def train_and_save_model():
    print("=" * 60)
    print("  RIDER FATIGUE DETECTION - RANDOM FOREST MODEL TRAINING")
    print("=" * 60)

    df = generate_benchmark_dataset()
    print(f"Dataset generated: {len(df):,} total frames across 4 states.")
    print("\nClass Counts:\n", df["target"].value_counts().sort_index())

    X = df[["ear", "mar", "blink_rate", "yawn_rate", "closure_duration"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    rf.fit(X_train_scaled, y_train)

    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- MODEL EVALUATION RESULTS ---")
    print(f"Overall Classification Accuracy: {acc * 100:.2f}%")
    print("\nPer-Class Breakdown:")
    
    target_names = ["Alert (Focused)", "Mild Fatigue", "Severe Fatigue", "Drowsy"]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    # Save Models
    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest_model.pkl"
    scaler_path = "models/scaler.pkl"

    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[SUCCESS] Trained Random Forest model saved to: {model_path}")
    print(f"[SUCCESS] Scaler saved to: {scaler_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_and_save_model()
