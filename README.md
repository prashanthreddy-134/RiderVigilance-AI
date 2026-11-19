# 🚗 RiderVigilance AI
### Real-Time Rider Drowsiness Detection and Distraction Monitoring System

RiderVigilance AI is a real-time computer vision and machine learning-based monitoring system designed to identify signs of rider fatigue, drowsiness and mobile phone-related distractions.

The system uses facial landmark detection, eye and mouth analysis, machine learning classification and object detection to monitor rider behaviour through a webcam. A Flask-based web dashboard displays monitoring information and provides access to session logs.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Technologies Used](#-technologies-used)
- [System Architecture](#-system-architecture)
- [Detection Methodology](#-detection-methodology)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [Dashboard](#-dashboard)
- [API and Logging](#-api-and-logging)
- [Machine Learning Models](#-machine-learning-models)
- [Configuration](#-configuration)
- [Future Enhancements](#-future-enhancements)
- [Limitations](#-limitations)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## 📖 Project Overview

Drowsiness and distraction can reduce a rider's attention and reaction time. RiderVigilance AI aims to address this problem by monitoring visual indicators of fatigue and distraction using computer vision.

The application processes camera input and uses multiple detection components:

1. **Facial Landmark Detection:** MediaPipe FaceLandmarker identifies facial landmarks.
2. **Eye Monitoring:** Eye Aspect Ratio (EAR) is used to analyse eye closure.
3. **Yawn Detection:** Mouth Aspect Ratio (MAR) is used to analyse mouth opening.
4. **Fatigue Classification:** A trained Random Forest model classifies fatigue-related features.
5. **Mobile Phone Detection:** YOLOv8 ONNX detects mobile phones in camera frames.
6. **Web Monitoring:** Flask provides a dashboard for viewing monitoring information.
7. **Session Logging:** Detection events and session information are stored in CSV files.

The system combines these components into a real-time rider monitoring application.

## 🎯 Project Objectives

- Develop a real-time rider monitoring system using computer vision.
- Identify facial indicators associated with drowsiness.
- Monitor eye closure and yawning behaviour.
- Apply machine learning for fatigue classification.
- Detect mobile phone usage using object detection.
- Display monitoring information through a web dashboard.
- Maintain session and event records for later analysis.
- Provide an extensible foundation for intelligent rider safety monitoring.

## 🚀 Key Features

### 1. Real-Time Face Landmark Detection
- Uses MediaPipe FaceLandmarker.
- Processes facial landmarks from camera frames.
- Provides facial information for eye and mouth analysis.

### 2. Eye Closure and Drowsiness Monitoring
- Uses Eye Aspect Ratio (EAR).
- Analyses eye closure over time.
- Applies a continuous eye-closure duration condition.
- Tracks repeated short eye closures as part of fatigue monitoring.

### 3. Yawning Detection
- Uses Mouth Aspect Ratio (MAR).
- Analyses mouth opening behaviour.
- Applies a duration-based condition for yawn detection.

### 4. Machine Learning-Based Fatigue Classification
- Uses a trained Random Forest classifier.
- Loads the saved model from `models/random_forest_model.pkl`.
- Uses the saved scaler from `models/scaler.pkl`.

### 5. Mobile Phone Detection
- Uses a YOLOv8 ONNX model.
- Processes camera frames for mobile phone detection.
- Loads the model from `models/yolov8n.onnx`.

### 6. Web-Based Monitoring Dashboard
- Flask-based web application.
- Real-time monitoring interface.
- Live webcam switching option.
- Monitoring information and graphical displays.
- Alarm sound control.
- CSV log export option.

### 7. Session and Event Logging
- Stores monitoring session information.
- Saves generated CSV logs in the `logs/` directory.
- Supports reviewing recorded events.

---

## 🛠 Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Facial Landmark Detection | MediaPipe FaceLandmarker |
| Eye Analysis | Eye Aspect Ratio (EAR) |
| Mouth Analysis | Mouth Aspect Ratio (MAR) |
| Object Detection | YOLOv8 |
| Model Format | ONNX |
| Machine Learning | Scikit-learn |
| Classification Algorithm | Random Forest |
| Feature Scaling | StandardScaler |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data Logging | CSV |
| Development Environment | Visual Studio Code |

---

## 🏗 System Architecture

The system follows a modular architecture in which camera input is processed by computer vision and machine learning components before monitoring information is presented through the web dashboard.

### System Workflow

```text
       WEBCAM INPUT
             |
             v
       CAMERA MANAGER
             |
             v
     COMPUTER VISION PIPELINE
             |
       +-----+------+
       |            |
       v            v
  FACE LANDMARK   YOLOv8 ONNX
   DETECTION       DETECTION
       |
       +----------------------+
       |                      |
       v                      v
   EYE ANALYSIS           MOUTH ANALYSIS
     (EAR)                    (MAR)
       |                      |
       +----------+-----------+
                  |
                  v
          FATIGUE CLASSIFIER
           (RANDOM FOREST)
                  |
                  v
        MONITORING INFORMATION
                  |
          +-------+-------+
          |               |
          v               v
     FLASK SERVER      CSV LOGGER
          |
          v
      WEB DASHBOARD
          |
          v
    USER MONITORING
```

---

## 🔍 Detection Methodology

### 1. Facial Landmark Detection

MediaPipe FaceLandmarker identifies facial landmark positions from camera frames.

These landmarks are used to analyse eye and mouth movements.

### 2. Eye Aspect Ratio (EAR)

EAR is used to estimate the degree of eye openness.

The general formula is:

\[
EAR = \frac{||p_2-p_6||+||p_3-p_5||}{2||p_1-p_4||}
\]

Where:

- \(p_1\) and \(p_4\) represent horizontal eye landmarks.
- \(p_2,p_3,p_5,p_6\) represent vertical eye landmarks.

A decrease in EAR indicates that the eyes are closing.

### 3. Mouth Aspect Ratio (MAR)

MAR is used to analyse mouth opening.

The general approach compares vertical mouth distances with horizontal mouth width.

An increase in mouth opening can indicate yawning.

### 4. Fatigue Classification

A trained Random Forest classifier is used for fatigue-related classification.

The saved model and scaler are loaded during application startup.

### 5. Mobile Phone Detection

The YOLOv8 ONNX model processes image frames to identify mobile phones.

Detection results are used as an indicator of possible distraction.

---

## 📂 Project Structure

```text
RiderVigilance-AI/
│
├── core/
│   ├── __init__.py
│   ├── camera_manager.py
│   ├── cv_pipeline.py
│   ├── fatigue_classifier.py
│   ├── logger.py
│   └── mobile_detector.py
│
├── models/
│   ├── face_landmarker.task
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── yolov8n.onnx
│   └── yunet.onnx
│
├── web/
│   ├── __init__.py
│   ├── app.py
│   │
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   │
│   └── templates/
│       └── index.html
│
├── logs/
│
├── main.py
├── test_webcam.py
├── train_model.py
├── verify_system.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙ Installation and Setup

### Prerequisites

Install the following before running the application:

- Python 3.11
- Git
- Visual Studio Code (recommended)
- Webcam
- Working internet connection for initial package installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/prashanthreddy-134/RiderVigilance-AI.git
```

### Step 2: Navigate to the Project Directory

```bash
cd RiderVigilance-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate the environment again.

### Step 5: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 6: Install and Verify Compatible Dependencies

The tested environment used:

- Python 3.11.9
- MediaPipe 1.2.3
- Protobuf 7.36.2
- Scikit-learn 1.9.0

If needed, install the tested versions:

```bash
python -m pip install mediapipe==1.2.3
python -m pip install protobuf==7.36.2
python -m pip install scikit-learn==1.9.0
```

Check the installation:

```bash
python -m pip check
```

---

## ▶ Running the Application

After completing the installation, start the application:

```bash
python main.py
```

The application initializes the trained machine learning model, object detection model, facial landmark detector and Flask server.

When startup is successful, the dashboard is available at:

**Local Dashboard:**

```text
http://127.0.0.1:5000
```

**Telemetry API:**

```text
http://127.0.0.1:5000/api/telemetry
```

Keep the terminal running while using the application.

Press `CTRL + C` to stop the server.

---

## 🖥 Dashboard

The project provides a web-based monitoring interface built using Flask, HTML, CSS and JavaScript.

Dashboard capabilities include:

- System status display.
- Live camera monitoring option.
- Real-time monitoring information.
- Graphical monitoring displays.
- Alarm sound control.
- CSV log export functionality.

### Dashboard Screenshot

Add your actual dashboard screenshot to the repository.

Suggested location:

```text
screenshots/dashboard.png
```

Then display it here:

```markdown
![RiderVigilance AI Dashboard](screenshots/dashboard.png)
```

---

## 🔌 API and Logging

The Flask application exposes a telemetry endpoint:

```text
GET /api/telemetry
```

Local endpoint:

```text
http://127.0.0.1:5000/api/telemetry
```

The application also maintains CSV session logs.

Generated log files are stored in:

```text
logs/
```

These files can be used for reviewing recorded monitoring sessions and detection events.

---

## 🤖 Machine Learning Models

The project uses the following model files:

| Model File | Purpose |
|---|---|
| `random_forest_model.pkl` | Fatigue classification |
| `scaler.pkl` | Feature scaling |
| `yolov8n.onnx` | Mobile phone detection |
| `face_landmarker.task` | Facial landmark detection |
| `yunet.onnx` | Additional face detection model asset |

The model files are included in the repository because they are required by the application.

---

## 🔧 Configuration

Before running the application, verify the following:

1. All model files are present in the `models/` directory.
2. The webcam is connected and accessible.
3. The Python virtual environment is activated.
4. Required dependencies are installed.
5. Camera access is permitted by the operating system.
6. Port `5000` is available.

If the webcam does not open, check camera permissions and whether another application is using it.

---

## 🔮 Future Enhancements

Possible future improvements include:

- Integration with dedicated hardware for rider monitoring.
- GPS-based location tracking.
- Emergency notification integration.
- Mobile application support.
- Improved fatigue classification using larger datasets.
- Performance optimization for low-power devices.
- Additional distraction detection capabilities.
- Cloud-based monitoring and analytics.
- Extended model evaluation under different lighting conditions.

---

## ⚠ Limitations

- Detection performance depends on camera quality and lighting conditions.
- Facial landmark detection may be affected by face orientation and occlusion.
- Object detection performance depends on the trained model and camera input.
- The system requires a functioning camera and compatible runtime environment.
- Detection results should not be treated as a guaranteed indication of rider safety.

---

## 🛡 Disclaimer

RiderVigilance AI is an academic and prototype project developed for educational and research purposes.

It is intended to demonstrate computer vision and machine learning techniques for rider fatigue and distraction monitoring. It is not a certified vehicle safety system and must not replace responsible riding practices or dedicated safety equipment.

---

## 👨‍💻 Author

**Prashanth Reddy S**

Information Science and Engineering Student

Brindavan College of Engineering, Bengaluru

**GitHub:**  
https://github.com/prashanthreddy-134

**Project Repository:**  
https://github.com/prashanthreddy-134/RiderVigilance-AI

---

## ⭐ Support

If you find this project useful, consider giving the repository a star on GitHub.

Thank you for visiting RiderVigilance AI!
