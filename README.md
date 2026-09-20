# AI Shoplifting & Concealment Detection System

## Overview

An AI-powered CCTV surveillance system designed to detect suspicious shoplifting and item-concealment behavior using computer vision, human pose estimation, object detection, and temporal behavioral analysis.

The system analyzes surveillance video, tracks people, evaluates body and hand movements, and generates evidence when suspicious activity is detected.

## Key Features

- Person detection and tracking
- Human pose estimation
- Hand and wrist movement analysis
- Shoplifting and concealment behavior detection
- Temporal behavioral analysis using BiLSTM
- Suspect identification
- Automated suspect snapshots
- Automated incident and evidence logging
- Email alert support

## System Pipeline

CCTV Video  
↓  
Person Detection  
↓  
Pose Estimation  
↓  
Person Tracking  
↓  
Behavioral Analysis  
↓  
Shoplifting Detection  
↓  
Suspect Identification  
↓  
Evidence Logging

## Technology Stack

- Python
- YOLO
- YOLO Pose Estimation
- OpenCV
- PyTorch
- BiLSTM
- Computer Vision

## Project Structure

```text
shoplifting-detection-AI/
│
├── models/
│   ├── best_bilstm.pth
│   ├── bilstm_classifier.py
│   └── __init__.py
│
├── utils/
│   ├── email_alerts.py
│   ├── normalization.py
│   └── __init__.py
│
├── extract_dataset.py
├── requirements.txt
├── run_inference.py
├── train.py
├── yolov8n.pt
└── yolov8n-pose.pt
```

## How It Works

The system processes CCTV footage and detects people using computer vision models. Pose estimation is used to extract human body keypoints and movement information. The extracted temporal features are analyzed to identify behavioral patterns associated with potential shoplifting or concealment.

When suspicious activity is detected and verified by the system, suspect snapshots and incident evidence can be automatically logged for further investigation.

## Output

The system produces detection results from surveillance footage and can automatically log detected incidents, suspect snapshots, and evidence.

Example outputs include:

- Suspect identification
- Suspect snapshots
- Behavioral detection results
- Incident logs
- Video evidence

## Future Improvements

- Real-time CCTV camera integration
- Multi-camera tracking
- Improved behavioral classification
- Real-time monitoring dashboard
- Automated forensic report generation
- Improved detection across different retail environments

## License

This project is developed as a prototype for the Hack Devengers 2.0 hackathon.
