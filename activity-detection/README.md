# Activity Detection

Activity recognition using MediaPipe pose landmarks combined with a RandomForest ML classifier. Detects common body activities in real time from the RealSense RGB stream.

> **Hardware Required:** Intel RealSense D455 via **USB 3.0**

---

## Scripts

| Script | Description |
|---|---|
| `realsense_activity_detection.py` | Pose estimation + activity classification |

---

## Activities Detected

- Standing
- Walking
- Sitting
- Waving
- Jumping

---

## Install

```bash
pip install pyrealsense2 opencv-python numpy mediapipe scikit-learn pandas
```

---

## Usage

```bash
python realsense_activity_detection.py
```

Press `q` to quit.

---

## Output Files

- `activity_detection_output.avi` — recorded video (excluded from git)
- `activity_log.json` — per-frame activity log (excluded from git)

---

## Custom Classifier Training

The RandomForest classifier can be retrained on your own labeled pose data. See comments inside `realsense_activity_detection.py` for the training data format and how to swap in a custom model.
