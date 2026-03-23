# Object Detection

Real-time YOLO object detection using the Intel RealSense D455, with 3D world coordinates (X, Y, Z in meters) derived from the camera's depth stream.

> **Hardware Required:** Intel RealSense D455 via **USB 3.0**

---

## Scripts

| Script | Description |
|---|---|
| `object_detection.py` | Main detection script — YOLO bounding boxes overlaid on RGB+depth feed |

---

## Install

```bash
pip install pyrealsense2 opencv-python numpy ultralytics
```

## Model Weights

Uses `yolov8n.pt` or `yolov8s.pt`. Both **auto-download on first run** via the `ultralytics` package — no manual setup needed.

---

## Usage

```bash
python object_detection.py
```

### Keyboard Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save screenshot |
| `r` | Toggle recording |

---

## Output Files

- `object_detection_output.avi` — recorded video (excluded from git)
- `detection_log.json` — per-frame detection log (excluded from git)

---

## Configuration

Edit the top of `object_detection.py` to adjust:
- `MODEL` — swap between `yolov8n.pt` (faster) and `yolov8s.pt` (more accurate)
- `CONFIDENCE_THRESHOLD` — filter low-confidence detections
- `DISPLAY_DEPTH` — toggle depth colormap overlay
