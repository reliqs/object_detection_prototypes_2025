# Segmentation

Five segmentation methods in a single live script, switchable with keyboard keys while the camera is running.

> **Hardware Required:** Intel RealSense D455 via **USB 3.0**

---

## Scripts

| Script | Description |
|---|---|
| `segmented_object-detection.py` | Multi-method segmentation with live key switching |

---

## Segmentation Methods

| Key | Method |
|---|---|
| `1` | YOLO-seg (instance segmentation masks) |
| `2` | SAM — Segment Anything Model (Facebook Research) |
| `3` | Watershed |
| `4` | GrabCut |
| `5` | Contour-based |

---

## Install

```bash
pip install pyrealsense2 opencv-python numpy ultralytics scikit-image
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### SAM Model Weights

SAM requires `sam_vit_b_01ec64.pth` (~375 MB). This file is **not included in the repo**.

Download from the official SAM repo:
```
https://github.com/facebookresearch/segment-anything#model-checkpoints
```

Place the downloaded `.pth` file in this directory (`segmentation/`) before running.

---

## Usage

```bash
python segmented_object-detection.py
```

Press keys `1`–`5` to switch segmentation method at any time. Press `q` to quit.

---

## Output Files

Various `*_segmentation_output.avi` files are written per method (excluded from git).
