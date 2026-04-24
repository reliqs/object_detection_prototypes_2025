# RealSense D455 Computer Vision Experiments

<img width="100%" alt="segmentation_frame_20260423_174756" src="https://github.com/user-attachments/assets/a9194ba3-4e89-4add-9b18-dee12ceab89c" />

A collection of computer vision scripts for the Intel RealSense D455 depth camera, covering object detection, segmentation, activity recognition, predator-vision thermal overlays, and 3D scanning.

After having worked as a designer and engineer in the Anticipatory Computing and Systems Prototyping Labs at Intel Labs in 2014-2019, I decided to finally release some projects using the D455. These are scripts that employ YOLO and SAM variants for object detection in real-time. We've gone from a snail's pace of something like 2-4 fps (~2015), to 24+ fps for real-time detection since. 

> **Hardware Required:** Intel RealSense D455 connected via **USB 3.0** (USB 2.0 will cause stream failures or severely degraded performance).

---

## Directory Overview

| Directory | Description |
|---|---|
| [`object-detection/`](./object-detection/) | Real-time YOLO object detection with 3D coordinates from depth data |
| [`segmentation/`](./segmentation/) | 5 segmentation methods: YOLO-seg, SAM, Watershed, GrabCut, Contour |
| [`activity-detection/`](./activity-detection/) | MediaPipe pose landmarks + ML classifier for activity recognition |
| [`predator-vision/`](./predator-vision/) | Thermal/FLIR-style visualization with YOLO detection + depth colormapping |
| [`3d-scanner/`](./3d-scanner/) | SLAM-based 3D reconstruction, exports to OBJ/PLY/STL/GLTF |

---

## Common Dependencies

```bash
pip install pyrealsense2 opencv-python numpy ultralytics
```

Additional dependencies are listed in each subdirectory's README.

## Intel RealSense SDK

All scripts require the **Intel RealSense SDK 2.0** (`pyrealsense2`).

- Install guide: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
- Or via pip: `pip install pyrealsense2`

---

