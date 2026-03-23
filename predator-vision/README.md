# Predator Vision

Thermal/FLIR-style visualization combining YOLO object detection with depth-based colormapping, inspired by the heat-vision aesthetic from the Predator films.

> **Hardware Required:** Intel RealSense D455 via **USB 3.0**

---

## Scripts

| Script | Purpose |
|---|---|
| `predator_vision_complete.py` | Full system with mode menu — **primary entry point** |
| `predator_vision_complete-1.py` | Variant copy of complete version |
| `predator_vision_complete-2.py` | Variant copy of complete version |
| `predator_vision_simple_start.py` | Simplified quick-start, easier camera init |
| `predator_object_detection.py` | Standalone detection + thermal overlay, no menu |
| `predator_vision_fixed.py` | Menu/display fix iteration |
| `predator_vision_fixed-text.py` | HUD text size fix (smaller font) |
| `predator_debug_version.py` | Diagnostic: tests menu rendering |
| `predator_minimal_test.py` | Diagnostic: tests all imports one-by-one |

---

## Install

```bash
pip install pyrealsense2 opencv-python numpy ultralytics matplotlib scipy
```

---

## Recommended Start Order

If you're setting this up for the first time or running into issues:

1. `python predator_minimal_test.py` — verifies all imports load correctly
2. `python predator_vision_simple_start.py` — confirms camera streams and basic overlay
3. `python predator_vision_complete.py` — full experience with mode menu

---

## Thermal Style Options

The following colormaps can be selected at runtime or configured in the script:

| Style | Description |
|---|---|
| `predator` | Green-tinted night-vision style |
| `flir` | FLIR white-hot palette |
| `iron` | Iron/heat map |
| `rainbow` | Full-spectrum rainbow |
| `hot` | Black → red → yellow → white |
| `plasma` | Matplotlib plasma colormap |
| `viridis` | Matplotlib viridis colormap |

---

## Troubleshooting

- **Camera not detected:** Confirm USB 3.0 connection. USB 2.0 will fail or produce corrupt frames.
- **Menu not rendering:** Try `predator_vision_fixed.py` or `predator_debug_version.py` to isolate the issue.
- **Import errors:** Run `predator_minimal_test.py` to identify which package is missing.
- **Low frame rate:** Switch to `yolov8n.pt` (nano model) for faster inference.
