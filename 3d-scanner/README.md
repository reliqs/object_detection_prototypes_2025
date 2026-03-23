# 3D Scanner

SLAM-based 3D reconstruction using the Intel RealSense D455. Walk around an object or environment and the script builds a 3D mesh in real time, exported to OBJ/PLY/STL/GLTF.

> **Hardware Required:** Intel RealSense D455 via **USB 3.0**
> **RAM:** 16 GB recommended for large scans

---

## Scripts

Scripts are listed in order of development — later versions are more capable:

| Script | Lines | Notes |
|---|---|---|
| `realsense_3d_scanner.py` | ~440 | Original basic SLAM implementation |
| `realsense_3d_scanner-updated.py` | ~630 | Auto-detects device stream configuration |
| `realsense_3d_scanner-open3d.py` | ~840 | Open3D integration; supports SLAM + simple mode |
| `realsense_3d_scanner_23.py` | ~825 | Most complete non-IMU version — **recommended** |
| `realsense_3d_scanner_IMU.py` | ~1069 | IMU sensor fusion for improved pose tracking |
| `realsense_3d_scanner (1).py` | ~841 | Variant copy |

**Recommended entry point:** `realsense_3d_scanner_23.py`

---

## Install

```bash
pip install pyrealsense2 opencv-python numpy open3d
```

For the IMU variant (`realsense_3d_scanner_IMU.py`):

```bash
pip install scipy
```

---

## Usage

```bash
python realsense_3d_scanner_23.py
```

### Scanning Modes

| Mode | Description |
|---|---|
| **SLAM** | Tracks camera pose frame-to-frame; reconstructs geometry relative to a fixed origin |
| **Simple** | Accumulates point clouds without pose estimation; faster but drifts over distance |

Select mode at startup via the on-screen prompt.

---

## Output

Scans are saved as `.obj` files (excluded from git due to file size).

To view output files:
- **MeshLab** (free): `File → Import Mesh`
- **Blender** (free): `File → Import → Wavefront (.obj)`

---

## Troubleshooting

- **Stream failures or corrupted frames:** Check that you are using **USB 3.0**. USB 2.0 cannot sustain the required bandwidth.
- **"Device not found" error:** Close Intel RealSense Viewer if it is open — only one application can access the camera at a time.
- **Out of memory during large scans:** Reduce scan duration or enable the decimation filter (see script comments) to downsample point clouds.
- **Poor pose tracking / mesh drift:** Use `realsense_3d_scanner_IMU.py` to incorporate accelerometer/gyroscope data for better pose estimation.
