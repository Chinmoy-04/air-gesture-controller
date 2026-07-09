# Air-Gesture Touchless Controller

A real-time, touchless hand-gesture control system for computers — built as an Embedded AI course project at the **Deggendorf Institute of Technology** under Prof. Dr. Thomas Ewender.

The system recognises hand gestures via a camera and maps them to OS-level actions: advancing presentation slides, controlling media playback, toggling fullscreen, moving the mouse cursor, and managing desktop windows — all without touching any input device.

---

## Table of Contents

1. [Demo](#demo)
2. [System Overview](#system-overview)
3. [Hardware](#hardware)
4. [Gesture Classes](#gesture-classes)
5. [Control Profiles](#control-profiles)
6. [Architecture](#architecture)
7. [Dataset and Training](#dataset-and-training)
8. [Model Export to IMX500](#model-export-to-imx500)
9. [Model Performance](#model-performance)
10. [File Structure](#file-structure)
11. [Installation](#installation)
12. [Running the System](#running-the-system)
13. [Pipeline Constants](#pipeline-constants)
14. [OS Integration Details](#os-integration-details)
15. [Known Limitations](#known-limitations)
16. [Tech Stack](#tech-stack)

---

## Demo

> **Hardware:** Raspberry Pi 5 + Sony IMX500 AI Camera  
> **Inference:** Runs entirely on the IMX500's on-sensor NPU — the Pi CPU is not used for neural network processing

See [`MODEL_PERFORMANCE.md`](MODEL_PERFORMANCE.md) for training curves, confusion matrices, and validation visualisations.

---

## System Overview

The project has **two runtime targets** sharing the same gesture engine:

| Target | Entry point | Inference | Use |
|---|---|---|---|
| **Raspberry Pi 5** | `pi_tester.py` | Sony IMX500 NPU (on-sensor) | Deployment |
| **Windows laptop** | `local_tester.py` | YOLO11n on CPU/GPU via Ultralytics | Development & testing |

On the Pi, the camera module performs YOLO11n inference on its own Neural Processing Unit. The Pi receives only bounding boxes and confidence scores over the MIPI CSI-2 interface — not raw frames for YOLO processing — leaving the CPU free for gesture logic and UI.

On Windows, a standard webcam feed is run through the YOLO11n model via Ultralytics, enabling full development and testing without Pi hardware.

---

## Hardware

### Raspberry Pi 5
- Broadcom BCM2712, quad-core Cortex-A76 @ 2.4 GHz
- 8 GB LPDDR4X RAM
- Raspberry Pi OS Bookworm (Debian 12, 64-bit)
- **X11 session required** (default Wayland is incompatible with xdotool/pyautogui global key injection)

### Sony IMX500 AI Camera (Raspberry Pi AI Camera)
- 12.3 MP stacked CMOS sensor with Sony's edge NPU **built directly onto the sensor die**
- On-sensor inference: the Pi receives output tensors (bboxes, scores) — not raw pixels
- Model loaded at startup as `network.rpk` firmware into the camera's flash
- Inference input: **640 × 640** (model training resolution)
- Camera frame to Pi: **640 × 480** @ 15 FPS, RGB888 pixel format (stored as BGR in memory — OpenCV's native format)
- Interface: **Picamera2** Python API (wraps libcamera)

### Development / Training Machine
- Windows 10/11 with CUDA-capable GPU (NVIDIA RTX)
- Used for YOLO11n training and IMX500 model export

---

## Gesture Classes

The model detects **6 gesture classes**:

| Gesture | Class name | Role |
|---|---|---|
| 👍 Thumbs up | `thumbs_up` | Static — profile action |
| 👎 Thumbs down | `thumbs_down` | Static — profile action |
| 🖐 Open palm | `open_palm` | Static — profile action |
| ✊ Closed fist | `fist_close` | Static — profile action / click sequence |
| ☝ Index finger up | `index_finger_up` | Dynamic — drives swipe tracking |
| ✌ Two fingers | `two_fingers` | Dynamic — drives cursor / click sequences |

---

## Control Profiles

Switchable at runtime with the **P key**. The app always starts in **Presentation** mode.

### Presentation Profile (default)

| Gesture | Action |
|---|---|
| Thumbs Up | Next slide — `Page Down` |
| Thumbs Down | Previous slide — `Page Up` |
| Open Palm | Toggle fullscreen (mode selectable: `F` / `F11` / `Ctrl+L`) |
| Closed Fist | Play / Pause media (`playerctl` on Pi, `XF86AudioPlay` fallback) |
| Index Finger → swipe right | Seek forward ~5 s |
| Index Finger → swipe left | Seek back ~5 s |
| Index Finger → swipe up | Volume up |
| Index Finger → swipe down | Volume down |
| Two Fingers | Move mouse cursor (relative / trackpad mode) |
| Two Fingers → Fist | Left click |
| Two Fingers → Palm | Right click |

Fullscreen mode is cycled at runtime with the **M key**:

| Mode | Key sent | Target apps |
|---|---|---|
| Video | `F` | YouTube, VLC, HTML5 video |
| PDF / Browser | `F11` | SumatraPDF, Edge, Chrome |
| Acrobat | `Ctrl+L` | Adobe Reader |

### Desktop / OS Profile (press P once)

| Gesture | Action |
|---|---|
| Thumbs Up | Switch application — cycles open windows via `wmctrl` |
| Thumbs Down | Open terminal — `lxterminal` → `xterm` fallback |
| Open Palm | App launcher — `lxpanelctl run` |
| Closed Fist | Close active window — `wmctrl -i -c` |
| Two Fingers | Move mouse cursor (same as presentation) |
| Two Fingers → Fist / Palm | Left / right click (same as presentation) |

---

## Architecture

### Detection-to-Action Pipeline

```
Camera frame
    │
    ▼  IMX500 NPU (Pi) / YOLO11n GPU-CPU (Windows)
Raw detections: [(class_name, confidence, bbox_xywh), ...]
    │
    ▼  pick_best_from_list()  —  confidence ≥ 0.75, NMS IoU 0.45
Three parallel detection streams:
  ├── static_detection   (thumbs_up / thumbs_down / open_palm / fist_close)
  ├── index_detection    (index_finger_up — swipe tracking)
  └── mouse_detection    (two_fingers — cursor + click sequences)
    │
    ▼  GesturePipeline.process_frame()
    │
    ├── Lane 1: Mouse / cursor
    │     two_fingers visible:
    │       ScreenCursorController.update() → relative delta × sensitivity (4.0)
    │       → pyautogui.moveTo()
    │     two_fingers lost → 12-frame grace period holds cursor position
    │     Click sequences (within 1.2 s window):
    │       two_fingers → fist_close  → left click
    │       two_fingers → open_palm   → right click
    │
    ├── Lane 2: Static gestures
    │     StaticGestureVoter: gesture must win 7 consecutive frames
    │     Cooldown: 1.5 s between any two fired actions
    │     → execute_static_gesture(gesture, profile)
    │         Presentation: PgDn / PgUp / fullscreen / play-pause
    │         Desktop:      alt-tab / open terminal / launcher / close window
    │
    └── Lane 3: Swipe tracking
          Centroid added to deque (max 15 frames, EMA α=0.35)
          Static gesture held → blocks swipe (STATIC_BLOCKS_SWIPE)
          detect_swipe() evaluates:
            horizontal: |Δx| > max(55 px, bbox_w × 0.28) AND v_x > 2.5 px/frame
            vertical:   |Δy| > max(70 px, bbox_h × 0.30) AND v_y > 2.5 px/frame
          → execute_swipe() → seek / volume via playerctl / wpctl
```

### Cursor Control — Relative (Trackpad) Mode

Instead of mapping hand position absolutely to screen coordinates (which causes jarring jumps on tracking loss), the cursor uses **relative delta movement**:

- First frame of a tracking session reads the actual current cursor position as the anchor
- Each subsequent frame: `Δ_screen = Δ_hand × CURSOR_RELATIVE_SENSITIVITY (4.0)`
- On tracking loss and resume: cursor picks up from its current position, no jump

### Key Design Decisions

| Decision | Rationale |
|---|---|
| `MAX_DETECTIONS = 1` | Only the highest-confidence box per frame is used — eliminates confusion from background detections |
| `STATIC_BLOCKS_SWIPE` | Holding a static gesture suppresses the swipe lane, preventing accidental media triggers while holding a pose |
| Profile always resets to Presentation on startup | Desktop/OS profile can open terminals and close windows — persisting it across sessions caused accidental destructive actions |
| Background thread for window tracking | Per-frame `xdotool` subprocess calls (30/sec at 15 FPS) caused severe latency; replaced with 500 ms polling thread |

---

## Dataset and Training

### Dataset

| Stat | Value |
|---|---|
| Total images | **690** |
| Manual phone captures + LabelImg labels | 529 |
| Auto-captured via `capture_training_data.py` | 161 |
| Train split | 552 (80%) |
| Validation split | 138 (20%) |
| Classes | 6 |
| Annotation format | YOLO (normalised `cx cy w h` per line) |

All images were captured from scratch — no public dataset was used. Phone-captured images cover varied lighting conditions, backgrounds, skin tones, and hand distances. The `capture_training_data.py` script bootstrapped additional data using the model's own detections as pseudo-labels (semi-supervised augmentation loop).

### `capture_training_data.py` — Auto-Labelling Tool

Runs the current model on a live webcam feed and automatically saves image-label pairs:

- **Auto mode** (A key toggle): saves a frame when the same class wins `--vote-frames` (default 5) consecutive detections at `≥ --conf` (default 0.65) confidence
- **Manual mode** (Space key): saves the current frame on demand
- Minimum save interval (default 1.5 s) prevents near-identical duplicates
- Output: flat `updated dataset/images/` + `updated dataset/labels/` directories

### Two-Phase Training (`train_best.py`)

Both phases use **YOLO11n** (nano variant) — the largest model compatible with Sony IMX500 memory constraints.

#### Phase 1 — High-Augmentation Warmup

| Hyperparameter | Value |
|---|---|
| Epochs | 200 (early stop patience 30) |
| Optimiser | AdamW, lr0=0.001, cosine decay to LRF=0.05 |
| Batch size | 16 |
| Backbone frozen | First 5 warmup epochs, then full unfreeze |
| Label smoothing | 0.1 |
| Dropout | 0.1 |
| Multi-scale | ±25% |

**Augmentation (Ultralytics):** Mosaic 1.0, MixUp 0.25, CutMix 0.15, CopyPaste 0.10, HSV jitter (H±0.025, S±0.80, V±0.55), rotation ±25°, translate ±20%, scale ±75%, shear ±8°, random erasing 0.50, RandAugment. Vertical flip disabled (upside-down hands are not valid).

**Albumentations (custom pipeline, strength 1.0):** Motion blur, Gaussian blur, median blur, Gaussian noise, ISO noise, CLAHE, brightness/contrast, JPEG compression artefacts, CoarseDropout (up to 10 rectangular holes).

Mosaic disabled for the final 20 epochs (`close_mosaic=20`) for cleaner convergence.

#### Phase 2 — Clean-Image Fine-Tuning

Resumes from Phase 1 best checkpoint.

| Hyperparameter | Value |
|---|---|
| Epochs | 50 (early stop patience 15) |
| Optimiser | AdamW, lr0=0.0002, cosine decay to LRF=0.10 |
| Batch size | 16 |
| Mosaic / MixUp / CutMix | **Disabled** |
| Albumentations strength | 0.4 (mild) |

All composite augmentations are off — the model fine-tunes on realistic single images. This phase consistently adds **+1–3% mAP50** over Phase 1 alone.

---

## Model Export to IMX500

The IMX500 cannot execute a standard PyTorch `.pt` file. The export pipeline:

```
best.pt  (FP32 PyTorch weights)
    │
    ▼  model.export(format="imx", int8=True, data=dataset.yaml, imgsz=640)
    │  Ultralytics calls imxconv-pt (Sony Edge-MDT) under the hood
    │  INT8 post-training quantisation — calibrated on validation split
    │
packerOut.zip  (Sony intermediate format, generated on Windows dev machine)
    │
    ▼  imx500-package -i packerOut.zip -o out   ← must run on Raspberry Pi
    │
network.rpk  (firmware blob loaded into IMX500 flash at camera startup)
```

**Export notes:**
- `ultralytics.engine.exporter.LINUX = True` patch required — Ultralytics gates IMX export behind a Linux check
- `imxconv-pt` requires **Java 17+** (not Java 8)
- Sony's `imxconv-pt` uses Google `ortools` DLLs; Windows Application Control policies may block them — adjust security settings or export from WSL
- The `imx500-package` step must be run on the Pi itself

---

## Model Performance

| Metric | Value |
|---|---|
| Precision | **95.9%** |
| Recall | **96.9%** |
| mAP50 | **98.1%** |
| mAP50-95 | **84.4%** |

Full charts (training curves, confusion matrices, PR curves, validation visualisations) in [`MODEL_PERFORMANCE.md`](MODEL_PERFORMANCE.md).

---

## File Structure

```
AISD/
├── gesture_engine.py          # Core pipeline: voting, EMA, cursor, swipe, profiles, actions
├── gesture_ui.py              # OpenCV UI: sidebar legend, status panel, lite/full mode
├── os_actions.py              # Platform OS control: xdotool, wmctrl, wpctl, playerctl
├── local_tester.py            # Windows entry: webcam + YOLO inference on CPU/GPU
├── pi_tester.py               # Pi entry: IMX500 + Picamera2, coordinate scaling
├── export.py                  # best.pt → INT8 IMX500 packerOut.zip
├── train_best.py              # Two-phase YOLO11n training (recommended)
├── train_augmented.py         # Earlier single-phase training (superseded)
├── capture_training_data.py   # Auto-labelled webcam data collection
├── prepare_training_data.py   # 80/20 split + dataset.yaml generation
├── dataset.yaml               # YOLO dataset config (6 classes, paths)
├── .gesture_config.json       # Runtime user prefs (fullscreen mode; NOT profile)
├── MODEL_PERFORMANCE.md       # Training metrics and visualisation charts
├── model_performance/         # Image assets for MODEL_PERFORMANCE.md
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── BoxPR_curve.png / BoxP_curve.png / BoxR_curve.png / BoxF1_curve.png
│   ├── labels.jpg
│   ├── train_batch{0,1,2}.jpg
│   └── val_batch{0,1,2}_{labels,pred}.jpg
├── yolo_data/                 # Split dataset
│   ├── train/images/ + labels/
│   └── val/images/ + labels/
└── runs/detect/               # Ultralytics training outputs
    └── runs/detect/train-best-p2/  ← final model
        └── weights/best.pt
```

---

## Installation

### Windows (development / training)

```powershell
# Python 3.10+ required
python -m venv .venv
.venv\Scripts\Activate.ps1

# CUDA-enabled PyTorch (adjust cu128 for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Project dependencies
pip install ultralytics opencv-python pyautogui albumentations

# IMX500 export tools (Java 17+ required in PATH / JAVA_HOME)
pip install edge-mdt[pt]
```

### Raspberry Pi 5 (deployment)

```bash
# System packages
sudo apt update && sudo apt full-upgrade
sudo apt install -y python3-picamera2 imx500-all imx500-tools
sudo apt install -y xdotool wmctrl python3-xlib playerctl

# Switch display server to X11 (required for xdotool / pyautogui)
sudo raspi-config
# → Advanced Options → Wayland → X11 → reboot

# Re-enable VNC after switching (X11 uses a different service)
sudo raspi-config
# → Interface Options → VNC → Enable

# Python venv on Pi
python3 -m venv ~/gesture-controller/.venv --system-site-packages
source ~/gesture-controller/.venv/bin/activate
pip install pyautogui ultralytics
```

---

## Running the System

### Raspberry Pi

```bash
cd ~/gesture-controller
source .venv/bin/activate

python3 pi_tester.py \
    --model best_imx_model/out/network.rpk \
    --labels best_imx_model/labels.txt \
    --threshold 0.60 \
    --fps 15

# Optional flags:
#   --full-ui          Higher-quality UI (heavier on Pi CPU)
#   --threshold 0.55   Lower threshold if INT8 model misses detections
#   --fps 10           Reduce frame rate to ease Pi CPU load
```

### Windows

```powershell
# auto-picks the newest best.pt under runs/detect/
python local_tester.py
```

### Runtime Controls

| Key | Action |
|---|---|
| **P** | Cycle profile: Presentation ↔ Desktop/OS |
| **M** | Cycle fullscreen mode: Video (`F`) → PDF (`F11`) → Acrobat (`Ctrl+L`) |
| **Q** | Quit |

---

### Training

```bash
# Full two-phase training (recommended)
python train_best.py

# Phase 1 only
python train_best.py --phase 1

# Phase 2 only (auto-finds Phase 1 weights)
python train_best.py --phase 2

# Phase 2 from a specific checkpoint
python train_best.py --phase 2 --weights path/to/best.pt
```

### Collecting More Training Data

```bash
# Capture thumbs_up images
python capture_training_data.py --class thumbs_up

# Stricter confidence, faster capture interval
python capture_training_data.py --class fist_close --conf 0.70 --interval 1.0

# In the window: A = toggle auto-capture | Space = manual save | Q = quit

# Rebuild dataset split and retrain
python prepare_training_data.py
python train_best.py
```

### Exporting to Pi

```bash
# On Windows dev machine — generates packerOut.zip
python export.py

# Copy to Pi
scp -r runs/detect/runs/detect/train-best-p2/weights/best_imx_model/ \
    admin@<PI_IP>:~/gesture-controller/best_imx_model/

# On the Pi — package into network.rpk
imx500-package -i best_imx_model/packerOut.zip -o best_imx_model/out
```

---

## Pipeline Constants

All tunable in `gesture_engine.py`:

| Constant | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.75` | Min detection score to enter the pipeline (Pi uses `--threshold 0.60`) |
| `INFERENCE_IOU` | `0.45` | NMS IoU threshold |
| `MAX_DETECTIONS` | `1` | Only the single best box per frame is processed |
| `STATIC_VOTE_FRAMES` | `7` | Consecutive frames required to fire a static gesture |
| `COOLDOWN_SECONDS` | `1.5` | Minimum gap between any two actions |
| `CENTROID_QUEUE_LENGTH` | `15` | Frames of centroid history for swipe detection |
| `MIN_QUEUE_SAMPLES_FOR_SWIPE` | `8` | Minimum frames before swipe is evaluated |
| `CENTROID_EMA_ALPHA` | `0.35` | EMA smoothing for swipe centroid trajectory |
| `SWIPE_THRESHOLD_X_FRAC` | `0.28` | Horizontal swipe threshold as fraction of bbox width |
| `SWIPE_THRESHOLD_Y_FRAC` | `0.30` | Vertical swipe threshold as fraction of bbox height |
| `SWIPE_MIN_X` | `55 px` | Absolute minimum horizontal swipe displacement |
| `SWIPE_MIN_Y` | `70 px` | Absolute minimum vertical swipe displacement |
| `SWIPE_MIN_VELOCITY` | `2.5 px/frame` | Minimum average velocity to register a swipe |
| `MOUSE_EMA_ALPHA` | `0.22` | EMA smoothing for cursor centroid (smoother than swipe) |
| `CURSOR_RELATIVE_MODE` | `True` | Trackpad-style relative delta movement |
| `CURSOR_RELATIVE_SENSITIVITY` | `4.0` | Screen pixels per camera pixel of hand movement |
| `MOUSE_SEQUENCE_WINDOW_SECONDS` | `1.2` | Time window for click sequence detection |
| `MOUSE_GRACE_FRAMES` | `12` | Frames cursor holds position after `two_fingers` lost |

---

## OS Integration Details

### Linux / Raspberry Pi Tool Stack

| Tool | Purpose | Install |
|---|---|---|
| `xdotool` | X11 key injection to specific windows | `sudo apt install xdotool` |
| `wmctrl` | X11 window list, focus control, close | `sudo apt install wmctrl` |
| `wpctl` | PipeWire volume control | Included with PipeWire |
| `playerctl` | MPRIS media control over D-Bus (no focus needed) | `sudo apt install playerctl` |
| `lxpanelctl` | LXDE run dialog (app launcher) | Included with LXDE |
| `python3-xlib` | X11 Python bindings for pyautogui cursor | `sudo apt install python3-xlib` |

### Why X11 (not Wayland)

Raspberry Pi OS Bookworm defaults to Wayland (labwc compositor). Wayland's security model prevents applications from injecting global keyboard/mouse events to other windows — `xdotool` and `pyautogui` cannot send Alt+Tab, window manager shortcuts, or cursor movements across apps on Wayland. The project requires **switching to X11** via `raspi-config → Advanced Options → Wayland → X11`.

After switching, the VNC service must be re-enabled — Wayland and X11 each use a separate VNC daemon (`wayvnc` vs `vncserver-x11-serviced`).

---

## Known Limitations

| Limitation | Detail |
|---|---|
| VNC headless resolution | Pi 5 headless X11 caps at ~704×432 (no EDID from monitor). Connect a physical HDMI monitor for full resolution. |
| Wayland incompatibility | Desktop/OS profile requires X11. All gesture features work on Wayland for presentation control only. |
| INT8 confidence shift | Quantisation from FP32 to INT8 can lower raw confidence scores. Use `--threshold 0.55–0.65` on Pi. |
| No formal latency benchmark | 60–80 ms gesture-to-action estimate is based on 15 FPS frame period (~67 ms) + processing overhead. Not measured with instrumentation. |


---

## Tech Stack

| Component | Technology |
|---|---|
| Object detection model | [YOLO11n](https://docs.ultralytics.com/) (Ultralytics) |
| Training framework | Ultralytics 8.x + PyTorch (CUDA AMP) |
| Augmentation | Ultralytics built-in + [Albumentations](https://albumentations.ai/) |
| Data annotation | [LabelImg](https://github.com/HumanSignal/labelImg) |
| Pi camera API | [Picamera2](https://github.com/raspberrypi/picamera2) (libcamera) |
| On-sensor inference | Sony IMX500 NPU (Edge-MDT / imxconv-pt) |
| Cursor / keyboard | [PyAutoGUI](https://pyautogui.readthedocs.io/) + xdotool (Linux) |
| Window management | wmctrl |
| Media control | playerctl (MPRIS D-Bus) |
| Volume control | wpctl (PipeWire / WirePlumber) |
| UI | OpenCV (`cv2`) |
| Language | Python 3.10+ |

---

*Embedded AI course project — Deggendorf Institute of Technology, 2026*
