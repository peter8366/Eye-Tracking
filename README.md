# Eye-Tracking

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9-ee4c2c.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00a3a3.svg?style=flat&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5c3ee8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)

**Real-time point-of-gaze estimation on a commodity webcam, split across a lightweight
client and a GPU inference server.**

This project estimates *where on the screen a person is looking* — not just the gaze
direction — from a single RGB camera. A laptop or desktop captures the video and runs
face/landmark detection locally, while the appearance model runs remotely on a GPU. The
two halves talk over a TCP socket, which keeps the client light enough to run on machines
without a dedicated GPU.

---

## Why a split architecture?

Appearance-based gaze models are too heavy for many client machines, but streaming raw
video to a server wastes bandwidth and adds latency. This pipeline splits the work at the
point where the data is smallest:

| Stage | Runs on | Why there |
|---|---|---|
| Capture, face mesh, head-pose estimation | **Client** | Needs the camera; cheap enough for CPU |
| Normalization to canonical face/eye crops | **Client** | Shrinks the payload to three small crops |
| CNN gaze regression | **Server** | The only step that benefits from a GPU |
| 3D gaze ray → screen intersection | **Server** | Reuses the calibration held server-side |

Only the normalized crops and head-pose metadata cross the network — compressed with
`zlib` and framed with a 4-byte length header — instead of full video frames.

---

## Pipeline

```
                    ┌──────────────── CLIENT ────────────────┐
   Webcam /         │  MediaPipe Face Mesh                   │
   RealSense  ────► │      ↓                                 │
                    │  solvePnP → head pose (rvec, tvec)     │
                    │      ↓                                 │
                    │  MPIIFaceGaze normalization            │
                    │  → face crop + left/right eye crops    │
                    └──────────────────┬─────────────────────┘
                                       │  TCP  (pickle + zlib)
                    ┌──────────────────▼──── SERVER ─────────┐
                    │  CNN (full-face + two eyes, SE blocks) │
                    │      ↓                                 │
                    │  (pitch, yaw) → 3D gaze vector         │
                    │      ↓                                 │
                    │  ray–plane intersection with screen    │
                    │      ↓                                 │
                    │  gaze point in screen pixels           │
                    └────────────────────────────────────────┘
```

**Geometry.** Facial landmarks are lifted into the camera coordinate system using the
intrinsics from calibration. The regressed `(pitch, yaw)` is converted to a 3D gaze vector,
rotated back out of the normalized space, and intersected with the plane of the monitor.
The intersection is then mapped to pixel coordinates using the physical screen dimensions,
so the output is a concrete point on the display rather than an abstract direction.

**Model.** A multi-stream CNN that consumes the full-face crop together with both eye
crops. Squeeze-and-Excitation blocks re-weight the channels of each stream, and a
learnable per-subject bias absorbs the systematic offset between individuals — the
component that personalization would otherwise have to estimate at calibration time.

---

## Repository layout

```
local/                            client side — capture and preprocessing
├── main_hybrid.py                entry point: camera loop, normalization, streaming
├── mpii_face_gaze_preprocessing  MPIIFaceGaze-style face/eye normalization
├── utils.py                      camera matrix, landmark lifting, ray-plane math
├── webcam.py                     webcam / RealSense capture source
├── visualization.py              3D scene and gaze overlay
└── calibration_matrix.yaml       camera intrinsics produced by calibration

server/                           server side — inference and screen mapping
├── servercheck_hybrid.py         entry point: socket server + gaze-point inference
├── servercheck_iTracker.py       same protocol, iTracker-style model
├── model.py                      multi-stream CNN with SE blocks
├── camera_calibration.py         chessboard intrinsic calibration
└── requirements1.txt             server environment

eye_tracking_for_everyone/        standalone iTracker baseline (Haar cascade based)
```

---

## Getting started

### 1. Environment

The client and the server are separate environments; install each on its own machine.

```bash
pip install -r requirements1.txt
```

The client additionally needs `pyrealsense2` and `screeninfo` if you use an Intel RealSense
camera or automatic monitor detection.

### 2. Camera calibration

Intrinsics are camera-specific and must be measured once. Print a chessboard pattern and run:

```bash
python camera_calibration.py
```

This writes `calibration_matrix.yaml`, which the client loads at startup.

### 3. Screen geometry

Set the physical size and resolution of your display in `servercheck_hybrid.py`.
Presets for common panels are included:

```python
monitor_mm     = (597.7, 336.2)   # 27" FHD
monitor_pixels = (1920, 1080)
```

Getting this wrong shifts the predicted gaze point systematically, so it is worth measuring
the visible panel area rather than trusting the nominal diagonal.

### 4. Run

Start the server first, then the client:

```bash
# on the GPU machine
python servercheck_hybrid.py

# on the machine with the camera
python main_hybrid.py
```

The server listens on port `4444` by default and prints the estimated gaze point per frame.

---

## Acknowledgements

The preprocessing and data conventions follow two open-source projects, and this
implementation reuses their normalization procedure:

- [GazeCapture / Eye Tracking for Everyone](https://github.com/CSAILVision/GazeCapture) — CSAIL
- [Efficiency in real-time webcam gaze tracking](https://github.com/pperle/gaze-tracking-pipeline) — pperle
