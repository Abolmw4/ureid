# UReID — Person Re-Identification Service

A Python service that performs **person re-identification (Re-ID)** on video streams. It detects people in each frame with **YOLOv8**, validates each detection with **MediaPipe pose landmarks**, extracts appearance features with a **TorchScript Re-ID network**, assigns a stable ID to every tracked person via a **cosine-similarity gallery**, and writes an annotated output video — all split across two parallel processes connected by a queue.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Logging](#logging)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Person detection** per frame using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`classes=[0]`, i.e. persons only).
- **Detection quality gating** with [MediaPipe Pose](https://developers.google.com/mediapipe): a person is only tracked when all six key body landmarks (shoulders, hips, ankles) are visible above a configurable threshold — partial/occluded detections are labeled `Unknown`.
- **Appearance-based re-identification** using a TorchScript Re-ID model (`reidNet.pt`) producing feature embeddings of cropped person images.
- **ID assignment** against an in-memory gallery using **cosine similarity** with a configurable threshold; gallery entries expire automatically after a configurable idle time.
- **Multiprocessing pipeline**: detection/Re-ID and video writing run as separate `multiprocessing.Process` instances communicating through a shared `Queue` (frames are serialized with `pickle`).
- **Annotated output video** with bounding boxes and assigned IDs drawn on each frame.
- **Structured logging** to both console and `logs.log`, including class and function names.

---

## Architecture

```
                ┌─────────────────────────────────────────────┐
 input video ──▶│  Ureid (Process 1)                          │
                │  1. YOLOv8 person detection                 │
                │  2. MediaPipe pose validation (good/bad)    │
                │  3. ReID feature extraction (reidNet.pt)    │
                │  4. Cosine-similarity matching vs. gallery  │
                │  5. ID assignment + gallery expiry          │
                └──────────────┬──────────────────────────────┘
                               │ multiprocessing.Queue
                               │ (pickled {id, frame, cordinate})
                               ▼
                ┌─────────────────────────────────────────────┐
 output video ◀─│  WriteVideo (Process 2)                     │
                │  - Draw bounding box + ID per person        │
                │  - Write annotated frames (XVID .avi)       │
                └─────────────────────────────────────────────┘
```

---

## Project Structure

```
ureid/
├── main.py                              # Entry point: wires up both processes
├── configs/
│   └── config_service.json              # Runtime configuration
├── reid_service/
│   ├── reidentification_service.py      # Ureid & WriteVideo processes
│   ├── reidentification_util.py         # Logger + JSON config loader
│   └── data/
│       ├── <input video>.mp4            # Input video
│       └── output1.avi                  # Annotated output video
├── weights/
│   ├── reidNet.pt                       # TorchScript Re-ID model
│   └── yolov8s.pt                       # YOLOv8 person detector
└── logs.log                             # Runtime log output
```

---

## Requirements

- **Python 3.8+**
- Dependencies:
  - `torch`, `torchvision`
  - `ultralytics` (YOLOv8)
  - `mediapipe`
  - `opencv-python`
  - `numpy`
- Model weights (not included — place them in `weights/`):
  - `weights/yolov8s.pt` — [YOLOv8 small](https://github.com/ultralytics/assets/releases) weights
  - `weights/reidNet.pt` — a TorchScript-exported person Re-ID model accepting `1×3×256×128` inputs and returning an embedding vector

CPU is supported out of the box; CUDA is used automatically when `device` is set to `"cuda"` and a GPU is available.

---

## Installation

```bash
git clone https://github.com/<your-username>/ureid.git
cd ureid

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch torchvision ultralytics mediapipe opencv-python numpy
```

Then download/place the model weights:

```bash
mkdir -p weights
# YOLOv8s:
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt -O weights/yolov8s.pt
# Copy your TorchScript Re-ID model:
cp /path/to/reidNet.pt weights/reidNet.pt
```

---

## Configuration

All runtime parameters live in `configs/config_service.json`:

| Key | Description | Default |
|---|---|---|
| `video_format` | Accepted input video extensions | `["mp4", "avi"]` |
| `visibility_score_for_body_landmarks` | Minimum MediaPipe landmark visibility for a "good" person | `0.65` |
| `item_elapsed_gallery` | Minutes of inactivity before a gallery entry expires | `3` |
| `similarity_threshold` | Cosine-similarity (%) required to match an existing ID | `70.00` |
| `fps` | Frame rate of the annotated output video | `90` |
| `reid_model` | Path to the TorchScript Re-ID model | `weights/reidNet.pt` |
| `yolo_model` | Path to the YOLOv8 weights | `weights/yolov8s.pt` |
| `vidoe_src` | Path to the input video (or a directory; falls back to webcam `0`) | `reid_service/data/....mp4` |

---

## Usage

1. Put your input video at the path configured in `vidoe_src`.
2. Run the service:

```bash
python main.py
```

3. The annotated result is written to `reid_service/data/output1.avi`, with each detected person boxed and labeled with their assigned ID (or `Unknown` for detections that fail the pose-visibility check).

---

## How It Works

1. **Detection** — Each frame is passed through YOLOv8; only person detections are kept.
2. **Quality gate** — Each cropped person is checked with MediaPipe Pose. If landmarks `[11, 12, 23, 24, 27, 28]` (shoulders, hips, ankles) all exceed `visibility_score_for_body_landmarks`, the crop is considered a "good person"; otherwise it is forwarded as `Unknown`.
3. **Embedding** — Good crops are resized to `256×128`, normalized with ImageNet statistics, and embedded by the Re-ID network.
4. **Matching** — The embedding is compared against every gallery entry with cosine similarity. If the best match exceeds `similarity_threshold`, the existing ID is reused and the entry's timestamp refreshed; otherwise a new ID is created and added to the gallery.
5. **Expiry** — Gallery entries idle longer than `item_elapsed_gallery` minutes are removed, so IDs can be recycled for genuinely new people.
6. **Output** — The `WriteVideo` process consumes `{id, frame, cordinate}` messages from the queue, draws the bounding box and ID, and writes the frame to disk.

---

## Logging

Logs are emitted at `DEBUG` level to **both** the console and `logs.log`, in the format:

```
2024-10-11 09:59:47,136 | reid_service.reidentification_util | DEBUG => [Ureid : assign_id] = debug->Query_id:4 == Gallery_id:2
```

Each record includes the class and function that produced it, which makes tracing the multiprocessing pipeline straightforward.

---

## Roadmap

- [ ] Make output video path configurable
- [ ] Support multi-camera / RTSP input sources
- [ ] Persist gallery across runs (re-identify people across sessions)
- [ ] Add `requirements.txt` / `pyproject.toml`
- [ ] Add unit tests and CI

---

## License

This project is currently unlicensed. Add a `LICENSE` file (e.g. MIT, Apache-2.0) if you plan to distribute it.
