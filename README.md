# depth_estimation

Depth and range estimation from cameras, in three independent pieces:

| directory | what it does |
|---|---|
| [`monoculer_camera_yolo/`](monoculer_camera_yolo/) | **Monocular drone detection + metric range.** Detects drones with a Roboflow YOLO model, tracks them, and turns each bounding box into a distance using an assumed target size. Runs on CPU, GPU, or a Hailo accelerator |
| [`monocular_depth_anything/`](monocular_depth_anything/) | **Monocular depth maps** via Depth Anything V2, with optional metric scaling |
| [`stereo/`](stereo/) + top-level scripts | **Stereo depth**: calibration, rectification, block matching and HITNET, triangulation to metric range |

Each has its own `SETUP.md` and its own virtualenv. They do not import from one
another, so install only the one you need.

## Quick start

```bash
git clone https://github.com/BHavikJ12/depth_estimation.git
cd depth_estimation
```

Then pick a project:

```bash
cd monoculer_camera_yolo && ./setup.sh     # see its SETUP.md
cd monocular_depth_anything && ./setup.sh  # see its SETUP.md
```

Full instructions:

* [monoculer_camera_yolo/SETUP.md](monoculer_camera_yolo/SETUP.md) — laptop or desktop
* [monoculer_camera_yolo/SETUP_RPI_HAILO.md](monoculer_camera_yolo/SETUP_RPI_HAILO.md) — Raspberry Pi 5 with a Hailo AI HAT
* [monocular_depth_anything/SETUP.md](monocular_depth_anything/SETUP.md)
* [HITNET.md](HITNET.md) — the stereo pipeline

## What is deliberately not in this repo

Three kinds of thing are excluded, because they are large, machine-specific, or
belong to someone else. Everything here tells you how to get them back.

### Virtualenvs (`venv/`)

Not committed and not copyable. A venv is gigabytes, it is built for one CPU
architecture, and every script inside hard-codes an absolute path, so it breaks
the moment the folder is moved or renamed. Each project rebuilds one:

```bash
./setup.sh          # creates venv/, installs deps, runs the test suites
```

### Model weights

None are committed. Each project fetches its own:

| project | weights | how |
|---|---|---|
| `monoculer_camera_yolo` | Roboflow [`drone-detection-rchy7`](https://universe.roboflow.com/drone-detection-ssbrv/drone-detection-rchy7) (~7 MB ONNX) | downloaded on first run into `~/.cache/roboflow-onnx/`. Needs a free API key from [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api) |
| `monocular_depth_anything` | [Depth Anything V2](https://huggingface.co/depth-anything) (~100 MB) | downloaded from Hugging Face on first run |
| stereo / HITNET | HITNET ONNX models | `./download_hitnet_models.sh` |

### Third-party repositories

Cloned separately rather than vendored, so they stay on their own upstreams:

```bash
git clone https://github.com/NVlabs/Fast-FoundationStereo.git
git clone https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation.git
git clone https://github.com/TemugeB/manual_align_depth_to_color.git
```

### Test footage and outputs

`testdata/`, `outputs/` and any `*.mp4` are ignored — they are hundreds of
megabytes of clips. Both `monoculer_camera_yolo` and `monocular_depth_anything`
ship test suites that generate their own synthetic footage, so `./setup.sh`
verifies a fresh clone with no video files present at all.

## Verifying a clone works

Neither test suite needs an API key, a camera, a network connection, or any
downloaded weights:

```bash
cd monoculer_camera_yolo
./venv/bin/python selftest.py        # geometry, tracker, ONNX decode, HUD
./venv/bin/python pipeline_test.py   # the real CLI over a synthetic clip
```
