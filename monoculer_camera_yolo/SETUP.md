# Installing monoculer_camera_yolo on another machine

From nothing to a working live detection window. Reference machine is Ubuntu
22.04 with Python 3.10.12; notes for macOS and Windows are at the end.

Total: ~420 MB of disk, one 7 MB model download, about five minutes.

---

## 0. What you have to do

Four things, in order. Nothing else is required.

| # | step | needs internet? |
|---|---|---|
| 1 | install system packages (`apt`) | yes |
| 2 | `git clone` the repo (~315 KB) | yes |
| 3 | `./setup.sh` — builds the venv, runs the tests | yes (pip) |
| 4 | get the model weights (~7 MB): download, or copy a cache | only for the download |

Steps 2-4 each have an offline alternative if the machine has no network; they
are noted in place.

---

## 1. What the target machine needs

| | |
|---|---|
| **Python** | 3.10 or newer. numpy, requests and onnxruntime all require it |
| **Disk** | ~420 MB (venv 411 MB, source 132 KB, model cache 7 MB per model version) |
| **Network** | once, to install packages and fetch the model weights. Offline afterwards |
| **GPU** | not required. It runs 27 fps on CPU at 720p |
| **Roboflow API key** | free, from https://app.roboflow.com/settings/api |

No CUDA, no docker, no torch.

### System packages (Ubuntu/Debian)

The OpenCV wheel bundles most of its own libraries but not all of them. On a
fresh install you need:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip \
                    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
                    libxcb1 libx11-6 libxkbcommon-x11-0
```

`python3-venv` is the one people forget — without it `python3 -m venv` fails
with a confusing "ensurepip is not available" message. The `libgl1` / `libglib`
group is what `import cv2` needs; the `libx*` / `libxcb` group is what the live
display window needs. On a headless server you can skip the second group and
always pass `--no-display`.

Check before going further:

```bash
python3 --version        # must be 3.10+
```

If it is older, install a newer Python (`sudo apt install python3.11 python3.11-venv`)
and use `python3.11 -m venv` in step 3.

---

## 2. Get the code

```bash
git clone https://github.com/BHavikJ12/depth_estimation.git
cd depth_estimation/monoculer_camera_yolo
```

That is about 315 KB for the whole repository. The virtualenv, the model
weights and the test clips are deliberately not in it — see the repo
[README](../README.md) for where each comes from. All three are rebuilt or
re-fetched by the steps below, so a fresh clone is all you need.

> **Never copy a `venv/` between machines**, even between two identical
> laptops. It is built for one CPU architecture and every script inside
> hard-codes an absolute path, so it breaks as soon as the folder moves — a
> renamed folder is enough to produce `bad interpreter` from `./venv/bin/pip`.
> Rebuild it with `./setup.sh` instead; it takes about a minute.

### No network on the target machine?

Clone on a machine that has one and carry the result over:

```bash
git clone https://github.com/BHavikJ12/depth_estimation.git
tar czf depth_estimation.tar.gz --exclude=.git depth_estimation
# then scp / USB stick, and on the far side:
tar xzf depth_estimation.tar.gz && cd depth_estimation/monoculer_camera_yolo
```

You will still need pip packages and the model weights; section 4 covers moving
the weights cache (which then needs no key at all), and
`pip download -r requirements.txt` builds a wheel bundle.

### What should have arrived

```
camera.py  detector.py  onnx_model.py  hailo_model.py  overlay.py
ranging.py  sources.py  tracker.py  video_writer.py  run.py
selftest.py  pipeline_test.py  requirements.txt  requirements.lock.txt
setup.sh  README.md  SETUP.md  SETUP_RPI_HAILO.md
```

## 3. Build the environment

```bash
cd ~/monoculer_camera_yolo
./setup.sh
```

That creates `venv/`, installs the four dependencies, and runs both test suites.
It should end with `all checks passed` and `pipeline ok`.

If you would rather do it by hand, or `setup.sh` will not run:

```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip wheel
./venv/bin/pip install -r requirements.txt
./venv/bin/python selftest.py
./venv/bin/python pipeline_test.py
```

### Reproducing exact versions

`requirements.txt` takes the newest compatible releases, which is usually what
you want. To match this machine exactly instead:

```bash
./venv/bin/pip install -r requirements.lock.txt
```

The reference set is opencv-contrib-python 5.0.0.93, numpy 2.2.6,
requests 2.34.2, onnxruntime 1.23.2 on Python 3.10.12.

---

## 4. The model weights

The repo contains no weights. You need the ~7 MB ONNX on the machine before it
can detect anything, and there are two ways to get it.

### A. Let it download itself (needs a key, once)

```bash
export ROBOFLOW_API_KEY=xxxxxxxxxxxx
echo 'export ROBOFLOW_API_KEY=xxxxxxxxxxxx' >> ~/.bashrc   # survives new shells
```

Free key from <https://app.roboflow.com/settings/api>. The first run fetches the
weights into `~/.cache/roboflow-onnx/` and everything after that is local.

### B. Copy the cache from a machine that already has it (no key, no internet)

```bash
# on the working machine
tar czf roboflow-cache.tar.gz -C ~/.cache roboflow-onnx
# on the target machine
tar xzf roboflow-cache.tar.gz -C ~/.cache
```

**Once the cache is in place the key is not needed at all** — not at startup,
not ever. The key is required only to download something not already cached, and
the error message says exactly that when it happens.

You will still need the pip packages, so either network access for step 3 or a
wheel bundle built with `pip download -r requirements.txt`.

`ROBOFLOW_ONNX_CACHE=/some/path` moves the cache elsewhere, useful for a larger
disk or for testing a clean download.

---

## 5. Verify it works

```bash
./venv/bin/python selftest.py        # geometry, tracker, ONNX decode, HUD
./venv/bin/python pipeline_test.py   # the whole CLI over a synthetic clip
```

Both run with no key, no camera and no network. If they pass, everything except
the model download is proven.

Then a real run — this one also confirms the key and the weights download:

```bash
./venv/bin/python run.py --source 0 \
    --model-id drone-detection-rchy7/8 --keep-classes 1 \
    --conf 0.40 --hfov 70 --target-size 0.5
```

A window should open on your webcam with a telemetry block in the corner. `q`
quits. First run pauses a few seconds while it fetches the 7 MB model.

See [README.md](README.md) for the full set of sources — video files, streams,
images, folders — and for calibrating the range.

---

## 6. Optional: run on the GPU

Only worth it if the CPU is not keeping up; 27 fps at 720p is usually plenty.

```bash
./venv/bin/pip uninstall -y onnxruntime
./venv/bin/pip install onnxruntime-gpu
```

It is a drop-in replacement — [onnx_model.py](onnx_model.py) picks up
`CUDAExecutionProvider` automatically when it is available, and the startup line
tells you which one it chose:

```
detector backend=onnx  ...  onnx 640x640 stretch yolov5v6n on CUDA
```

If it still says `on CPU`, the CUDA/cuDNN runtime versions do not match what
that onnxruntime build wants. The `onnxruntime-gpu[cuda,cudnn]` extras pull
matching runtimes as pip wheels, which avoids having to align the system CUDA
install.

---

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| `bad interpreter: .../venv/bin/python3` from `./venv/bin/pip` | the project folder was renamed or moved. A venv hard-codes its own path into every script's shebang. Either recreate it, or `sed -i 's\|/old/path/venv\|/new/path/venv\|g' venv/bin/*`. `./venv/bin/python -m pip` works either way |
| `ensurepip is not available` | `sudo apt install python3-venv` |
| `ImportError: libGL.so.1: cannot open shared object file` | `sudo apt install libgl1 libglib2.0-0` |
| pip fails with `Name or service not known` naming `pypi.ngc.nvidia.com` | an NVIDIA PyIndex `extra-index-url` in `~/.config/pip/pip.conf`. Prefix every pip call with `PIP_CONFIG_FILE=/dev/null`; `setup.sh` already does. `--index-url` does **not** help — the extra index comes from the config file and survives it |
| `error: No Roboflow API key` | `export ROBOFLOW_API_KEY=...` |
| `Roboflow rejected the API key (401)` | wrong key, or the key's workspace cannot see the model |
| no window appears, no error | on Wayland, prefix with `QT_QPA_PLATFORM=xcb`. On a headless box use `--no-display --save out.mp4` |
| `cannot open source '0'` | no camera at that index. `ls /dev/video*`, try `--source 1`. In Docker/WSL pass the device through |
| `cv2.error` on `imshow` | an OpenCV built without GUI support (usually `opencv-python-headless` got installed). `pip uninstall opencv-python-headless` and reinstall `opencv-contrib-python` |
| very slow, a few fps | check the startup line says `on CUDA` if you installed the GPU runtime, or use `--stride 3` |
| detections look wrong or absent | not an install problem. See "What this model actually does, measured" in [README.md](README.md) — recall depends heavily on the footage |

---

## 8. Raspberry Pi with a Hailo accelerator

See [SETUP_RPI_HAILO.md](SETUP_RPI_HAILO.md). The short version: the Pi runs
everything here on its CPU at 3-6 fps with no extra work, and the Hailo path
needs the model recompiled to `.hef` on an x86 Linux machine first.

## 9. macOS and Windows

Neither is tested, but nothing here is Linux-specific — it is Python, OpenCV and
onnxruntime, all of which ship wheels for both.

**macOS:** skip the `apt` step entirely; the OpenCV wheel is self-contained.
`python3 -m venv venv` then the same commands. On Apple Silicon, onnxruntime
runs on CPU fine; there is no CUDA, so ignore section 6.

**Windows:** skip the `apt` step. The venv paths differ:

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python run.py --source 0 ...
```

`setup.sh` is bash, so run those four commands by hand or use WSL. Under WSL a
webcam needs USB passthrough, so prefer video files there.

---

## 10. Uninstalling

Everything lives in two places:

```bash
rm -rf ~/monoculer_camera_yolo          # the project and its venv
rm -rf ~/.cache/roboflow-onnx     # the downloaded model weights
```

Nothing is installed system-wide, and nothing is written outside those two paths
except the `export` line if you added one to `~/.bashrc`.
