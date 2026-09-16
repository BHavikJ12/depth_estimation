# Installing on a Raspberry Pi with a Hailo AI HAT

For a Raspberry Pi 5 with the AI Kit (Hailo-8L, 13 TOPS) or AI HAT+ (Hailo-8,
26 TOPS). For an ordinary laptop or desktop see [SETUP.md](SETUP.md).

**Read section 0 before buying time on this.** The Hailo path is not a matter of
installing a package: the model has to be recompiled on a separate x86 machine
before the Pi can run it at all, and there is a CPU-only fallback that may be
good enough.

---

## 0. What you are signing up for

The accelerator cannot run ONNX. It runs `.hef`, a Hailo-specific compiled
format, and **the compiler only runs on x86_64 Linux** — not on the Pi itself.
So the work splits across two machines:

```
  your laptop (x86_64 Ubuntu)              the Raspberry Pi 5 + Hailo
  ───────────────────────────              ──────────────────────────
  weights.onnx                             hailo-all  (apt)
      │  Hailo Dataflow Compiler           this project + venv
      │  (free account, ~30 min)                  │
      ▼                                           ▼
  drone.hef  ──────── scp ────────────────►  run.py --backend hailo
```

Your laptop is Ubuntu 22.04 x86_64, so it can do the compile — though with 15 GB
of RAM you are under Hailo's recommended 32 GB and the quantisation step may need
a smaller calibration set (section 3).

### Consider skipping Hailo first

Everything in this project runs on the Pi's CPU with no Hailo involvement, using
the same `--backend onnx` as your laptop. Expect roughly **3-6 fps** at 640×640
on a Pi 5, against maybe **30+ fps** through the Hailo. If you are processing
recorded clips, or tracking a target that does not move much between frames,
`--backend onnx --stride 3` on the CPU may be all you need, and it is a ten
minute install instead of an afternoon.

Sections 1-2 get you that. Sections 3-5 add the accelerator.

---

## 1. Pi-side prerequisites

* Raspberry Pi 5 (the AI HAT is PCIe; a Pi 4 needs a different M.2 adapter and
  is not covered here)
* Raspberry Pi OS **Bookworm 64-bit**, kernel **> 6.6.31**
* The HAT seated on the PCIe connector, Pi powered off while fitting

```bash
sudo apt update && sudo apt full-upgrade
sudo rpi-eeprom-update -a          # firmware, then reboot
sudo reboot
uname -a                           # confirm kernel > 6.6.31
```

Set PCIe to Gen 3 — the HAT runs at half bandwidth otherwise:

```bash
sudo raspi-config
#   6 Advanced Options  ->  A8 PCIe Speed  ->  Yes (Gen 3)
sudo reboot
```

---

## 2. The project on the Pi (CPU path, works immediately)

```bash
sudo apt install -y python3-venv python3-pip \
                    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
                    libxcb1 libx11-6 libxkbcommon-x11-0 ffmpeg
```

`ffmpeg` matters: without it the annotated video falls back to MPEG-4 Part 2,
which many players show as a black frame.

Get the code straight from GitHub — the Pi needs nothing from your laptop yet:

```bash
git clone https://github.com/BHavikJ12/depth_estimation.git
cd depth_estimation/monoculer_camera_yolo
python3 -m venv venv --system-site-packages     # see the note below
./venv/bin/pip install -r requirements.txt
./venv/bin/python selftest.py
./venv/bin/python pipeline_test.py
```

The repo is ~315 KB and carries no weights, no venv and no test clips, so this
is quick even on a slow link. Do **not** copy a venv over from the laptop: it is
built for x86_64 and will not run on the Pi's ARM at all.

> **`--system-site-packages` is not optional here.** The Hailo Python bindings
> are installed system-wide by apt and cannot be pip-installed into an isolated
> venv. Build the venv without that flag and `import hailo_platform` fails with
> no obvious reason. It is harmless on the CPU-only path, so use it either way.

Set the key and check it runs:

```bash
echo 'export ROBOFLOW_API_KEY=xxxxxxxx' >> ~/.bashrc && source ~/.bashrc
./venv/bin/python run.py --source 0 \
    --model-id drone-detection-rchy7/8 --keep-classes 1 \
    --conf 0.40 --hfov 70 --no-display --max-frames 60
```

`--source 0` is a USB camera. Point it at a video file instead if you copied one
over; the repo ships no test clips.

On `aarch64`, pip builds some wheels from source; the first install is slower
than on a laptop. If `opencv-contrib-python` has no aarch64 wheel for your
Python version, `sudo apt install python3-opencv` and drop OpenCV from
`requirements.txt`.

At this point everything works, on the CPU. What follows makes it fast.

---

## 3. Compile the model to `.hef` (on your laptop, not the Pi)

### 3.1 Get the ONNX

Already cached by the first run on your laptop:

```bash
ls ~/.cache/roboflow-onnx/drone-detection-rchy7--8/weights.onnx
```

That is a YOLOv5-v6n, 640×640, 3 classes, "stretch to" resize — the architecture
Hailo's model zoo has a ready-made configuration for, which is the easy case.

### 3.2 Install the Dataflow Compiler

Register at the [Hailo Developer Zone](https://hailo.ai/developer-zone/) (free)
and download the **Dataflow Compiler** wheel for Linux x86_64, plus the **Hailo
Model Zoo**. Match the DFC version to the HailoRT version the Pi ends up with
(`hailortcli fw-control identify` reports it) — a HEF compiled by a newer DFC
than the runtime will refuse to load.

```bash
python3 -m venv ~/hailo-dfc && source ~/hailo-dfc/bin/activate
pip install hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl
pip install hailo_model_zoo-<version>-py3-none-any.whl
hailo --version
```

### 3.3 Build a calibration set

Quantisation to int8 needs a few hundred representative images — and
*representative* is the operative word. Calibrate on stock COCO photos and the
quantised model will be worse on your footage than the numbers suggest. Use your
own clips: put them in a `testdata/` folder (the repo ships none) and run:

```bash
cd path/to/depth_estimation/monoculer_camera_yolo
mkdir -p /tmp/calib
./venv/bin/python - <<'EOF'
import cv2 as cv, glob, os
os.makedirs("/tmp/calib", exist_ok=True)
n = 0
for p in sorted(glob.glob("testdata/t*.*")):
    c = cv.VideoCapture(p); total = int(c.get(cv.CAP_PROP_FRAME_COUNT))
    for i in range(0, total, max(1, total // 40)):
        c.set(cv.CAP_PROP_POS_FRAMES, i); ok, f = c.read()
        if ok:
            cv.imwrite(f"/tmp/calib/{n:04d}.jpg", cv.resize(f, (640, 640)))
            n += 1
    c.release()
print(n, "calibration images")
EOF
```

Aim for 256-1024 images spread across backgrounds — sky, trees, ground — since
that is exactly where this model's behaviour changes.

### 3.4 Compile

```bash
source ~/hailo-dfc/bin/activate
hailomz compile yolov5n \
    --ckpt ~/.cache/roboflow-onnx/drone-detection-rchy7--8/weights.onnx \
    --hw-arch hailo8l \
    --calib-path /tmp/calib \
    --classes 3
```

`--hw-arch` is `hailo8l` for the AI Kit and `hailo8` for the AI HAT+; a HEF built
for one will not load on the other. Output lands as `yolov5n.hef`.

If quantisation is killed for memory on 15 GB, cut the calibration set to ~256
images and close everything else. If `hailomz` cannot match the ONNX to the
`yolov5n` config, the node names differ from the model zoo's expectation and you
need the manual three-step route (`hailo parser onnx` → `hailo optimize` →
`hailo compiler`) with explicit `--start-node-names` / `--end-node-names` taken
from the graph — Netron will show you the names.

Copy it over:

```bash
scp yolov5n.hef pi@raspberrypi.local:~/monoculer_camera_yolo/drone.hef
```

---

## 4. HailoRT on the Pi

```bash
sudo apt install -y hailo-all
sudo reboot
```

That one meta-package pulls the PCIe kernel module (via DKMS), the HailoRT
runtime and `hailortcli`, the TAPPAS GStreamer plugins, and the system-wide
`hailort` Python bindings.

Verify the device before blaming the model for anything:

```bash
hailortcli fw-control identify
#   Device Architecture: HAILO8L      <- or HAILO8
#   Firmware Version: 4.x.y
lspci | grep -i hailo
./venv/bin/python -c "import hailo_platform; print(hailo_platform.__version__)"
```

If the import fails but `hailortcli` works, the venv was built without
`--system-site-packages`. Rebuild it:

```bash
rm -rf venv && python3 -m venv venv --system-site-packages
./venv/bin/pip install -r requirements.txt
```

Measure the raw device throughput, independent of this project:

```bash
hailortcli run drone.hef
```

---

## 5. Run it on the accelerator

```bash
./venv/bin/python run.py --source 0 \
    --backend hailo --weights drone.hef \
    --keep-classes 1 --class-names 0,drone,bird \
    --conf 0.40 --hfov 70 --target-size 0.5
```

A `.hef` passed to `--weights` selects the Hailo backend on its own, so
`--backend hailo` is belt-and-braces. No `ROBOFLOW_API_KEY` is needed on this
path — the weights are local and nothing is fetched.

The startup line confirms what is actually running:

```
detector backend=hailo  ...  hailo 640x640 stretch drone.hef, classes=['0', 'drone', 'bird']
```

Everything else — sources, ranging, tracking, CSV, video output — is unchanged;
see [README.md](README.md).

### Headless

The Pi usually has no display. Use `--no-display` with `--save`:

```bash
./venv/bin/python run.py --source 0 --backend hailo --weights drone.hef \
    --keep-classes 1 --conf 0.40 --hfov 70 \
    --no-display --save out.mp4 --save-csv track.csv --max-frames 900
```

### The Pi camera

`--source 0` expects a V4L2 device. A CSI camera via `libcamera` may not appear
as one; check `libcamera-hello --list-cameras` and `ls /dev/video*`. If it does
not, either use a USB camera or bridge the CSI camera to V4L2 with
`rpicam-vid ... -o - | ...`. USB is the path of least resistance.

---

## 6. Honest status of this backend

**[hailo_model.py](hailo_model.py) has never been run against real hardware.**
There is no Hailo device on the machine it was written on. What *is* verified,
offline in `selftest.py`:

* the NMS output format unpacks to the right frame pixels, class index and score
* the confidence threshold is applied
* an NMS output is told apart from a raw tensor output
* empty classes do not crash it

What is **not** verified: that the HailoRT calls (`VDevice`, `ConfigureParams`,
`InferVStreams`) are correct against your installed version, and that a HEF
compiled by the route in section 3 produces the output layout assumed here.
HailoRT's Python API has changed across releases.

The likely failure is the output format. The code handles both a HEF with NMS
compiled in (per-class lists of normalised boxes) and a raw YOLO head, and picks
between them by inspecting what comes back. If boxes land in the wrong place or
nothing is detected while `hailortcli run drone.hef` reports fine throughput,
that decision is where to look first — print `results[self.out_infos[0].name]`
in `__call__` and compare its shape against the two branches.

---

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| `hailortcli` not found | `sudo apt install hailo-all`, then reboot |
| `fw-control identify` fails | HAT not seated, or PCIe not enabled. Check `lspci \| grep -i hailo` |
| `import hailo_platform` fails, CLI works | venv built without `--system-site-packages`; rebuild it |
| `HEF version mismatch` on load | DFC and HailoRT versions differ. Recompile with a DFC matching `fw-control identify` |
| HEF loads but detects nothing | wrong `--hw-arch` at compile time, or the output-format branch — see section 6 |
| boxes in the wrong place | the compiled resize mode differs from what the code assumes. The Roboflow model is "stretch to"; if the HEF letterboxes, `HailoDetectionModel(..., letterbox=True)` |
| memory error during `hailomz compile` | shrink the calibration set to ~256 images; 15 GB is under Hailo's recommended 32 GB |
| Killed / OOM on the Pi | add swap, or use `--max-side 640` to cut decode memory |
| `libgomp` allocation error | `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` in `~/.bashrc` |
| output video plays black | `sudo apt install ffmpeg`; without it the writer falls back to MPEG-4 Part 2 |
| detections look wrong, not absent | not an install problem — see "What this model actually does, measured" in [README.md](README.md) |

---

## 8. A warning that outlives the install

Quantising to int8 costs accuracy, and this model has very little to spare. It
was measured on your own clips as unreliable except for airborne targets against
sky, and at small sizes it fires on any dark blob. A quantised HEF will be
somewhat worse than that, not better.

So compare before trusting it: run the same clip through `--backend onnx` on the
laptop and `--backend hailo` on the Pi, and diff the CSVs. If the HEF loses
targets the ONNX finds, that is quantisation, and a larger or more
representative calibration set is the lever.

The accelerator makes a mediocre detector fast. It does not make it good.

---

## Sources

* [Raspberry Pi AI software documentation](https://www.raspberrypi.com/documentation/computers/ai.html)
* [hailo-ai/hailo-rpi5-examples install guide](https://github.com/hailo-ai/hailo-rpi5-examples/blob/main/doc/install-raspberry-pi5.md)
* [Hailo Developer Zone](https://hailo.ai/developer-zone/)
