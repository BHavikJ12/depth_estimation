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

### The checklist

| # | step | where | needed for |
|---|---|---|---|
| 1 | firmware, kernel, PCIe Gen 3 | Pi | both paths |
| 2 | `apt` system packages | Pi | both paths |
| 3 | `git clone` + `./setup.sh` | Pi | both paths |
| 4 | model weights: download or copy a cache | Pi | **CPU path only** |
| 5 | install the Dataflow Compiler | laptop | Hailo only |
| 6 | build a calibration set from your own clips | laptop | Hailo only |
| 7 | compile ONNX → `.hef`, `scp` it to the Pi | laptop | Hailo only |
| 8 | `apt install hailo-all` | Pi | Hailo only |

Steps 1-4 give you a working CPU install. 5-8 add the accelerator. **The Hailo
path needs no Roboflow key and no weights download at all** — the `.hef` is a
local file you built, so step 4 is skippable if you are going straight to Hailo.

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

### The model weights

The CPU path needs the ~7 MB ONNX, and there are two ways to get it onto the Pi:

```bash
# A. let it download itself on first run (needs a key and internet, once)
echo 'export ROBOFLOW_API_KEY=xxxxxxxx' >> ~/.bashrc && source ~/.bashrc
#    ...then the first run of run.py fetches it into ~/.cache/roboflow-onnx/

# B. or copy the cache from a machine that has already run it — no key needed
#    on the laptop:
tar czf roboflow-cache.tar.gz -C ~/.cache roboflow-onnx
scp roboflow-cache.tar.gz pi@raspberrypi.local:~/
#    on the Pi:
tar xzf roboflow-cache.tar.gz -C ~/.cache
```

With the cache in place the Pi runs fully offline and never asks for a key.

Check it runs:

```bash
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

`--hw-arch` is `hailo8l` for the AI Kit and `hailo8` for the AI HAT+; a HEF built
for one will not load on the other.

**Model Zoo 5.4.0 has no plain `yolov5n` detection config** — only
`yolov5n_seg*` (segmentation). Run `hailomz compile --help` and check the
`model_name` choices before assuming `yolov5n` exists in whatever version you
installed. Use `yolov5s` instead: YOLOv5n and YOLOv5s share the same head
architecture and default anchors (n/s/m/l/x all use the same anchors in the
stock architecture), differing only in channel width, so `yolov5s`'s config is
the right template even though the actual weights are the nano variant.

```bash
source ~/hailo-dfc/bin/activate
hailomz compile yolov5s \
    --ckpt ~/.cache/roboflow-onnx/drone-detection-rchy7--8/weights.onnx \
    --hw-arch hailo8 \
    --calib-path /tmp/calib \
    --classes 3
```

Three real problems showed up compiling the actual Roboflow export, in order:

**a) Dynamic input shape.** The ONNX reports its input as
`[batch, 3, height, width]` with every dim dynamic, and parsing fails with
`Could not parse the model due to dynamic shapes`. The `--tensor-shapes` flag
the error message suggests does not exist on `hailomz compile` in this
version (`unrecognized arguments`). Fix it on the ONNX itself first, with
`onnxsim` (already installed as a DFC dependency):

```bash
onnxsim ~/.cache/roboflow-onnx/drone-detection-rchy7--8/weights.onnx \
    /tmp/weights_fixed.onnx \
    --overwrite-input-shape images:1,3,640,640
```

Use `/tmp/weights_fixed.onnx` as `--ckpt` from here on, not the original file.

**b) Missing NMS config file in the wheel.** Compiling then fails later, during
`_handle_classes_argument`, with:

```
FileNotFoundError: .../hailo_model_zoo/cfg/postprocess_config/yolov5s_nms_config.json
```

That whole directory is absent from the `hailo_model_zoo-5.4.0` wheel — a
packaging gap, not something you did wrong. Pull the file from the matching
GitHub release tag instead of guessing its contents (it encodes real anchor
box values, wrong ones silently produce bad boxes, not an error):

```bash
PKGDIR=$(python -c "import hailo_model_zoo, os; print(os.path.dirname(hailo_model_zoo.__file__))")
mkdir -p "$PKGDIR/cfg/postprocess_config"
gh api "repos/hailo-ai/hailo_model_zoo/contents/hailo_model_zoo/cfg/postprocess_config/yolov5s_nms_config.json?ref=v5.4.0" \
    --jq '.content' | base64 -d > "$PKGDIR/cfg/postprocess_config/yolov5s_nms_config.json"
```

Match `?ref=` to whatever `hailomz --version` actually reports, not always
`v5.4.0`.

**c) Detection-head layer names don't match.** Compiling then fails with:

```
HailoNNException: The layer named conv63 doesn't exist in the HN
```

The config's `bbox_decoders` hardcode the reference `yolov5s`'s internal layer
names (`conv55`/`conv63`/`conv70`) for its three detection heads. Our model is
narrower (fewer parameters than the reference `yolov5s`), so its equivalent
heads land at different indices. Find the real ones from the `.har` that
`hailomz compile` saves on its first (failing) attempt:

```bash
source ~/hailo-dfc/bin/activate
python - <<'EOF'
from hailo_sdk_client import ClientRunner
runner = ClientRunner(har="yolov5s.har")
hn = runner.get_hn_model()
print("real output layers:", [l.name for l in hn.get_real_output_layers()])
EOF
```

That printed `conv47`, `conv54`, `conv60` here, in the same stride-8/16/32
order the reference names filled — the anchor `w`/`h` values stay the same,
only the `encoded_layer` field per decoder changes:

```bash
python - <<EOF
import json
path = "$PKGDIR/cfg/postprocess_config/yolov5s_nms_config.json"
with open(path) as f:
    cfg = json.load(f)
mapping = {"conv55": "conv47", "conv63": "conv54", "conv70": "conv60"}  # from your own .har, not these exact numbers
for d in cfg["bbox_decoders"]:
    old = d["encoded_layer"]
    d["encoded_layer"] = mapping[old]
with open(path, "w") as f:
    json.dump(cfg, f, indent=4)
EOF
```

Re-run the same `hailomz compile` command from above once all three are fixed.
It ends with `HEF file written to yolov5s.hef` — despite the model name in the
filename, this is your `yolov5v6n` weights, compiled.

If quantisation is killed for memory on 15 GB, cut the calibration set to ~256
images and close everything else.

Copy it over:

```bash
scp yolov5s.hef pi@raspberrypi.local:~/monoculer_camera_yolo/drone.hef
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
    --keep-classes 1 --class-names 'bird,drone,flock of birds' \
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

**Update: run against real Hailo-8 hardware (AI HAT+, HailoRT 4.23.0).** A HEF
compiled by the route in section 3 does load and run. One real bug turned up
in the process, now fixed:

`_ensure_open` created and entered the `InferVStreams` pipeline *before*
activating the network group. HailoRT requires the opposite order — the
network group must be active before an inference pipeline is built on top of
it. The symptom was every inference call raising
`HailoRTNetworkGroupNotActivatedException` /
`HAILO_NETWORK_GROUP_NOT_ACTIVATED(69)` immediately. `close()` had the matching
bug in reverse (deactivating before closing the pipeline that was still using
it). Both are now ordered activate → open pipeline, and close pipeline →
deactivate.

What is now verified against real hardware: HEF loading, network group
configuration and activation, and running a live inference call without the
HailoRT calls themselves raising. What is **still not verified**: the actual
output format and box positions. Section 3's `bbox_decoders` fix used the
stock COCO anchor values (see the layer-name-mismatch note in section 3.4) —
those weren't checked against this specific trained model's real anchors,
which auto-anchor training can shift away from the defaults. Before trusting
any range/position number this backend reports, compare its boxes against the
same footage run through `--backend onnx` (known-good) and look for a
systematic offset or scale error — that would point at the anchor values,
not a code bug.

What *was already* verified offline in `selftest.py`, before hardware access:

* the NMS output format unpacks to the right frame pixels, class index and score
* the confidence threshold is applied
* an NMS output is told apart from a raw tensor output
* empty classes do not crash it

---

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| `error: No Roboflow API key` | only the CPU path needs one, and only to *download*. Either set the key, or copy `~/.cache/roboflow-onnx/` over from a machine that has it. On `--backend hailo` this should never appear |
| `... is not cached ... has to be downloaded` | the weights are missing and there is no key. Section 2, "The model weights" |
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
