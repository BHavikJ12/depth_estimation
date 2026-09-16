# monocular_camera — monocular drone detection + range

Detects drones with the Roboflow Universe model
[`drone-detection-ssbrv/drone-detection-rchy7`](https://universe.roboflow.com/drone-detection-ssbrv/drone-detection-rchy7),
tracks them across frames, and turns each bounding box into a metric range
using an assumed physical target size.

Runs on a **live camera, a video file, a stream, a single image, or a folder of
images** — same command, only `--source` changes.

Self-contained: its own `venv/`, no imports from the stereo code one directory
up, though it will read that project's calibration YAMLs.

---

## 1. Setup

```bash
git clone https://github.com/BHavikJ12/depth_estimation.git
cd depth_estimation/monoculer_camera_yolo
./setup.sh                              # venv + 4 packages + both test suites
export ROBOFLOW_API_KEY=xxxxxxxxxxxx    # free: https://app.roboflow.com/settings/api
```

Put the export in `~/.bashrc` so it survives new shells. The first run downloads
the model's ONNX weights (7 MB) to `~/.cache/roboflow-onnx/`; after that it is
fully local and offline, and the key is no longer needed — it is required only
to download something not already cached. Copying that cache directory to
another machine is enough to run there with no key at all.

> **pip on this machine:** `~/.config/pip/pip.conf` (NVIDIA PyIndex) points at
> `pypi.ngc.nvidia.com`, which does not resolve, so every pip call fails with a
> DNS error naming that host. Prefix with `PIP_CONFIG_FILE=/dev/null`.
> `setup.sh` already does.

---

## 2. Run it

All four examples use the same flags; only `--source` differs. A window opens
unless you pass `--no-display`.

### Live camera

```bash
./venv/bin/python run.py --source 0 \
    --model-id drone-detection-rchy7/8 --keep-classes 1 \
    --conf 0.40 --hfov 70 --target-size 0.5
```

`--source 0` is the first camera, `1` the second, and so on (`ls /dev/video*`).
Runs about 27 fps on CPU at 720p, so it keeps up with a normal webcam.

### Video file

```bash
./venv/bin/python run.py --source v1.mp4 \
    --model-id drone-detection-rchy7/8 --keep-classes 1 \
    --conf 0.40 --hfov 70 --target-size 0.5 \
    --save v1_out.mp4 --save-csv v1_track.csv
```

### Network stream

```bash
./venv/bin/python run.py --source rtsp://192.168.1.50:554/stream1 ...
```

Any URL OpenCV can open works — `rtsp://`, `http://`, an MJPEG endpoint.

### One image

```bash
./venv/bin/python run.py --source shot.jpg ... --save shot_out.jpg
```

The window waits for a keypress so you can actually look at it.

### A folder of images

```bash
./venv/bin/python run.py --source frames/ ... --save frames_out/
# or a glob:
./venv/bin/python run.py --source 'frames/*.jpg' ...
```

Files are processed in sorted order, and `--save` is treated as an output
**folder**, one annotated image per input, keeping the original filenames. Quote
the glob so the shell does not expand it first. Add `--step` to advance one
keypress at a time.

### Window controls

| key | |
|---|---|
| `q` or `Esc` | quit |
| any key | next frame, when `--step` is on or the source is a single image |

On Wayland the window goes through XWayland. If nothing appears, prefix the
command with `QT_QPA_PLATFORM=xcb`.

---

## 3. Options that matter

| flag | what it does |
|---|---|
| `--source` | camera index, video path, stream URL, image path, folder, or glob |
| `--model-id` | `drone-detection-rchy7/8` is the full-dataset model (mAP 75.9). `run.py --list-versions` prints them all |
| `--keep-classes 1` | this model has 3 classes and only id 1 is the aircraft; without this you get birds and dataset noise |
| `--conf` | detection threshold. 0.40 is a reasonable start; raise it if the sky is full of false boxes |
| `--target-size` | the assumed real span in metres. **Every range scales with this** |
| `--fx` / `--hfov` / `--calib` | focal length. **Every range scales with this too** — see below |
| `--stride N` | run the detector every Nth frame, tracker coasts between. Cheap speedup |
| `--save` | annotated video, image, or folder, matching the input kind |
| `--save-csv` | per-frame range telemetry |
| `--no-display` | headless |
| `--step` | wait for a keypress between frames |
| `--max-frames N` | stop early, useful for a quick look |

Backends, if you ever need something other than the default:

| `--backend` | how | cost |
|---|---|---|
| `onnx` (default) | the model's ONNX weights, run in-process | ~400 MB venv, offline after first run |
| `hosted` | HTTPS to Roboflow per frame | 150–400 ms/frame |
| `server` | Roboflow's container here, via `--endpoint` | a docker pull |
| `ultralytics` | your own `.pt`/`.onnx` via `--weights` | no Roboflow dependency |

Roboflow's own `inference` package would run the model in-process too, but it
resolves to ~250 packages — torch, CUDA 13, transformers, diffusers, easyocr,
SAM-2, groundingdino — and downgrades OpenCV 5.0 to 4.12, all for one small
detector. `--backend onnx` runs the same weights without any of it. Install
`onnxruntime-gpu` if the CPU stops keeping up.

---

## 4. Getting a range you can trust

```
Z = f_px * S_real / span_px
```

`span_px` is `max(box_width, box_height)`, because a target's apparent width
shrinks with viewing angle and never grows, so the larger box dimension is the
least foreshortened estimate of the real span.

Two inputs, and **the output is wrong by whatever factor either one is wrong by**.

### Focal length

`--hfov 70` is a placeholder. A guess that is 10% off makes every range 10% off,
systematically, with nothing on screen to show it. In order of preference:

```bash
--calib ../calibration_data/sample_calibration.yaml   # a real calibration
--fx 1404                                             # a measured focal length
--hfov 70                                             # the lens datasheet
```

Intrinsics are rescaled automatically if the frame is not the resolution they
were calibrated at.

### Target size

The default 0.5 m is a typical quadcopter motor-to-motor span. A fixed-wing RC
plane is nearer 1.2 m, and getting this wrong scales every range by the ratio.
Rather than argue about the true span, measure it once:

```bash
./venv/bin/python run.py --source shot_at_12m.jpg --solve-size 12 \
    --model-id drone-detection-rchy7/8 --keep-classes 1 --hfov 70
# --target-size 0.4732   (473 mm effective span)
```

Park the target at a tape-measured distance, point the camera at it, and this
prints the `--target-size` that makes the arithmetic agree. That one number
absorbs the real airframe span, the detector's box-tightness bias, and the
typical viewing angle together, which is worth more than any datasheet figure.

### How far this works

Error is proportional to range, because it scales as `1/span_px`. With fx = 960
and a 0.5 m target:

| true range | span | 1σ |
|---:|---:|---:|
| 5 m | 96 px | ±1.0 m |
| 10 m | 48 px | ±2.1 m |
| 25 m | 19 px | ±6.2 m |
| 60 m | 8 px | ±24 m |

The HUD prints its own `±` and a "usable to ~Nm" figure, which is where box
jitter alone costs 25%. Past that the number is a bearing with a decimal point
attached.

---

## 5. Output

`--save-csv` writes one row per track per frame:

```
frame, t_s, track_id, locked, conf, range_m, sigma_m, x_m, y_m, z_m, az_deg, el_deg, span_px
```

`x/y/z` are in the camera frame (+X right, +Y down, +Z forward). `az/el` are
degrees off boresight — that pair, not the range, is what a gimbal or a guidance
loop actually consumes. `locked` marks the chase target.

On screen: every track gets a box, the locked one gets the range readout, a
`CHASE TARGET` callout and a crosshair, and the top-left block carries fps,
focal length, target size, track count and the lock's range with its trend
(`CLOSING` / `STEADY` / `OPENING`).

---

## 6. Tests

Both run without an API key, a camera, or a network.

```bash
./venv/bin/python selftest.py       # geometry, tracker, ONNX decode, HUD
./venv/bin/python pipeline_test.py  # the real CLI over a synthetic clip
```

`pipeline_test.py` renders a 0.5 m target closing from 40 m to 7.5 m at a known
focal length and checks the recovered range against ground truth: 1.4% mean
error inside 20 m, with every frame's error inside the `±` the tool reports.
That last part is the real check — a range that is wrong and says so is usable;
one that is wrong and claims precision is not.

---

## 7. Layout

```
camera.py         intrinsics: calibration YAML | measured fx | lens FOV -> focal px
sources.py        one iterator over camera / video / stream / image / folder
detector.py       the Roboflow model, four backends, class filtering
onnx_model.py     fetches the ONNX weights once, runs them with onnxruntime
ranging.py        Z = f*S/span, its uncertainty, and the 3D bearing
tracker.py        Kalman + association, and the sticky chase-target lock
overlay.py        the HUD
run.py            CLI
selftest.py       offline checks
pipeline_test.py  end-to-end check against known ranges
```

`testdata/` and `outputs/` are not in the repo — they are hundreds of megabytes
of clips. Both test suites generate their own synthetic footage, so a fresh
clone verifies itself with no video files present.

---

## 8. What this model actually does, measured

Recall depends heavily on the footage. Measured on three clips, model version 8
at `--conf 0.40 --keep-classes 1`:

| clip | what it is | result |
|---|---|---|
| `v1.mp4` | fixed-wing RC plane over a field, 848x480 | **works.** 453 detections over 1076 frames, tracked across the clip |
| `v2.mp4` | FPV racing footage *from* a drone | mostly false positives on clouds and the sun; there is no target aircraft in frame |
| `testdata/air_to_air.mp4` | quadcopter against sky, with another system's HUD burned in | **misses the target.** 32 of 296 frames, and those fire on HUD artifacts rather than the drone |

The air-to-air case is the instructive failure: that target is a ~40 px dark
quadcopter against bright sky, which is exactly the regime a general-purpose
detector is weakest in. Before trusting this on your own footage, run it and
look at the boxes — a confident-looking range on the wrong object is the
failure mode to watch for, and the HUD's `Conf:` figure will not warn you,
because the detector is genuinely confident about the wrong thing.

If you need recall at range, that is a detector problem, not a ranging problem:
fine-tune on air-to-air data (Det-Fly, Anti-UAV, Drone-vs-Bird), add tiled
inference (SAHI), or put a motion-compensated candidate stage ahead of the
detector. `--model-id drone-detection-rchy7/1` is also worth trying; it reports
a higher mAP on a different split.

## 9. Known limits
* **Box convention.** The size prior is calibrated against *whichever detector's
  boxes you are using*. Two detectors that box the same target differently
  produce ranges differing by that ratio, so `--solve-size` has to be redone if
  you change `--model-id`.
* **The size prior.** Anything that is not `--target-size` across reads at the
  wrong range by exactly the ratio of the spans, and nothing in the pipeline can
  notice.
* **False positives on clouds and sun.** This model fires on bright sky texture.
  `--keep-classes 1` and a higher `--conf` help; they do not cure it.
* **No parallax.** One camera means all of the above is unavoidable. The stereo
  pipeline one directory up gives true metric range with no size prior, out to
  roughly `baseline * fx / 1px` — about 10–15 m on a 52 mm baseline. The two are
  complementary: stereo close in, size prior beyond it.
