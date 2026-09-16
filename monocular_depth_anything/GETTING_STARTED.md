# Getting started — on this machine

A run-it-now guide for **bhavik@Ubuntu 22.04**, GTX 1650 (4 GB), the CCB USB
camera on `/dev/video0`. For what the code does and why, see [README.md](README.md).

Everything below has been run on this machine and works as written.

---

## 1. Run it

The venv is already built and the default weights are already downloaded, so:

```bash
cd depth_estimation/monocular_depth_anything
./venv/bin/python run.py --source 0
```

A window opens with the camera on the left and depth on the right. Press `h`
for the key list, `q` to quit.

Expect **~21 fps** at 1280x720. Warm, red-to-orange is near; blue-to-purple is far.

> Starting from a clean checkout instead? `./setup.sh` builds the venv and runs
> the self-test. Weights download on first use (~100 MB for the default model).

**Always call the venv's python explicitly** (`./venv/bin/python`), or activate
it first with `source venv/bin/activate`. Plain `python3` is the system 3.10 and
has none of the dependencies.

---

## 2. Check everything is healthy

```bash
./venv/bin/python selftest.py
```

14 checks, no camera and no weights needed, about a second. The one to watch is
the last:

```
[  ok  ] onnxruntime providers  CUDA provider loads (v3-small)
```

If that says **CPU only** or **listed but refused to load**, you are about to
run at 3 fps instead of 21 — see [Troubleshooting](#7-troubleshooting).

Add `--model --bench` to also load the network and time it at every resolution.

---

## 3. Driving the window

| key | does | key | does |
|---|---|---|---|
| `q` / `Esc` | quit | `c` | next colormap |
| `space` | pause | `v` | next view |
| `s` | save png + npy + ply | `i` | flip the colour ramp |
| `r` | start / stop recording | `f` | freeze the colour range |
| `[` `]` | network resolution down / up | `h` | key list overlay |
| `x` | clear pinned points | | |

Mouse:

- **move** — reads out the distance and bearing under the cursor
- **left-click** — pin that point so its number stays on screen
- **right-click** — remove the last pin
- **middle-click** — anchor the metric scale here (see below)
- **hold shift while moving** — drag the split seam between camera and depth

`v` cycles five views: `split` (default, a wipe — best for checking that a depth
edge really sits on the object's edge), `depth`, `overlay`, `confidence`
(V3 only), and `camera`.

---

## 4. Getting real distances

Out of the box the numbers are **relative**, not metres. The HUD says
`scale: relative` so you always know which you are looking at. Relative depth
correctly answers "which of these is nearer" and nothing more.

### The good way: let it measure the floor (recommended)

Tape-measure from the floor to the camera lens once, then:

```bash
./venv/bin/python run.py --source 0 --scale ground --camera-height 1.20
```

The HUD changes to `scale: ground 1.2m (N pts)` and every readout is in metres.
It re-fits the floor plane on **every frame**, so the scale stays correct as you
move the camera around. Point it at a wall with no floor in view and it says
`ground lost` and falls back to relative rather than inventing a number.

Three things to get right:

- Keep some **floor visible in the lower part of the frame** — that is where it
  looks for the plane.
- The height must be **to the lens**, not to the tripod head or the desk.
- It cannot tell a floor from a table top, a road, or a worktop. Whichever flat
  surface dominates the bottom of the frame becomes "the floor" and is assumed
  to be `--camera-height` below the camera.

### The quick way: click a point you have measured

```bash
./venv/bin/python run.py --source 0 --scale anchor --anchor-range 2.0
```

Middle-click something exactly 2.0 m away and the whole map is scaled from it.
Exact at the moment you click, and it drifts as the scene changes, because
nothing re-locks it. Fine for a camera that is not moving.

### The absolute way: a metric model

```bash
./setup.sh --metric                  # adds torch + transformers, ~2.5 GB
./venv/bin/python run.py --source 0 --model v2-metric-indoor
```

Outputs metres directly with no floor and no tape measure. Pick `-indoor`
(trained 0–20 m) or `-outdoor` (0–80 m) to match where you are — using the
wrong one is a systematic error with nothing on screen to give it away.

---

## 5. Your camera, specifically

`/dev/video0` is the **CCB Camera** — the stereo unit this project's parent
directory is built around. Two of its modes matter here:

| mode | what you get |
|---|---|
| **1280x720** (the default) | one eye. This is what you want. |
| 2560x720 | **both eyes side by side in one frame.** |

Do not pass `--width 2560`. A monocular depth model handed a side-by-side pair
treats it as one strange wide photograph and the output is meaningless. The
default is already correct; nothing to do unless you go looking for trouble.

`/dev/video1` is the same device's metadata node and will not open — that is
normal, not a fault.

Other useful sizes: `--width 640 --height 480` lightens the USB and display load
without touching inference cost (the network resizes internally either way).

If you have calibrated this camera with the parent project, use the real
intrinsics instead of V3's estimate:

```bash
./venv/bin/python run.py --source 0 --calib ../calibration_data/sample_calibration.yaml
```

The bottom-left of the HUD always prints which intrinsics are in use.

---

## 6. Speed on this GPU

Measured here, 1280x720 in, GTX 1650:

| `--res` | v3-small | v2-small |
|---:|---:|---:|
| 252 | 50 fps | 26 fps |
| 364 | 32 fps | 12 fps |
| **504 / 518** (defaults) | **19 fps** | 4 fps |
| 616 | 11 fps | — |

`[` and `]` change this live, so you can trade sharpness for frame rate while
watching the result. If you want V2 specifically, pass `--res 252` — at its
default it is four times the work of V3 for the same frame.

Only `v3-small`, `v2-small` and the small metric models fit comfortably in 4 GB.
`v3-large` (~1.3 GB of weights) will be tight; `v3-base` is the sensible step up.

---

## 7. Troubleshooting

**pip hangs for a minute then fails.** `~/.config/pip/pip.conf` points at
`pypi.ngc.nvidia.com`, which does not resolve here. Prefix every pip call:

```bash
PIP_CONFIG_FILE=/dev/null ./venv/bin/pip install ...
```

`setup.sh` already does this. (That file also sets `no-cache-dir`, so every
install re-downloads — expect `--metric` to pull the full 2.5 GB.)

**Running at ~3 fps, HUD says `cpu`.** The CUDA provider did not load. Run
`./venv/bin/python selftest.py` for the reason. It is almost always the
`onnxruntime-gpu` CUDA/cuDNN wheels missing:

```bash
PIP_CONFIG_FILE=/dev/null ./venv/bin/pip install --force-reinstall "onnxruntime-gpu[cuda,cudnn]>=1.19"
```

Force the issue with `--device cuda`, which fails loudly instead of quietly
falling back.

**`cannot open source: 0`.** Something else holds the camera — close any other
viewer, or check with `fuser -v /dev/video0`. Access is granted to you by a
per-login ACL, not by group membership, so it only works while you are logged in
at the desktop. Over SSH, add yourself to the `video` group once and log back in:

```bash
sudo usermod -aG video bhavik
```

**`v2-metric-indoor is a torch model and 'torch' is not installed`.** Either run
`./setup.sh --metric`, or stay on ONNX and get metres from the floor with
`--scale ground --camera-height ...`.

**No window appears, or it crashes on `imshow`.** You are probably on SSH or a
bare Wayland session. Run headless instead:

```bash
./venv/bin/python run.py --source 0 --no-display --snapshot     # one capture
./venv/bin/python run.py --source 0 --no-display --save out.mp4 # record
```

**The saved video plays as a black screen.** Your OpenCV has no H.264 encoder,
so it falls back to `mp4v`, which only VLC reliably plays. Install ffmpeg once
and `--save` switches to H.264 automatically:

```bash
sudo apt install ffmpeg
```

Check what you have with `ffprobe -v error -show_entries stream=codec_name -of default=nw=1 out.mp4`
— `h264` plays everywhere, `mpeg4` is the one that goes black. Force it either
way with `--codec h264` or `--codec mp4v`. The picture itself is fine in both;
it is purely a container/codec problem.

**Depth looks flat / everything one colour.** Usually a featureless close-up
(a blank wall fills the frame). Back off, or press `f` to freeze the colour
range on a good frame first.

---

## 8. Saving things

Press `s` (or add `--snapshot` when headless). Five files land in `captures/`:

| file | what it is |
|---|---|
| `*_view.png` | exactly what was on screen, HUD included |
| `*_rgb.png` | the clean camera frame |
| `*_depth.npy` | float32 depth — **metres** when a scale is known, else relative |
| `*_depth_mm.png` | 16-bit millimetres, the standard RGB-D format (metric runs only) |
| `*_cloud.ply` | coloured 3D point cloud |

Open the cloud with MeshLab or CloudCompare:

```bash
sudo apt install meshlab && meshlab captures/*_cloud.ply
```

Read the depth back in Python:

```python
import numpy as np
d = np.load("captures/20260915_124036_depth.npy")
print(d.shape, d[360, 640])     # distance at the centre pixel
```

Other outputs: `--save out.mp4` records the view, `--save-csv track.csv` logs
every pinned point per frame as `frame, t_s, pin, u, v, value, units, scale, fx_px`.

---

## 9. Processing a file instead of the camera

```bash
./venv/bin/python run.py --source clip.mp4                       # watch it
./venv/bin/python run.py --source clip.mp4 --save depth.mp4 --no-display
./venv/bin/python run.py --source photo.jpg --snapshot           # one still
```

Files get **every** frame processed (the run prints `mode: synchronous`), so a
clip takes as long as the frame rate table above implies. A live camera instead
never makes the display wait — it shows the newest depth map available and drops
what it missed. Override either with `--sync` / `--async`.

---

## 10. Where things live

```
depth_estimation/monocular_depth_anything/
├── venv/          the Python environment — use ./venv/bin/python
├── models/        downloaded weights (~340 MB so far); safe to delete, re-downloads
├── captures/      whatever you save with `s`
└── *.py           the code; README.md explains the design
```

Full option list: `./venv/bin/python run.py --help`.
Model list with sizes: `./venv/bin/python run.py --list-models`.
