# monocular_depth_anything — live depth from one camera

> **Just want to run it?** [GETTING_STARTED.md](GETTING_STARTED.md) is the
> step-by-step for this machine — the camera, the GPU, and the errors you will
> actually hit. This file explains what the code does and why.

Installing on a different machine: [SETUP.md](SETUP.md).

Runs [Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3) (or
[V2](https://github.com/DepthAnything/Depth-Anything-V2)) on a webcam, a clip or
a stream, and turns the output into something you can measure with. Self-
contained: its own `venv/`, no dependency on the stereo code one level up,
though it will read that project's calibration YAML if you point it at one.

```
models.py     the V3 / V2 / V2-Metric backends and the preprocessing each wants
geometry.py   intrinsics, back-projection, PLY export
metric.py     relative depth -> metres: ground-plane self-calibration, anchors
render.py     colour ramp, the five views, the HUD
run.py        the live loop
selftest.py   offline checks, no camera and no weights needed
```

## Setup

```bash
./setup.sh                 # picks the GPU or CPU runtime for this machine
./setup.sh --metric        # also torch + transformers, for the V2-Metric models
```

Weights download on first use into `models/` (v3-small is ~100 MB). No API key,
no account, nothing phones home after that.

> This machine has an NVIDIA PyIndex `extra-index-url` in `~/.config/pip/pip.conf`
> pointing at `pypi.ngc.nvidia.com`, which does not resolve. Every pip call here
> must run with `PIP_CONFIG_FILE=/dev/null`; `setup.sh` detects that specific
> setting and works around it on its own.

## Run

```bash
# webcam, relative depth, nothing to configure
./venv/bin/python run.py --source 0

# metres, self-calibrating off the floor -- measure the camera height with a tape
./venv/bin/python run.py --source 0 --scale ground --camera-height 1.20

# metres with no floor in sight, from a metric model (needs ./setup.sh --metric)
./venv/bin/python run.py --source 0 --model v2-metric-indoor

# process a clip: every frame, recorded, with real intrinsics
./venv/bin/python run.py --source clip.mp4 --calib ../calibration_data/sample_calibration.yaml \
    --save out.mp4 --no-display

./venv/bin/python run.py --list-models
```

| key | | key | |
|---|---|---|---|
| `q` | quit | `c` | colormap |
| `space` | pause | `v` | view: split / depth / overlay / confidence / camera |
| `s` | save png + npy + ply | `i` | flip the colour ramp |
| `r` | start/stop recording | `f` | freeze the colour range |
| `[` `]` | network resolution | `h` | help overlay |

Mouse: move to probe a distance, **left-click** to pin a point, right-click to
undo, **middle-click** to anchor the scale, hold shift while moving to drag the split seam.

## Which model

`--list-models` prints all eleven. The three that matter:

| model | output | notes |
|---|---|---|
| `v3-small` (default) | depth, relative | also predicts the camera intrinsics |
| `v2-small` | inverse depth, relative | sharper edges; no far field to speak of; see `--res` below |
| `v2-metric-indoor` / `-outdoor` | **metres** | absolute, no anchoring; needs torch |

Measured here on a GTX 1650 (4 GB) with a 1280x720 source:

| `--res` | v3-small | | v2-small | |
|---:|---:|---:|---:|---:|
| | network size | fps | network size | fps |
| 252 | 252x140 | 53 | 448x252 | 26 |
| 364 | 364x210 | 32 | 644x364 | 12 |
| 504 / 518 *(defaults)* | 504x280 | **19** | 924x518 | **4.2** |
| 616 | 616x350 | 11 | — | — |

V2 is four times the work at its default, and that is not a fair fight — it is
the preprocessing. V3 scales the *longest* side to `--res`, so a 16:9 frame
becomes 504x280. V2 inherits DPT's aspect-preserving resize, which scales the
axis closest to the target and lets the other overshoot: the same frame becomes
924x518, 3.4x the pixels. Both are what their networks were trained on, so
neither is a bug — but if you want V2 at a usable frame rate, pass `--res 252`
(26 fps) rather than assuming it is inherently slow.

On a live camera the display never waits for the network — capture, inference
and the viewer each run on their own thread, so the preview stays at camera
rate and the depth map refreshes at the rate above, one inference behind. For a
*file*, that would silently drop frames, so files default to `--sync` and every
frame gets its own depth map. Force either with `--sync` / `--async`.

## Getting metres out of it

This is the part worth reading twice, because the screen will happily show you
a number in either case.

**V3 and V2 do not predict distance.** They predict depth up to a scale factor
that the network re-chooses on every frame. Relative depth answers "which of
these two things is nearer", correctly and robustly, and that is all it answers.

**`--scale ground` is the mode to use on anything that moves.** Tape-measure the
camera's height above the floor once and pass it. Every frame, the bottom 45% of
the depth map is back-projected, RANSAC fits a plane through it, and the scale
is set so that plane sits exactly `--camera-height` below the camera. Because it
re-locks on every frame, the model's own drifting normalisation cancels out.
Against a synthetic floor it recovers depth to **under a millimetre** across
four decades of arbitrary model units, and it holds to 5% with obstacles
scattered over the floor and sensor noise on top (`selftest.py`).

It fails, by design, when it should: point the camera at a wall and it reports
`ground lost` and keeps showing relative depth rather than inventing a scale.
What it cannot detect is a *wrong height* — and it will just as happily lock
onto a table top, a road surface, or the sea. The plane it found is only the
floor because you say it is.

**`--scale anchor`** — middle-click a point at a known distance (set with
`--anchor-range`, default 1 m) and the scale is fixed from that one
measurement. Exact at the moment you set it, and it drifts as the scene changes,
because nothing re-locks it. Fine for a fixed camera looking at a fixed scene.

**The V2-Metric models** skip the problem: they were trained on metric data and
output metres directly. The catch is that each was trained for one kind of
scene — Hypersim indoors (0–20 m), VKITTI outdoors (0–80 m) — and applying the
wrong one is a systematic error with nothing on screen to betray it.

## Intrinsics, and V3's party trick

A point cloud, a bearing, or a ground-plane fit all need a focal length. It
comes from the first of these that is available: `--calib` (a YAML from
`../calibrate.py`), `--fx`, `--hfov`, then **V3's own per-frame prediction**.
V3 regresses the camera matrix along with the depth — on the test clip here it
returns fx = 992 px, a 63° horizontal field of view — which is what lets
`--scale ground` work on a camera nobody has ever calibrated. The HUD always
prints which one is in use. It is an estimate from image content, so it moves
with the scene; calibrate if the number matters.

## Output

`s` (or `--snapshot` headless) writes five files into `captures/`: the view as
displayed, the raw frame, `_depth.npy` (float32, metres when a scale is known),
`_depth_mm.png` (16-bit millimetres, the usual RGB-D convention, metric only),
and `_cloud.ply` — a coloured point cloud that opens in MeshLab or CloudCompare.

`--save out.mp4` records the view, as H.264 when ffmpeg is on PATH and `mp4v`
otherwise — the latter only plays in VLC, so the run warns when it falls back.
`--codec` overrides the choice. `--save-csv` logs every pinned point per
frame: `frame, t_s, pin, u, v, value, units, scale, fx_px`.

## Where this is weak

* **No parallax.** One camera cannot measure distance; it can only recognise
  what things at that distance look like. Everything above is a way of
  borrowing a scale from somewhere else. The stereo pipeline one directory up
  gives true metric range with no prior at all, out to roughly
  `baseline * fx / 1px`. Use stereo close in and this beyond it.
* **Thin structures and glass.** Railings, wires and chain-link go to the
  background; windows and mirrors return the reflected scene's depth. V2 at 518
  px resolves finer detail here than v3-small at 504.
* **Temporal flicker.** Each frame is predicted independently, so a surface can
  breathe by a few percent between frames even when nothing moves. `--scale
  ground` suppresses most of it (the scale re-locks to something physical);
  the colour range is smoothed for the same reason. If you need genuinely
  stable video depth, the upstream project's Video-Depth-Anything models are
  built for it and are not wired up here.
* **The far field on V2.** Inverse depth near zero is an unbounded distance, so
  V2's depth is capped at 30x the scene median and its sky is meaningless.
  V3 predicts depth directly and does not have this problem.
