# HITNET on the AR0144 stereo camera

HITNET ([Tankovich et al., 2020](https://arxiv.org/abs/2007.12140)) replaces
the hand-tuned cost volume in `stereo/matching.py` (StereoSGBM + WLS) with a
learned one. In practice that means it fills blank walls, tabletops and other
textureless regions where SGBM returns holes, and it does it in one forward
pass instead of an 8-direction aggregation.

It does **not** replace calibration. HITNET assumes a rectified pair, so
`calibrate.py` and `stereo/rectify.py` are still the things that decide
whether the millimetres mean anything.

---

## Why ONNX and not the TensorFlow repo

`HITNET-Stereo-Depth-estimation/` (cloned here) is the TensorFlow front-end.
It cannot be run: it loads Google's frozen `.pb` graphs, and as of
2026-09-14 all three return **HTTP 403 AccessDenied** —

```
https://storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/eth3d.pb
https://storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/middlebury_d400.pb
https://storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/flyingthings_finalpass_xl.pb
```

Anonymous read on that bucket has been revoked. Installing TensorFlow would
not help — there is no model to load. See
`HITNET-Stereo-Depth-estimation/CANNOT_RUN_READ_ME.md`.

PINTO0309 converted the same trained weights to ONNX before that happened
([PINTO_model_zoo/142_HITNET](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/142_HITNET)),
and that archive is still public. That is what this setup uses.

---

## Setup

```bash
pip install -r requirements.txt     # brings onnxruntime-gpu + CUDA 12/cuDNN 9 wheels (~3 GB)
./download_hitnet_models.sh         # ~440 MB archive -> models/hitnet/ (~59 MB kept)
python check_hitnet_setup.py        # verifies runtime, models, calibration, camera; benchmarks
```

`onnxruntime-gpu[cuda,cudnn]` ships CUDA 12 and cuDNN 9 as pip wheels, so
the system CUDA 11.8 toolkit is not used and does not need to match. Those
wheels land in `site-packages/nvidia/*/lib`, which is not on the loader path
— `stereo/hitnet.py` calls `onnxruntime.preload_dlls()` to fix that. Without
it the CUDA provider fails to find `libcublasLt.so.12` and silently falls
back to CPU, which is about 20x slower.

## Running it

```bash
python live_depth_hitnet.py                  # default model, click a pixel for distance
python live_depth_hitnet.py --compare        # HITNET and SGBM side by side
python live_depth_hitnet.py --model models/hitnet/middlebury_d400/480x640/model_float32.onnx
python live_depth_hitnet.py --cpu            # no GPU
```

Controls match `live_depth.py`: click to read the distance at a pixel
(reported with the `dZ = Z²/(B·f)·dd` uncertainty from `stereo/depth_error.py`),
`q` to quit.

---

## Which model

Measured on this machine (GTX 1650 4 GB, 1280x720 per eye). Accuracy is
against the Middlebury *cones* ground truth (`bad2.0` = share of pixels off
by more than 2 px):

| model | input | FPS | MAE | bad2.0 |
| --- | --- | ---: | ---: | ---: |
| **eth3d** | **240x320** | **25.8** | **0.98 px** | **11.3 %** |
| eth3d | 480x640 | 7.9 | 0.69 px | 7.1 % |
| eth3d | 720x1280 | 2.7 | 0.68 px | 6.5 % |
| middlebury_d400 | 240x320 | 9.6 | 0.96 px | 10.0 % |
| middlebury_d400 | 480x640 | 3.0 | **0.43 px** | **4.9 %** |
| middlebury_d400 | 720x1280 | — | out of GPU memory | |
| flyingthings_finalpass_xl | 240x320 | 5.8 | 2.44 px | 28.3 % |
| flyingthings_finalpass_xl | 480x640 | 1.5 | 1.12 px | 11.8 % |
| flyingthings_finalpass_xl | 720x1280 | — | out of GPU memory | |

* **`eth3d/240x320` is the default** — the only combination that keeps up
  with the camera (26 FPS against the sensor's 30), and end-to-end with
  rectification it measures ~41 ms/frame.
* **`middlebury_d400/480x640`** if you care about accuracy more than rate;
  it is about 2x better but runs at 3 FPS.
* **flyingthings is not worth using here.** It was trained on synthetic
  data and is the worst of the three on real images at every size.
* The two 720x1280 models that fail need a single 1.5 GB allocation, which
  does not fit in 4 GB. That is a hardware limit, not a configuration one.

### Is it actually better than the SGBM pipeline?

Same cones pair, same ground-truth pixels, `--compare` shows this live:

| method | coverage | MAE | bad2.0 |
| --- | ---: | ---: | ---: |
| SGBM + WLS (`conf > 100`) | 61.7 % | 0.38 px | 2.6 % |
| HITNET eth3d/240x320 | 100 % | 0.98 px | 11.3 % |
| HITNET middlebury_d400/480x640 | 100 % | 0.43 px | 4.9 % |

SGBM is not worse — *where it answers*, it is as good as the best HITNET
model. It just declines to answer 38 % of the image. HITNET's value here is
coverage, not per-pixel accuracy: it returns a usable disparity on the
textureless surfaces where SGBM's confidence check rejects everything.

Which you want depends on the task. For picking a distance to a detected
object (`object_depth.py`), SGBM's confident 62 % is fine and cheaper. For a
dense depth map with no holes, HITNET wins outright.

### Depth range, and why downscaling widens it

Each model has a maximum disparity baked in, measured in *its own input*
pixels. Because the frame is scaled down to reach that input, the range in
full-resolution pixels is *larger* than the trained number:

```
max_disparity_full = trained_max / scale      scale = min(in_w/w, in_h/h)
Z_min              = f · B / max_disparity_full
```

For `eth3d/240x320` on a 1280x720 eye: scale 0.25, so 128 px becomes 512 px,
and with f ≈ 960 px and B = 52 mm that is a near limit of **≈ 98 mm**. The
far end is set by sub-pixel resolution rather than range — expect useful
depth out to roughly 10 m, degrading as `Z²` per `stereo/depth_error.py`.

`live_depth_hitnet.py` prints these bounds at startup and warns if
`--min-depth-mm` asks for more disparity than the model can represent.

---

## Implementation notes

`stereo/hitnet.py` deviates from the upstream reference implementations in
two places, both of which matter once you want millimetres rather than a
colourful picture:

1. **Disparity is rescaled back to full-resolution pixels.** Disparity is a
   pixel measurement, so scaling the image by *s* scales the disparity by
   *s*. The reference code resizes the disparity map for display but never
   undoes that factor, which leaves depth wrong by 2–4x. `compute()` returns
   disparity in the units `cv.reprojectImageTo3D(disp, Q)` and `Z = f·B/d`
   both expect.

2. **Letterbox instead of plain resize.** A direct `cv.resize` of a 16:9
   frame into a 4:3 model input squashes the axes by different factors, and
   no single disparity rescale can undo that. `compute()` scales uniformly,
   pads, and crops the padding off the disparity before rescaling.

`compute()` returns `(disparity_px, valid_mask)`. The mask matters: HITNET
is a dense regressor and emits a number for every pixel including occlusions
and out-of-range surfaces — it never reports "no match". The mask is the
only rejection signal a single-output model gives, so it is built from the
trained disparity range, and `live_depth_hitnet.py` narrows it further using
`--min-depth-mm` / `--max-depth-mm`.

## Still to do

`check_hitnet_setup.py` currently reports only the nominal sample
calibration. Run `python calibrate.py` before trusting any distance —
`sample_calibration.yaml` is seeded from datasheet specs, not from this
particular camera, so both the focal length and the baseline are nominal.
