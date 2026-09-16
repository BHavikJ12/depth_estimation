"""HITNET dense stereo matching via ONNX Runtime.

Drop-in alternative to stereo/matching.py's SGBM+WLS pipeline. Where SGBM
scores hand-made cost volumes, HITNET (Tayar et al., https://arxiv.org/abs/2007.12140)
learns the matching and the propagation, so it fills textureless regions
that SGBM leaves as holes.

Why ONNX and not the original TensorFlow release: Google's frozen `.pb`
files (storage.googleapis.com/tensorflow-graphics/models/hitnet/...) now
return HTTP 403 for anonymous callers, so the TensorFlow repo cannot be
fed a model any more. PINTO0309 converted the same trained weights to
ONNX before access was pulled (PINTO_model_zoo/142_HITNET); those are
what `download_hitnet_models.sh` fetches.

Geometry notes that the upstream reference implementations get wrong, and
that matter as soon as you want millimetres rather than a pretty picture:

  * Every converted model has a FIXED input size, so a 1280x720 eye has to
    be scaled. Disparity is measured in pixels, so scaling the image by s
    scales the disparity by s too. The disparity this module returns is
    rescaled back into FULL-RESOLUTION pixels, which is the unit cv.Q and
    Z = f*B/d both expect.
  * A plain cv.resize to the model's 4:3 input would squash 16:9 frames
    horizontally and vertically by different factors, which no single
    disparity rescale can undo. So we scale uniformly and pad (letterbox)
    instead, then crop the padding off the disparity.
"""

import os
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import onnxruntime as ort


# Max disparity each family was trained for, in pixels at the model's own
# input width. Used to reject the garbage values HITNET emits where it has
# no real match (it is a dense regressor -- it never reports "unknown").
TRAINED_MAX_DISPARITY = {
    "eth3d": 128,
    "middlebury_d400": 400,
    "flyingthings_finalpass_xl": 320,
}


@dataclass
class ModelInfo:
    family: str          # eth3d | middlebury_d400 | flyingthings_finalpass_xl
    input_height: int
    input_width: int
    channels: int        # 2 (grayscale pair) or 6 (RGB pair)
    max_disparity: int   # at the model's own input width


def _preload_cuda():
    """Make the CUDA/cuDNN wheels visible to the CUDA execution provider.

    `onnxruntime-gpu[cuda,cudnn]` installs CUDA 12 and cuDNN 9 as pip
    packages under site-packages/nvidia/*/lib, which is not on the loader
    path. Without this the provider fails to load libcublasLt.so.12 and
    silently falls back to CPU -- a ~20x slowdown that only shows up as a
    warning on stderr.
    """
    try:
        ort.preload_dlls()
    except Exception:                                             # noqa: BLE001
        # Older runtimes have no preload_dlls; they expect a system CUDA
        # install, which may well already be on LD_LIBRARY_PATH.
        pass


def _family_from_path(model_path: str) -> str:
    """The PINTO archive keeps the family in the directory name."""
    parts = os.path.normpath(model_path).split(os.sep)
    for family in TRAINED_MAX_DISPARITY:
        if family in parts:
            return family
    return "unknown"


class HitNet:
    """Rectified stereo pair in, full-resolution disparity (pixels) out."""

    def __init__(self, model_path: str, providers=None, intra_threads: int = 0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"HITNET model not found: {model_path}\n"
                "Run `./download_hitnet_models.sh` to fetch the ONNX models."
            )

        if providers is None:
            # CUDA if the build and the driver can supply it, CPU otherwise.
            _preload_cuda()
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                # kSameAsRequested stops the default arena from doubling its
                # reservation on every growth, which is what makes the larger
                # models fail with CUBLAS_STATUS_ALLOC_FAILED on a 4GB card.
                providers.append(("CUDAExecutionProvider", {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                }))
            providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        if intra_threads:
            opts.intra_op_num_threads = intra_threads

        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self.providers = self.session.get_providers()

        # Everything about the model is read off the graph rather than the
        # filename, so a renamed file still behaves correctly.
        inp = self.session.get_inputs()[0]
        _, channels, height, width = inp.shape
        if channels not in (2, 6):
            raise ValueError(
                f"Unexpected HITNET input channel count {channels} (expected 2 or 6)."
            )

        self.input_name = inp.name
        # flyingthings predicts left AND right disparity; the rest predict left only.
        self.output_names = [o.name for o in self.session.get_outputs()]

        family = _family_from_path(model_path)
        self.info = ModelInfo(
            family=family,
            input_height=int(height),
            input_width=int(width),
            channels=int(channels),
            max_disparity=TRAINED_MAX_DISPARITY.get(family, 320),
        )

        # Letterbox parameters get cached on the first frame, since the eye
        # size never changes mid-stream.
        self._scale = None
        self._content = None   # (h, w) of the real image inside the padded tensor

    # ------------------------------------------------------------------ #

    def _letterbox(self, img):
        """Uniform downscale + bottom/right pad to the model's input size."""
        h, w = img.shape[:2]
        scale = min(self.info.input_width / w, self.info.input_height / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        # INTER_AREA is the right filter for downscaling; it averages instead
        # of point-sampling, which keeps the fine texture HITNET matches on.
        interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
        resized = cv.resize(img, (new_w, new_h), interpolation=interp)

        canvas = np.zeros((self.info.input_height, self.info.input_width) + img.shape[2:],
                          dtype=img.dtype)
        canvas[:new_h, :new_w] = resized

        self._scale = scale
        self._content = (new_h, new_w)
        return canvas

    def _prepare_input(self, left_bgr, right_bgr):
        left = self._letterbox(left_bgr)
        right = self._letterbox(right_bgr)

        if self.info.channels == 2:
            left = cv.cvtColor(left, cv.COLOR_BGR2GRAY)[:, :, None]
            right = cv.cvtColor(right, cv.COLOR_BGR2GRAY)[:, :, None]
        else:
            left = cv.cvtColor(left, cv.COLOR_BGR2RGB)
            right = cv.cvtColor(right, cv.COLOR_BGR2RGB)

        combined = np.concatenate((left, right), axis=-1).astype(np.float32) / 255.0
        return combined.transpose(2, 0, 1)[None, ...]   # NCHW

    # ------------------------------------------------------------------ #

    def __call__(self, left_bgr, right_bgr):
        return self.compute(left_bgr, right_bgr)

    def compute(self, left_bgr, right_bgr):
        """Returns (disparity_px, valid_mask) at the input images' own resolution.

        disparity_px is float32 in full-resolution pixels, i.e. exactly what
        cv.reprojectImageTo3D(disparity, Q) and Z = f*B/d expect.
        valid_mask is False where HITNET's output falls outside the disparity
        range the model was trained for -- it regresses a value everywhere,
        including at occlusions and off-range surfaces, so this is the only
        rejection signal a single-output model gives us.
        """
        src_h, src_w = left_bgr.shape[:2]
        tensor = self._prepare_input(left_bgr, right_bgr)

        # Output 0 is the left/reference disparity for every family.
        raw = self.session.run([self.output_names[0]], {self.input_name: tensor})[0]
        disp = np.squeeze(raw).astype(np.float32)

        # Drop the padding before rescaling, otherwise the black bars bleed
        # into the interpolation along the bottom edge.
        content_h, content_w = self._content
        disp = disp[:content_h, :content_w]

        valid = (disp > 0) & (disp < self.info.max_disparity) & np.isfinite(disp)

        # Back to full resolution, and back into full-resolution pixel units.
        disp_full = cv.resize(disp, (src_w, src_h), interpolation=cv.INTER_LINEAR)
        disp_full /= self._scale

        valid_full = cv.resize(valid.astype(np.uint8), (src_w, src_h),
                               interpolation=cv.INTER_NEAREST).astype(bool)
        disp_full[~valid_full] = 0.0

        return disp_full, valid_full

    # ------------------------------------------------------------------ #

    def max_disparity_at(self, src_size) -> float:
        """Largest disparity this model can represent for a (width, height)
        input, in FULL-RESOLUTION pixels.

        The trained limit is in model-input pixels, so downscaling the frame
        shrinks how much real-world disparity fits inside it -- which is
        exactly the near-range limit: Z_min = f*B / max_disparity_at(...).
        """
        w, h = src_size
        scale = min(self.info.input_width / w, self.info.input_height / h)
        return self.info.max_disparity / scale

    def describe(self) -> str:
        i = self.info
        return (f"HITNET {i.family} {i.input_width}x{i.input_height} "
                f"({i.channels}ch, max disp {i.max_disparity}px) "
                f"on {self.providers[0]}")


def depth_mm_from_disparity(disparity_px, valid_mask, focal_px, baseline_mm):
    """Z = f*B/d, with invalid pixels left at 0 instead of +inf."""
    depth = np.zeros_like(disparity_px, dtype=np.float32)
    np.divide(focal_px * baseline_mm, disparity_px, out=depth, where=valid_mask)
    return depth


def colorize_disparity(disparity_px, valid_mask, max_disparity=None):
    """MAGMA over a FIXED disparity range so colours mean the same thing
    frame to frame -- per-frame min/max normalisation (what the upstream
    repos do) makes a static scene shimmer as objects enter and leave."""
    if not np.any(valid_mask):
        return np.zeros(disparity_px.shape + (3,), dtype=np.uint8)

    if max_disparity is None:
        max_disparity = float(np.percentile(disparity_px[valid_mask], 99))
    max_disparity = max(max_disparity, 1.0)

    norm = np.clip(disparity_px / max_disparity, 0.0, 1.0) * 255.0
    color = cv.applyColorMap(norm.astype(np.uint8), cv.COLORMAP_MAGMA)
    color[~valid_mask] = 0
    return color
