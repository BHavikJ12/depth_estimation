"""Depth Anything backends: one image in, one depth map out.

Two families are supported, and the difference between them is the single most
important thing to understand before reading a number off the screen:

  V3  (`depth-anything/DA3-*`, ONNX)   predicts **depth**, larger = farther,
      in an arbitrary per-frame scale. It also predicts the camera intrinsics,
      which is what makes a metric point cloud possible without a calibration
      target. Use `metric.py` to pin its scale to metres.

  V2  (`Depth-Anything-V2-*`, ONNX)    predicts **inverse depth** (disparity),
      larger = nearer, also in an arbitrary scale.

  V2-Metric (`...-Metric-{Indoor,Outdoor}-*`, torch/transformers) predicts
      depth in **real metres** with no anchoring at all, at the cost of being
      trained for one scene type — Hypersim indoors, VKITTI outdoors — and of
      pulling in torch.

Every backend returns a `DepthResult` at the *network's* resolution; the caller
resizes to the display frame, because the scale factor is also needed to carry
the predicted intrinsics across to full resolution.
"""

import os
import time
from dataclasses import dataclass

import cv2 as cv
import numpy as np

# ImageNet statistics. Both V2 and V3 inherit them from DINOv2's pretraining,
# so this is shared across every backend here.
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# DINOv2 tokenises in 14x14 patches, so every input dimension must be a
# multiple of 14 or the patch embedding silently drops a strip of the image.
PATCH = 14


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str          # "onnx" | "torch"
    repo: str
    path: str = ""        # file within the repo, for onnx
    kind: str = "depth"   # "depth" (V3) | "inverse" (V2 rel) | "metric" (V2 metric)
    preproc: str = "da3"  # "da3" (longest side) | "dpt" (aspect-preserving 518)
    res: int = 504        # default processing resolution
    note: str = ""


MODELS = {m.name: m for m in [
    # --- Depth Anything V3: depth + confidence + predicted intrinsics --------
    ModelSpec("v3-small", "onnx", "onnx-community/depth-anything-v3-small",
              "onnx/model.onnx", "depth", "da3", 504, "default; ~100 MB"),
    ModelSpec("v3-base", "onnx", "onnx-community/depth-anything-v3-base",
              "onnx/model.onnx", "depth", "da3", 504, "~380 MB"),
    ModelSpec("v3-large", "onnx", "onnx-community/depth-anything-v3-large",
              "onnx/model.onnx", "depth", "da3", 504, "~1.3 GB; tight on 4 GB"),

    # --- Depth Anything V2, relative (inverse depth) -------------------------
    ModelSpec("v2-small", "onnx", "onnx-community/depth-anything-v2-small",
              "onnx/model.onnx", "inverse", "dpt", 518),
    # fp16 is a win on Ampere and later. On this Turing card it measured 3x
    # SLOWER than fp32 (845 ms vs 273 ms), so benchmark before reaching for it.
    ModelSpec("v2-small-fp16", "onnx", "onnx-community/depth-anything-v2-small",
              "onnx/model_fp16.onnx", "inverse", "dpt", 518,
              "GPU only; slower than fp32 on Turing"),
    ModelSpec("v2-base", "onnx", "onnx-community/depth-anything-v2-base",
              "onnx/model.onnx", "inverse", "dpt", 518),
    ModelSpec("v2-large", "onnx", "onnx-community/depth-anything-v2-large",
              "onnx/model.onnx", "inverse", "dpt", 518),

    # --- Depth Anything V2, metric: real metres, needs torch -----------------
    ModelSpec("v2-metric-indoor", "torch",
              "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
              kind="metric", preproc="dpt", res=518, note="Hypersim, 0-20 m"),
    ModelSpec("v2-metric-indoor-base", "torch",
              "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
              kind="metric", preproc="dpt", res=518),
    ModelSpec("v2-metric-outdoor", "torch",
              "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
              kind="metric", preproc="dpt", res=518, note="VKITTI, 0-80 m"),
    ModelSpec("v2-metric-outdoor-base", "torch",
              "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
              kind="metric", preproc="dpt", res=518),
]}


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class DepthResult:
    """One forward pass, at network resolution."""
    raw: np.ndarray                 # HxW float32, in the model's own units
    kind: str                       # "depth" | "inverse" | "metric"
    confidence: np.ndarray = None   # HxW float32, V3 only
    K: np.ndarray = None            # 3x3 predicted intrinsics, V3 only
    latency_ms: float = 0.0

    @property
    def is_metric(self):
        return self.kind == "metric"

    def depth(self):
        """The model's output as a depth (Z, larger = farther).

        V2 predicts inverse depth, so this reciprocates -- and that is where a
        V2 map stops being trustworthy. Sky and specular holes come back at
        essentially zero disparity, whose reciprocal is not "large" but
        undefined. The result is capped at 30x the median depth: far enough
        that nothing real is clipped, close enough that one patch of sky cannot
        set the extent of a point cloud or the inlier threshold of a plane fit.
        Use a V3 model if the far field matters.
        """
        if self.kind != "inverse":
            return self.raw.astype(np.float32)
        d = self.raw
        pos = d[d > 0]
        if pos.size == 0:
            return np.zeros_like(d, np.float32)
        eps = max(float(pos.min()) * 1e-3, 1e-9)
        z = 1.0 / np.maximum(d, eps)
        cap = float(np.median(z)) * 30.0
        return np.minimum(z, cap).astype(np.float32)

    def display(self):
        """The quantity to colourise, and whether high values mean *near*.

        Not the same question as `depth()`. A V2 map has to be painted in its
        native disparity: the reciprocal puts almost every pixel of a scene
        with sky in it into the bottom few percent of the range, and the
        picture goes flat. Disparity is what the network actually regresses and
        it is what every published Depth Anything visualisation shows.
        """
        if self.kind == "inverse":
            return self.raw.astype(np.float32), True
        return self.raw.astype(np.float32), False

# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def _round_to_patch(x):
    return max(PATCH, int(round(x / PATCH)) * PATCH)


def net_size_da3(w, h, res):
    """V3: scale the longest side to `res`, round both to a multiple of 14.

    Matches `upper_bound_resize` in the upstream input processor.
    """
    s = res / float(max(w, h))
    return _round_to_patch(w * s), _round_to_patch(h * s)


def net_size_dpt(w, h, res):
    """V2: DPTImageProcessor with keep_aspect_ratio=True.

    It computes a scale for each axis against the square `res` target and keeps
    the one closer to 1.0, so a 4:3 frame comes out *taller* than `res` rather
    than being letterboxed. Reproduced here rather than approximated, because
    feeding V2 a differently-shaped input than it was trained on costs real
    accuracy at the frame edges.
    """
    sh, sw = res / float(h), res / float(w)
    s = sw if abs(1 - sw) < abs(1 - sh) else sh
    return _round_to_patch(w * s), _round_to_patch(h * s)


def preprocess(bgr, res, preproc):
    """BGR uint8 frame -> normalised CHW float32 tensor, plus the size used."""
    h, w = bgr.shape[:2]
    nw, nh = (net_size_da3 if preproc == "da3" else net_size_dpt)(w, h, res)
    interp = cv.INTER_CUBIC if nw * nh > w * h else cv.INTER_AREA
    img = cv.resize(cv.cvtColor(bgr, cv.COLOR_BGR2RGB), (nw, nh), interpolation=interp)
    x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)), (nw, nh)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class OnnxDepthModel:
    """ONNX Runtime backend, CUDA when it loads and CPU when it does not."""

    def __init__(self, spec, res=None, device="auto", cache_dir="models", verbose=True):
        import onnxruntime as ort

        # onnxruntime-gpu ships the CUDA/cuDNN runtime as pip wheels whose .so
        # files are not on the loader path. Without this the CUDA provider
        # fails to load with a bare "libcublasLt.so.12 not found" and silently
        # drops to CPU -- 6x slower, no error. Guarded on the provider actually
        # existing, because on a CPU-only build preload_dlls warns about
        # missing CUDA, which is alarming and entirely expected.
        if hasattr(ort, "preload_dlls") and \
                "CUDAExecutionProvider" in ort.get_available_providers():
            try:
                ort.preload_dlls()
            except Exception:
                pass
        # The V3 graph trips a per-inference ScatterND advisory that would
        # otherwise print six lines for every frame of live video.
        ort.set_default_logger_severity(3)

        self.spec = spec
        self.res = res or spec.res
        self.path = ensure_weights(spec, cache_dir, verbose=verbose)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The V3 graph emits a pile of harmless ScatterND/Memcpy warnings.
        opts.log_severity_level = 3

        avail = ort.get_available_providers()
        want = []
        if device in ("auto", "cuda") and "CUDAExecutionProvider" in avail:
            want.append(("CUDAExecutionProvider", {"device_id": 0}))
        want.append("CPUExecutionProvider")

        self.sess = ort.InferenceSession(str(self.path), opts, providers=want)
        self.providers = self.sess.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError(
                "--device cuda requested but the CUDA provider would not load; "
                "run selftest.py for the reason")

        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_rank = len(inp.shape)     # V3 is (B,N,3,H,W), V2 is (B,3,H,W)
        self.fp16 = "float16" in inp.type
        self.outputs = [o.name for o in self.sess.get_outputs()]

    @property
    def device(self):
        return "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"

    def infer(self, bgr):
        x, _ = preprocess(bgr, self.res, self.spec.preproc)
        x = x[None] if self.input_rank == 4 else x[None, None]
        if self.fp16:
            x = x.astype(np.float16)
        t0 = time.perf_counter()
        out = self.sess.run(None, {self.input_name: np.ascontiguousarray(x)})
        dt = (time.perf_counter() - t0) * 1000.0
        named = dict(zip(self.outputs, out))

        def squeeze(a):
            a = np.asarray(a, np.float32)
            while a.ndim > 2:
                a = a[0]
            return a

        depth = squeeze(named.get("predicted_depth", out[0]))
        conf = squeeze(named["confidence"]) if "confidence" in named else None
        K = None
        if "intrinsics" in named:
            k = np.asarray(named["intrinsics"], np.float32).reshape(-1, 3, 3)[0]
            # V3 occasionally emits a degenerate K on a featureless frame; a
            # zero or negative focal length would poison every downstream
            # projection, so reject it here rather than guard at each use.
            if np.isfinite(k).all() and k[0, 0] > 1.0 and k[1, 1] > 1.0:
                K = k
        return DepthResult(depth, self.spec.kind, conf, K, dt)


class TorchDepthModel:
    """transformers backend, the only route to true metric depth."""

    def __init__(self, spec, res=None, device="auto", cache_dir="models", verbose=True):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise SystemExit(
                f"{spec.name} is a torch model and {e.name!r} is not installed.\n"
                f"  ./setup.sh --metric      (torch + transformers, ~2.5 GB)\n"
                f"Or stay on ONNX and get metres from the floor instead:\n"
                f"  run.py --model v3-small --scale ground --camera-height 1.2")

        self.spec = spec
        self.res = res or spec.res
        self.torch = torch
        self.dev = torch.device(
            "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
        if verbose:
            print(f"[models] loading {spec.repo} (transformers) on {self.dev}")
        self.proc = AutoImageProcessor.from_pretrained(spec.repo, cache_dir=cache_dir)
        self.net = AutoModelForDepthEstimation.from_pretrained(
            spec.repo, cache_dir=cache_dir).to(self.dev).eval()
        # fp16 halves both latency and VRAM on the metric models, and depth is
        # not a task where the last three mantissa bits matter.
        self.half = self.dev.type == "cuda"
        if self.half:
            self.net = self.net.half()
        self.providers = [f"torch:{self.dev.type}"]

    @property
    def device(self):
        return self.dev.type

    def infer(self, bgr):
        x, _ = preprocess(bgr, self.res, self.spec.preproc)
        t = self.torch.from_numpy(x[None]).to(self.dev)
        if self.half:
            t = t.half()
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.net(pixel_values=t).predicted_depth
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        d = out.float().squeeze().cpu().numpy()
        return DepthResult(d, self.spec.kind, None, None, dt)


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def ensure_weights(spec, cache_dir="models", verbose=True):
    """Fetch the ONNX file (and its external-data sibling) once, return a path."""
    from huggingface_hub import hf_hub_download

    local = os.path.join(cache_dir, spec.name)
    target = os.path.join(local, spec.path)
    if os.path.exists(target):
        return target
    if verbose:
        print(f"[models] fetching {spec.repo}:{spec.path} -> {local}")
    path = hf_hub_download(spec.repo, spec.path, local_dir=local)
    # Anything over the 2 GB protobuf limit is exported with its weights in a
    # side-car; ONNX Runtime opens it by relative name, so it has to land in
    # the same directory. Models under the limit simply have no such file.
    try:
        hf_hub_download(spec.repo, spec.path + "_data", local_dir=local)
    except Exception:
        pass
    return path


def load_model(name, res=None, device="auto", cache_dir="models", verbose=True):
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; choose from: {', '.join(MODELS)}")
    spec = MODELS[name]
    cls = OnnxDepthModel if spec.backend == "onnx" else TorchDepthModel
    return cls(spec, res=res, device=device, cache_dir=cache_dir, verbose=verbose)
