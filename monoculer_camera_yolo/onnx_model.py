"""Run a Roboflow-hosted model locally from its ONNX weights.

Roboflow's own local runtime is the `inference` pip package, but it resolves to
~250 dependencies (torch, CUDA, transformers, diffusers, easyocr, SAM-2,
groundingdino) and downgrades OpenCV, for a single small object detector. The
weights themselves are just an ONNX file, and this project already runs ONNX for
HITNET, so this fetches the artifacts once and runs them with onnxruntime.

The artifacts come from the same API endpoint the official runtime uses, and are
cached under ~/.cache/roboflow-onnx/ so the key and the network are needed only
on the first run.
"""

import json
import os
import zipfile
from pathlib import Path

import cv2 as cv
import numpy as np

CACHE_DIR = Path(os.environ.get("ROBOFLOW_ONNX_CACHE",
                                Path.home() / ".cache" / "roboflow-onnx"))
ARTIFACT_URL = "https://api.roboflow.com/ort/{model_id}"


def _find_urls(blob):
    """Walk the artifact JSON for the weights and environment URLs.

    Roboflow has moved these between top level and an "ort" sub-object across
    versions, so search rather than index into a fixed shape.
    """
    weights = env = None
    stack = [blob]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, str) and node.startswith("http"):
            head = node.split("?", 1)[0].lower()
            if head.endswith((".onnx", ".zip")) and weights is None:
                weights = node
            elif "environment" in head and env is None:
                env = node
    return weights, env


def fetch_artifacts(model_id, api_key, cache_dir=CACHE_DIR, force=False):
    """Download and cache (weights.onnx, environment.json) for a model id."""
    import requests

    dest = Path(cache_dir) / model_id.replace("/", "--")
    onnx_path, env_path = dest / "weights.onnx", dest / "environment.json"
    if onnx_path.exists() and env_path.exists() and not force:
        return onnx_path, env_path

    dest.mkdir(parents=True, exist_ok=True)
    r = requests.get(ARTIFACT_URL.format(model_id=model_id),
                     params={"api_key": api_key, "nocache": "true",
                             "device": "python", "dynamic": "true"},
                     timeout=60)
    if r.status_code in (401, 403):
        raise RuntimeError(
            f"Roboflow rejected the API key for {model_id} ({r.status_code}). "
            "Check the key, and that this workspace can access the model.")
    r.raise_for_status()
    blob = r.json()

    weights_url, env_url = _find_urls(blob)
    if not weights_url:
        raise RuntimeError(
            f"no ONNX weights URL in the artifact response for {model_id}; "
            f"keys seen: {sorted(blob)}")

    raw = requests.get(weights_url, timeout=600)
    raw.raise_for_status()
    body = raw.content
    if body[:2] == b"PK":                     # some models ship the onnx zipped
        tmp = dest / "weights.zip"
        tmp.write_bytes(body)
        with zipfile.ZipFile(tmp) as z:
            name = next(n for n in z.namelist() if n.endswith(".onnx"))
            onnx_path.write_bytes(z.read(name))
        tmp.unlink()
    else:
        onnx_path.write_bytes(body)

    if env_url:
        env_path.write_bytes(requests.get(env_url, timeout=120).content)
    else:
        env_path.write_text(json.dumps({}))
    return onnx_path, env_path


class OnnxDetectionModel:
    """YOLO-family ONNX detector: letterbox or stretch in, boxes out."""

    def __init__(self, onnx_path, env_path=None, providers=None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=providers)
        self.provider = self.session.get_providers()[0]

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # A dynamic axis reports as a string or None; fall back to 640, which is
        # what every Roboflow detection export uses.
        dims = [d if isinstance(d, int) else 0 for d in inp.shape]
        self.in_h = dims[2] or 640
        self.in_w = dims[3] or 640

        self.env = {}
        if env_path and Path(env_path).exists():
            try:
                self.env = json.loads(Path(env_path).read_text())
            except json.JSONDecodeError:
                self.env = {}
        class_map = self.env.get("CLASS_MAP") or {}
        self.classes = [class_map[k] for k in sorted(class_map, key=int)] if class_map else []
        self.letterbox = self._wants_letterbox()
        self.head = self._head_layout()

    def _head_layout(self):
        """'v5' heads carry an objectness column ahead of the class scores; 'v8'
        heads do not.

        This is read from the recorded MODEL_NAME rather than guessed from the
        tensor, because the obvious guess — that objectness bounds the class
        scores it gates — is simply untrue: this project's yolov5v6n emits
        objectness of 0.01 alongside class scores of 0.99 on the same anchor.
        """
        name = str(self.env.get("MODEL_NAME", "")).lower()
        if any(k in name for k in ("yolov5", "yolov6", "yolov7", "yolo-nas", "yolonas")):
            return "v5"
        if any(k in name for k in ("yolov8", "yolov9", "yolov10", "yolo11", "yolov11", "yolo12")):
            return "v8"
        return "auto"

    def _wants_letterbox(self):
        """Roboflow records the training-time resize mode; 'Fit' pads, 'Stretch'
        does not. Getting this wrong skews every box."""
        pre = self.env.get("PREPROCESSING")
        if isinstance(pre, str):
            try:
                pre = json.loads(pre)
            except json.JSONDecodeError:
                pre = {}
        mode = ((pre or {}).get("resize") or {}).get("format", "")
        return "fit" in str(mode).lower()

    # ---- pre / post ----------------------------------------------------

    def _preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        if self.letterbox:
            r = min(self.in_w / w, self.in_h / h)
            nw, nh = int(round(w * r)), int(round(h * r))
            canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
            canvas[:nh, :nw] = cv.resize(rgb, (nw, nh), interpolation=cv.INTER_LINEAR)
            meta = (r, r, 0.0, 0.0)
        else:
            canvas = cv.resize(rgb, (self.in_w, self.in_h), interpolation=cv.INTER_LINEAR)
            meta = (self.in_w / w, self.in_h / h, 0.0, 0.0)
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(blob), meta

    @staticmethod
    def _decode(out, head="auto"):
        """Normalise YOLOv5 and YOLOv8 head layouts to (N, 4) boxes + (N, C)
        scores, both in input-image pixels."""
        a = out[0]
        if a.ndim != 2:
            a = a.reshape(a.shape[0], -1)
        # v8 exports as (4+C, 8400); v5 as (25200, 5+C). The anchor count always
        # dwarfs the attribute count, so the long axis identifies the transpose.
        if a.shape[0] < a.shape[1]:
            a = a.T                                   # -> (anchors, attrs)
        if a.shape[1] < 5:
            raise RuntimeError(f"unexpected model output shape {out.shape}")
        boxes, rest = a[:, :4], a[:, 4:]

        if head == "auto":
            # Only shape is left to go on: a v8 head with a single class would
            # be 5 columns wide, which is also v5-with-no-classes, so assume the
            # objectness form, which is what Roboflow exports.
            head = "v5"
        if head == "v5":
            if rest.shape[1] == 1:
                return boxes, rest                    # objectness only
            return boxes, rest[:, :1] * rest[:, 1:]   # conf = objectness * class
        return boxes, rest

    def __call__(self, frame_bgr, confidence=0.4, iou=0.45):
        """Returns [(x1, y1, x2, y2, score, class_index), ...] in frame pixels."""
        blob, (sx, sy, px, py) = self._preprocess(frame_bgr)
        out = self.session.run(None, {self.input_name: blob})[0]
        boxes, scores = self._decode(np.asarray(out), self.head)

        best = scores.max(axis=1)
        keep = best >= confidence
        if not np.any(keep):
            return []
        boxes, scores, best = boxes[keep], scores[keep], best[keep]
        cls_idx = scores.argmax(axis=1)

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2 - px) / sx
        y1 = (cy - bh / 2 - py) / sy
        x2 = (cx + bw / 2 - px) / sx
        y2 = (cy + bh / 2 - py) / sy

        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv.dnn.NMSBoxes(rects, best.astype(float).tolist(), confidence, iou)
        if len(idxs) == 0:
            return []
        h, w = frame_bgr.shape[:2]
        results = []
        for i in np.asarray(idxs).ravel():
            results.append((float(np.clip(x1[i], 0, w)), float(np.clip(y1[i], 0, h)),
                            float(np.clip(x2[i], 0, w)), float(np.clip(y2[i], 0, h)),
                            float(best[i]), int(cls_idx[i])))
        return results

    def class_name(self, index):
        if 0 <= index < len(self.classes):
            return self.classes[index]
        return "drone"

    def describe(self):
        return (f"onnx {self.in_w}x{self.in_h} "
                f"{'letterbox' if self.letterbox else 'stretch'} "
                f"{self.env.get('MODEL_NAME', self.head)} "
                f"on {self.provider.replace('ExecutionProvider', '')}"
                + (f", classes={self.classes}" if self.classes else ""))
