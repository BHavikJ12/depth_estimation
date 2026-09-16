"""Drone detector backed by the Roboflow Universe model
`drone-detection-ssbrv/drone-detection-rchy7`.

Backends, all returning the same `Detection` list in the coordinate space of
the frame you passed in:

  onnx         The model's ONNX weights, fetched once and then run in-process
               with onnxruntime. Local, offline after the first run, and about
               400 MB of dependencies. This is the default.
  hosted       An HTTPS call per frame to Roboflow's serverless endpoint. Needs
               nothing but `requests`, but costs 150-400 ms of round trip per
               frame, so it is for stills and for checking the wiring.
  server       The same HTTP protocol against a Roboflow inference server
               running on this machine (`--endpoint http://localhost:9001`),
               if you would rather run their container.
  hailo        A compiled .hef on a Hailo-8/8L accelerator (Raspberry Pi AI
               HAT). Needs --weights pointing at the .hef; see
               SETUP_RPI_HAILO.md for how to produce one.
  ultralytics  A local .pt/.onnx you trained or exported yourself.

Roboflow's own `inference` package would also run the model in-process, but it
pulls ~250 dependencies including torch, CUDA, transformers, easyocr and SAM-2,
and downgrades OpenCV, all for one small detector. `--backend onnx` runs the
same weights without any of it. If `inference` happens to be installed anyway,
`--backend local` will use it.

Every backend but `ultralytics` needs a Roboflow API key. Get a free one at
https://app.roboflow.com/settings/api and export ROBOFLOW_API_KEY.
"""

import base64
import os
from dataclasses import dataclass

import cv2 as cv
import numpy as np

WORKSPACE = "drone-detection-ssbrv"
PROJECT = "drone-detection-rchy7"
DEFAULT_MODEL_ID = f"{PROJECT}/1"

_HOSTED_ENDPOINTS = ("https://serverless.roboflow.com", "https://detect.roboflow.com")
DEFAULT_LOCAL_ENDPOINT = "http://localhost:9001"


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "drone"
    class_id: int = 0

    @property
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @classmethod
    def from_roboflow(cls, pred, scale=1.0):
        """Roboflow reports centre-x/centre-y/width/height; we want corners."""
        w, h = float(pred["width"]), float(pred["height"])
        cx, cy = float(pred["x"]), float(pred["y"])
        return cls(
            x1=(cx - w / 2.0) * scale, y1=(cy - h / 2.0) * scale,
            x2=(cx + w / 2.0) * scale, y2=(cy + h / 2.0) * scale,
            confidence=float(pred.get("confidence", 0.0)),
            label=str(pred.get("class", "drone")),
            class_id=int(pred.get("class_id", 0)),
        )


def resolve_api_key(explicit=None):
    key = explicit or os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("ROBOFLOW_KEY")
    if not key:
        raise RuntimeError(
            "No Roboflow API key. Get a free one at https://app.roboflow.com/settings/api "
            "then `export ROBOFLOW_API_KEY=...`, or pass --api-key."
        )
    return key


def list_versions(api_key=None):
    """Ask Roboflow which trained versions of the project exist, so the model
    id does not have to be guessed."""
    import requests

    key = resolve_api_key(api_key)
    r = requests.get(f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}",
                     params={"api_key": key}, timeout=30)
    r.raise_for_status()
    project = r.json().get("project", {})
    versions = r.json().get("versions", [])
    return {
        "name": project.get("name", PROJECT),
        "classes": project.get("classes", {}),
        "versions": [
            {
                "id": v.get("id", ""),
                "version": str(v.get("id", "")).rsplit("/", 1)[-1],
                "name": v.get("name", ""),
                "map": (v.get("model") or {}).get("map"),
                "images": v.get("images"),
            }
            for v in versions
        ],
    }


class DroneDetector:
    """Uniform front end over the three backends."""

    def __init__(self, backend="auto", model_id=DEFAULT_MODEL_ID, api_key=None,
                 confidence=0.40, overlap=0.30, weights=None, max_side=None,
                 endpoint=None, keep_classes=None, class_names=None):
        self.model_id = model_id
        self.confidence = float(confidence)
        self.overlap = float(overlap)
        self.weights = weights
        self.max_side = max_side          # downscale before inference, boxes scale back
        self.endpoint = endpoint
        # This project's model has more than one class and the extras are not
        # useful here: an unfiltered run boxes birds and dataset noise as
        # confidently as it boxes the aircraft.
        self.keep_classes = set(keep_classes) if keep_classes else None
        self.class_names = list(class_names) if class_names else None
        self._impl = None

        if backend == "auto":
            backend = self._autodetect(weights, endpoint)
        self.backend = backend

        if backend == "hailo":
            self.api_key = None
            self._init_hailo()
        elif backend == "ultralytics":
            self.api_key = None
            self._init_ultralytics()
        elif backend == "onnx":
            self.api_key = resolve_api_key(api_key)
            self._init_onnx()
        elif backend == "local":
            self.api_key = resolve_api_key(api_key)
            self._init_local()
        elif backend in ("hosted", "server"):
            # A local server wants the key the first time it pulls a model, but
            # serves a cached one without it, so do not insist on one here.
            if backend == "hosted":
                self.api_key = resolve_api_key(api_key)
            else:
                self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
                self.endpoint = self.endpoint or DEFAULT_LOCAL_ENDPOINT
            self._init_http()
        else:
            raise ValueError(f"unknown backend {backend!r}")

    @staticmethod
    def _autodetect(weights, endpoint):
        if weights and str(weights).endswith(".hef"):
            return "hailo"
        if weights:
            return "ultralytics"
        if endpoint:
            return "server"
        try:
            import onnxruntime  # noqa: F401
            return "onnx"
        except ImportError:
            return "hosted"

    # ---- backend setup ------------------------------------------------

    def _init_hailo(self):
        from hailo_model import HailoDetectionModel
        if not self.weights:
            raise ValueError("backend 'hailo' needs --weights pointing at a .hef")
        self._impl = HailoDetectionModel(self.weights, self.class_names)

    def _init_onnx(self):
        from onnx_model import OnnxDetectionModel, fetch_artifacts
        onnx_path, env_path = fetch_artifacts(self.model_id, self.api_key)
        self._impl = OnnxDetectionModel(onnx_path, env_path)

    def _init_local(self):
        from inference import get_model
        self._impl = get_model(model_id=self.model_id, api_key=self.api_key)

    def _init_http(self):
        import requests
        self._session = requests.Session()
        # An explicit endpoint is used as given; otherwise the first cloud host
        # that answers is remembered for the rest of the run.
        self._resolved = self.endpoint

    def _init_ultralytics(self):
        from ultralytics import YOLO
        if not self.weights:
            raise ValueError("backend 'ultralytics' needs --weights")
        self._impl = YOLO(self.weights)

    # ---- inference ----------------------------------------------------

    def _prepare(self, frame):
        """Optionally shrink the frame; returns (frame_to_send, scale_back)."""
        if not self.max_side:
            return frame, 1.0
        longest = max(frame.shape[:2])
        if longest <= self.max_side:
            return frame, 1.0
        r = self.max_side / longest
        small = cv.resize(frame, (int(frame.shape[1] * r), int(frame.shape[0] * r)),
                          interpolation=cv.INTER_AREA)
        return small, 1.0 / r

    def detect(self, frame):
        """frame is BGR uint8. Returns detections in that frame's pixel space."""
        sent, scale = self._prepare(frame)
        if self.backend in ("onnx", "hailo"):
            out = self._detect_onnx(sent, scale)
        elif self.backend == "local":
            out = self._detect_local(sent, scale)
        elif self.backend in ("hosted", "server"):
            out = self._detect_http(sent, scale)
        else:
            out = self._detect_ultralytics(sent, scale)

        if self.keep_classes is not None:
            out = [d for d in out if d.class_id in self.keep_classes]
        if self.class_names:
            for d in out:
                if 0 <= d.class_id < len(self.class_names):
                    d.label = self.class_names[d.class_id]
        return out

    def _detect_onnx(self, frame, scale):
        out = self._impl(frame, self.confidence, self.overlap)
        return [Detection(x1 * scale, y1 * scale, x2 * scale, y2 * scale,
                          conf, self._impl.class_name(cid), cid)
                for x1, y1, x2, y2, conf, cid in out]

    def _detect_local(self, frame, scale):
        result = self._impl.infer(frame, confidence=self.confidence, iou_threshold=self.overlap)[0]
        out = []
        for p in result.predictions:
            d = Detection.from_roboflow(
                {"x": p.x, "y": p.y, "width": p.width, "height": p.height,
                 "confidence": p.confidence, "class": p.class_name,
                 "class_id": getattr(p, "class_id", 0) or 0},
                scale)
            if d.confidence >= self.confidence:
                out.append(d)
        return out

    def _detect_http(self, frame, scale):
        ok, buf = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        payload = base64.b64encode(buf).decode("ascii")
        params = {"api_key": self.api_key,
                  "confidence": int(self.confidence * 100),
                  "overlap": int(self.overlap * 100)}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        endpoints = [self._resolved] if self._resolved else list(_HOSTED_ENDPOINTS)
        last = None
        for base in endpoints:
            try:
                r = self._session.post(f"{base.rstrip('/')}/{self.model_id}",
                                       params=params, data=payload,
                                       headers=headers, timeout=30)
                r.raise_for_status()
                self._resolved = base
                return [Detection.from_roboflow(p, scale)
                        for p in r.json().get("predictions", [])]
            except Exception as exc:     # try the next host, report the last failure
                last = exc
        where = self._resolved or " / ".join(_HOSTED_ENDPOINTS)
        raise RuntimeError(f"inference request to {where} failed: {last}") from last

    def _detect_ultralytics(self, frame, scale):
        res = self._impl.predict(frame, conf=self.confidence, iou=self.overlap, verbose=False)[0]
        out = []
        names = getattr(res, "names", {}) or {}
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) * scale for v in b.xyxy[0].tolist())
            cid = int(b.cls[0])
            out.append(Detection(x1, y1, x2, y2, float(b.conf[0]),
                                 str(names.get(cid, "drone")), cid))
        return out

    def describe(self):
        bits = [f"backend={self.backend}", f"model={self.model_id}",
                f"conf>={self.confidence:.2f}"]
        if self.weights:
            bits.append(f"weights={self.weights}")
        if self.max_side:
            bits.append(f"max_side={self.max_side}")
        if self.endpoint:
            bits.append(f"endpoint={self.endpoint}")
        if self.backend in ("onnx", "hailo") and self._impl is not None:
            bits.append(self._impl.describe())
        if self.keep_classes is not None:
            bits.append(f"keep_classes={sorted(self.keep_classes)}")
        return "  ".join(bits)
