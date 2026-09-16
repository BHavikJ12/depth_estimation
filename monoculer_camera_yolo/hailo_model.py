"""Run the detector on a Hailo-8 / Hailo-8L accelerator via HailoRT.

Hailo does not run ONNX. A model must be compiled ahead of time into a `.hef`
on an x86_64 Linux machine with the Hailo Dataflow Compiler, then copied to the
Pi; see SETUP_RPI_HAILO.md. This module only loads and runs the result.

Two HEF flavours are handled, because which one you get depends on how it was
compiled:

  NMS baked in   The usual Hailo Model Zoo YOLO configuration. The device
                 returns decoded boxes, one variable-length list per class, as
                 normalised (ymin, xmin, ymax, xmax, score). Nothing to decode.
  raw tensors    The head comes back undecoded, and is fed through the same
                 YOLO decoder the ONNX path uses.

Note the input convention differs from ONNX: a HEF takes uint8 NHWC, and the
normalisation that `onnx_model.py` does by hand is compiled into the graph.
Dividing by 255 here would quietly halve every activation.
"""

import numpy as np

try:                                   # only present on a Hailo-equipped machine
    from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                                InferVStreams, InputVStreamParams, OutputVStreamParams,
                                VDevice)
    HAILO_AVAILABLE = True
except ImportError:                    # pragma: no cover - depends on the host
    HAILO_AVAILABLE = False


class HailoDetectionModel:
    """Same call signature as OnnxDetectionModel: model(frame) -> detections."""

    def __init__(self, hef_path, class_names=None, letterbox=False):
        if not HAILO_AVAILABLE:
            raise RuntimeError(
                "hailo_platform is not importable. On a Pi with an AI HAT install "
                "it with `sudo apt install hailo-all`, and build the venv with "
                "--system-site-packages so the system bindings are visible.")
        import cv2 as cv
        self._cv = cv

        self.hef_path = str(hef_path)
        self.classes = list(class_names) if class_names else []
        self.letterbox = letterbox

        self.hef = HEF(self.hef_path)
        self.target = VDevice(VDevice.create_params())
        cfg = ConfigureParams.create_from_hef(self.hef,
                                              interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, cfg)[0]
        self.ng_params = self.network_group.create_params()

        in_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = in_info.name
        self.in_h, self.in_w = in_info.shape[0], in_info.shape[1]   # NHWC

        self.out_infos = self.hef.get_output_vstream_infos()
        self.in_params = InputVStreamParams.make(self.network_group,
                                                 format_type=FormatType.UINT8)
        self.out_params = OutputVStreamParams.make(self.network_group,
                                                   format_type=FormatType.FLOAT32)
        self._pipeline = None
        self._activated = None

    # ---- lifecycle ----------------------------------------------------

    def _ensure_open(self):
        """Configure once and keep it open; per-frame setup costs more than the
        inference does."""
        if self._pipeline is None:
            self._pipeline = InferVStreams(self.network_group, self.in_params,
                                           self.out_params).__enter__()
            self._activated = self.network_group.activate(self.ng_params).__enter__()

    def close(self):
        if self._activated is not None:
            self._activated.__exit__(None, None, None)
            self._activated = None
        if self._pipeline is not None:
            self._pipeline.__exit__(None, None, None)
            self._pipeline = None

    # ---- pre / post ----------------------------------------------------

    def _preprocess(self, frame_bgr):
        cv = self._cv
        h, w = frame_bgr.shape[:2]
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        if self.letterbox:
            r = min(self.in_w / w, self.in_h / h)
            nw, nh = int(round(w * r)), int(round(h * r))
            canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
            canvas[:nh, :nw] = cv.resize(rgb, (nw, nh), interpolation=cv.INTER_LINEAR)
            meta = (r, r)
        else:
            canvas = cv.resize(rgb, (self.in_w, self.in_h), interpolation=cv.INTER_LINEAR)
            meta = (self.in_w / w, self.in_h / h)
        # uint8 NHWC with a batch axis; no scaling - it is compiled in.
        return np.expand_dims(canvas, 0), meta

    @staticmethod
    def is_nms_output(value):
        """HailoRT's NMS output is a per-class list of (n, 5) arrays, not a
        single dense tensor."""
        if isinstance(value, list):
            return True
        return isinstance(value, np.ndarray) and value.dtype == object

    def _decode_nms(self, per_class, frame_shape, confidence):
        """Boxes arrive normalised as (ymin, xmin, ymax, xmax, score), already
        in the *original* aspect only if the model was fed the whole frame —
        which it was, so they scale straight back to frame pixels."""
        h, w = frame_shape[:2]
        out = []
        for cls_id, dets in enumerate(per_class):
            if dets is None or len(dets) == 0:
                continue
            arr = np.asarray(dets, dtype=np.float32).reshape(-1, 5)
            for ymin, xmin, ymax, xmax, score in arr:
                if score < confidence:
                    continue
                out.append((float(xmin) * w, float(ymin) * h,
                            float(xmax) * w, float(ymax) * h,
                            float(score), int(cls_id)))
        return out

    def __call__(self, frame_bgr, confidence=0.4, iou=0.45):
        """Returns [(x1, y1, x2, y2, score, class_index), ...] in frame pixels."""
        self._ensure_open()
        blob, (sx, sy) = self._preprocess(frame_bgr)
        results = self._pipeline.infer({self.input_name: blob})
        value = results[self.out_infos[0].name]

        first = value[0] if isinstance(value, (list, np.ndarray)) and len(value) else value
        if self.is_nms_output(first):
            return self._decode_nms(first, frame_bgr.shape, confidence)

        # Raw head: reuse the ONNX decoder, then undo the resize.
        from onnx_model import OnnxDetectionModel
        import cv2 as cv
        boxes, scores = OnnxDetectionModel._decode(np.asarray(value), "v5")
        best = scores.max(axis=1)
        keep = best >= confidence
        if not np.any(keep):
            return []
        boxes, scores, best = boxes[keep], scores[keep], best[keep]
        cls_idx = scores.argmax(axis=1)
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = (cx - bw / 2) / sx, (cy - bh / 2) / sy
        x2, y2 = (cx + bw / 2) / sx, (cy + bh / 2) / sy
        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv.dnn.NMSBoxes(rects, best.astype(float).tolist(), confidence, iou)
        if len(idxs) == 0:
            return []
        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                 float(best[i]), int(cls_idx[i])) for i in np.asarray(idxs).ravel()]

    def class_name(self, index):
        if 0 <= index < len(self.classes):
            return self.classes[index]
        return "drone"

    def describe(self):
        return (f"hailo {self.in_w}x{self.in_h} "
                f"{'letterbox' if self.letterbox else 'stretch'} "
                f"{self.hef_path.split('/')[-1]}"
                + (f", classes={self.classes}" if self.classes else ""))
