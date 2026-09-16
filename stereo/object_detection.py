"""YOLOX (ONNX) object detector, wrapping OpenCV Model Zoo's reference pre/post-processing.

Model: object_detection_yolox_2022nov.onnx (COCO, 80 classes), from
https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox

The article this project follows (Teledyne's embedded stereo guide) pairs
stereo depth with a MobileNetV2-SSD detector run via TensorRT on a Jetson.
That model format (Caffe) is no longer loadable by OpenCV's DNN module
(support was dropped in OpenCV 5.0), so this uses an equivalent ONNX
detector that OpenCV's own DNN module is built and tested against.
"""

import numpy as np
import cv2 as cv

COCO_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
)


def letterbox(img_rgb, target_size=(640, 640)):
    """Resize preserving aspect ratio, pad the remainder with grey (114)."""
    padded = np.full((target_size[0], target_size[1], 3), 114.0, dtype=np.float32)
    ratio = min(target_size[0] / img_rgb.shape[0], target_size[1] / img_rgb.shape[1])
    new_w, new_h = int(img_rgb.shape[1] * ratio), int(img_rgb.shape[0] * ratio)
    resized = cv.resize(img_rgb, (new_w, new_h), interpolation=cv.INTER_LINEAR).astype(np.float32)
    padded[:new_h, :new_w] = resized
    return padded, ratio


class YoloXDetector:
    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.5, input_size=(640, 640)):
        self.net = cv.dnn.readNet(model_path)
        self.input_size = input_size
        self.strides = (8, 16, 32)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._generate_grids()

    def _generate_grids(self):
        grids, expanded_strides = [], []
        for stride in self.strides:
            h = self.input_size[0] // stride
            w = self.input_size[1] // stride
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
        self.grids = np.concatenate(grids, 1)
        self.expanded_strides = np.concatenate(expanded_strides, 1)

    def detect(self, frame_bgr):
        """Returns a list of dicts: {box: (x, y, w, h) in frame_bgr pixel
        coords, score, class_id, class_name}."""
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        padded, ratio = letterbox(rgb, self.input_size)

        blob = np.transpose(padded, (2, 0, 1))[np.newaxis, :, :, :]
        self.net.setInput(blob)
        out = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]

        out[:, :2] = (out[:, :2] + self.grids[0]) * self.expanded_strides[0]
        out[:, 2:4] = np.exp(out[:, 2:4]) * self.expanded_strides[0]

        boxes = out[:, :4]
        boxes_xywh = np.empty_like(boxes)
        boxes_xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xywh[:, 2:4] = boxes[:, 2:4]

        scores = out[:, 4:5] * out[:, 5:]
        max_scores = scores.max(axis=1)
        max_ids = scores.argmax(axis=1)

        keep = cv.dnn.NMSBoxesBatched(
            boxes_xywh.tolist(), max_scores.tolist(), max_ids.tolist(),
            self.conf_threshold, self.nms_threshold,
        )

        detections = []
        for i in keep:
            i = int(i)
            x, y, w, h = boxes_xywh[i] / ratio      # undo letterbox scale
            detections.append({
                "box": (x, y, w, h),
                "score": float(max_scores[i]),
                "class_id": int(max_ids[i]),
                "class_name": COCO_CLASSES[int(max_ids[i])],
            })
        return detections
