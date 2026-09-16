"""Combined stereo depth + object detection, per Teledyne's embedded stereo guide:
https://www.teledynevisionsolutions.com/en-in/learn/learning-center/machine-vision/how-to-build-a-custom-embedded-stereo-system-for-depth-perception

Their reference system runs a MobileNetV2-SSD detector via TensorRT on a
Jetson alongside SGM stereo, and reports each detected object's distance
instead of (or alongside) the raw dense depth map. This does the same
thing with what's available here: SGM (StereoSGBM, full 8-direction mode)
+ WLS-filtered dense depth (stereo/matching.py) for the range data, and an
ONNX YOLOX detector (stereo/object_detection.py) for the boxes.

For each detected object:
  * distance = the MEDIAN of the confident depth pixels inside a shrunk
    (inner 60%) crop of its box -- shrinking avoids the background pixels
    that box edges usually include, and median is robust to the speckle
    that still survives WLS filtering.
  * uncertainty = the Teledyne accuracy formula dZ = Z^2/(B*f)*dd
    (stereo/depth_error.py), so you get a +/- figure instead of a bare
    number that implies false precision.

Usage:
    python object_depth.py [--calib ...] [--model models/object_detection_yolox_2022nov.onnx]
"""

import argparse

import cv2 as cv
import numpy as np

from stereo import config
from stereo.camera import StereoCamera
from stereo.rectify import load_and_report, build_rectify_maps
from stereo.matching import make_matchers, make_wls_filter, compute_filtered_disparity
from stereo.object_detection import YoloXDetector
from stereo.depth_error import depth_uncertainty_mm

DEFAULT_MODEL = "models/object_detection_yolox_2022nov.onnx"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib", default=None, help="calibration YAML (defaults: real, else sample)")
    p.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--model", default=DEFAULT_MODEL, help="path to the YOLOX ONNX model")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--fast", action="store_true",
                   help="use 3-way SGM (faster, slightly less accurate) instead of full 8-direction SGM")
    return p.parse_args()


def box_depth_mm(disparity_raw, points_3d, confidence, box, min_confidence=100.0):
    """Median depth (mm) over the confident pixels in the inner 60% of `box`."""
    x, y, w, h = box
    h_img, w_img = disparity_raw.shape[:2]
    shrink = 0.2
    x0 = int(np.clip(x + shrink * w, 0, w_img - 1))
    x1 = int(np.clip(x + (1 - shrink) * w, x0 + 1, w_img))
    y0 = int(np.clip(y + shrink * h, 0, h_img - 1))
    y1 = int(np.clip(y + (1 - shrink) * h, y0 + 1, h_img))

    region_conf = confidence[y0:y1, x0:x1]
    region_disp = disparity_raw[y0:y1, x0:x1]
    region_z = points_3d[y0:y1, x0:x1, 2]

    valid = (region_conf > min_confidence) & (region_disp > 0) & np.isfinite(region_z)
    if not np.any(valid):
        return None
    return float(np.median(region_z[valid]))


def main():
    args = parse_args()
    calib = load_and_report(args.calib)
    image_size = (int(calib["image_width"]), int(calib["image_height"]))
    (map_lx, map_ly), (map_rx, map_ry) = build_rectify_maps(calib, image_size)

    mode = cv.STEREO_SGBM_MODE_SGBM_3WAY if args.fast else cv.STEREO_SGBM_MODE_HH
    left_matcher, right_matcher = make_matchers(mode=mode)
    wls_filter = make_wls_filter(left_matcher)

    baseline_mm = float(calib.get("baseline_mm", config.BASELINE_MM))
    focal_px = float(calib["camera_matrix_left"][0, 0])

    detector = YoloXDetector(args.model, conf_threshold=args.confidence, nms_threshold=args.nms)

    with StereoCamera(index=args.camera_index) as cam:
        while True:
            ok, left, right = cam.read()
            if not ok:
                print("Frame grab failed, retrying...")
                continue

            rect_l = cv.remap(left, map_lx, map_ly, cv.INTER_LINEAR)
            rect_r = cv.remap(right, map_rx, map_ry, cv.INTER_LINEAR)
            gray_l = cv.cvtColor(rect_l, cv.COLOR_BGR2GRAY)
            gray_r = cv.cvtColor(rect_r, cv.COLOR_BGR2GRAY)

            disparity_raw, confidence = compute_filtered_disparity(
                left_matcher, right_matcher, wls_filter, gray_l, gray_r, rect_l
            )
            points_3d = cv.reprojectImageTo3D(disparity_raw, calib["Q"])

            detections = detector.detect(rect_l)

            vis = rect_l.copy()
            for det in detections:
                x, y, w, h = det["box"]
                depth_mm = box_depth_mm(disparity_raw, points_3d, confidence, det["box"])

                x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
                cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

                if depth_mm is not None:
                    dz = depth_uncertainty_mm(depth_mm, baseline_mm, focal_px)
                    label = f"{det['class_name']} {det['score']:.2f}  {depth_mm/10:.0f}+/-{dz/10:.0f}cm"
                else:
                    label = f"{det['class_name']} {det['score']:.2f}  depth: n/a"

                (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv.rectangle(vis, (x0, y0 - th - 6), (x0 + tw + 4, y0), (0, 255, 0), -1)
                cv.putText(vis, label, (x0 + 2, y0 - 4), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

            cv.imshow("object depth", vis)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
