"""Live stereo depth viewer for the AR0144 stereo USB camera.

Loads a calibration file (real one if present, else falls back to the
nominal sample), rectifies the live left/right feed, computes a dense
disparity map with StereoSGBM, and shows a depth-colored view. Click
anywhere in the depth window to print the triangulated distance (mm) at
that pixel.

Usage:
    python live_depth.py [--calib calibration_data/stereo_calibration.yaml]
"""

import argparse

import cv2 as cv
import numpy as np

from stereo import config
from stereo.camera import StereoCamera
from stereo.rectify import load_and_report, build_rectify_maps
from stereo.matching import make_matchers, make_wls_filter, compute_filtered_disparity
from stereo.depth_error import depth_uncertainty_mm

last_click = {"xyz": None, "px": None}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib", default=None, help="calibration YAML (defaults: real, else sample)")
    p.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--fast", action="store_true",
                   help="use 3-way SGM (faster, slightly less accurate) instead of full 8-direction SGM")
    return p.parse_args()


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points_3d = param
        xyz = points_3d[y, x]
        last_click["xyz"] = xyz
        last_click["px"] = (x, y)


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

    cv.namedWindow("depth")

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

            # Confidence map scores how much the left/right passes agreed;
            # low-confidence pixels are exactly the speckle noise, so drop them
            # instead of colorizing them.
            valid = (confidence > 100) & (disparity_raw > 0)

            disp_vis = np.zeros_like(disparity_raw, dtype=np.uint8)
            if np.any(valid):
                d_min, d_max = disparity_raw[valid].min(), disparity_raw[valid].max()
                if d_max > d_min:
                    disp_vis = np.clip(
                        (disparity_raw - d_min) / (d_max - d_min) * 255, 0, 255
                    ).astype(np.uint8)
            depth_color = cv.applyColorMap(disp_vis, cv.COLORMAP_JET)
            depth_color[~valid] = 0

            cv.setMouseCallback("depth", on_mouse, points_3d)
            if last_click["px"] is not None:
                x, y = last_click["px"]
                xyz = last_click["xyz"]
                cv.circle(depth_color, (x, y), 5, (255, 255, 255), 2)
                if xyz is not None and np.isfinite(xyz).all():
                    depth_mm = float(np.linalg.norm(xyz))
                    dz = depth_uncertainty_mm(depth_mm, baseline_mm, focal_px)
                    label = f"{depth_mm:.0f} +/- {dz:.0f} mm"
                    cv.putText(depth_color, label, (x + 10, y), cv.FONT_HERSHEY_SIMPLEX,
                               0.7, (255, 255, 255), 2)

            cv.imshow("rectified left", rect_l)
            cv.imshow("depth", depth_color)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
