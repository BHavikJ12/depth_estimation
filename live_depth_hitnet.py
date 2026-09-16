"""Live stereo depth viewer using HITNET instead of SGBM.

Same job and same controls as live_depth.py -- rectify the AR0144's live
left/right feed, produce dense depth, click a pixel to read its distance --
but the disparity comes from the HITNET network (stereo/hitnet.py) rather
than StereoSGBM. HITNET fills textureless regions (blank walls, tabletops)
that SGBM leaves as holes, at the cost of needing a GPU to stay real-time.

Usage:
    python live_depth_hitnet.py
    python live_depth_hitnet.py --model models/hitnet/eth3d/240x320/model_float32.onnx
    python live_depth_hitnet.py --compare        # HITNET and SGBM side by side

Controls (with a window focused):
    click - report the distance at that pixel
    q     - quit
"""

import argparse
import time

import cv2 as cv
import numpy as np

from stereo import config
from stereo.camera import StereoCamera
from stereo.rectify import load_and_report, build_rectify_maps
from stereo.hitnet import HitNet, colorize_disparity
from stereo.depth_error import depth_uncertainty_mm

# eth3d at 240x320 is the default because it is the only combination that
# keeps up with the camera on a GTX 1650 (~26 FPS against the sensor's 30),
# and downscaling a 1280px eye to 320px means its 128px trained disparity
# range covers 512px at full resolution -- roughly 0.1 m to 12 m on this
# 52mm baseline. See HITNET.md for the speed/accuracy table; the 480x640
# models are about 1.5x more accurate if you can live with 3-8 FPS.
DEFAULT_MODEL = "models/hitnet/eth3d/240x320/model_float32.onnx"

last_click = {"px": None}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HITNET ONNX model path")
    p.add_argument("--calib", default=None, help="calibration YAML (defaults: real, else sample)")
    p.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--cpu", action="store_true", help="force CPU inference even if CUDA is available")
    p.add_argument("--max-depth-mm", type=float, default=5000.0,
                   help="furthest depth to trust and to colour")
    p.add_argument("--min-depth-mm", type=float, default=250.0,
                   help="nearest depth to trust; anything closer is treated as a mismatch")
    p.add_argument("--compare", action="store_true",
                   help="also run the SGBM+WLS pipeline and show both disparity maps")
    return p.parse_args()


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        last_click["px"] = (x, y)


def annotate_click(canvas, points_3d, baseline_mm, focal_px):
    """Draws the clicked pixel's triangulated distance, with the Teledyne
    dZ = Z^2/(B*f)*dd uncertainty so the number is not falsely precise."""
    if last_click["px"] is None:
        return
    x, y = last_click["px"]
    h, w = points_3d.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return

    xyz = points_3d[y, x]
    cv.circle(canvas, (x, y), 5, (255, 255, 255), 2)
    if not np.isfinite(xyz).all():
        cv.putText(canvas, "no depth", (x + 10, y), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        return

    depth_mm = float(np.linalg.norm(xyz))
    dz = depth_uncertainty_mm(depth_mm, baseline_mm, focal_px)
    cv.putText(canvas, f"{depth_mm:.0f} +/- {dz:.0f} mm", (x + 10, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    args = parse_args()
    calib = load_and_report(args.calib)

    image_size = (int(calib["image_width"]), int(calib["image_height"]))
    (map_lx, map_ly), (map_rx, map_ry) = build_rectify_maps(calib, image_size)

    baseline_mm = float(calib.get("baseline_mm", config.BASELINE_MM))
    focal_px = float(calib["camera_matrix_left"][0, 0])

    providers = ["CPUExecutionProvider"] if args.cpu else None
    net = HitNet(args.model, providers=providers)
    print(net.describe())
    if net.providers[0] == "CPUExecutionProvider" and not args.cpu:
        print("NOTE: running on CPU -- expect ~1 FPS at 480x640. "
              "Use --model .../240x320/... for something interactive.")

    # HITNET regresses a disparity for every pixel and never reports "no
    # match", so the working depth range is the only rejection signal there
    # is. Converting it to disparity bounds once (d = f*B/Z) also fixes the
    # colour scale, which keeps the palette stable frame to frame instead of
    # shimmering under per-frame re-normalisation.
    disp_floor = focal_px * baseline_mm / args.max_depth_mm
    disp_ceiling = focal_px * baseline_mm / args.min_depth_mm
    print(f"Trusting {args.min_depth_mm:.0f}-{args.max_depth_mm:.0f} mm "
          f"= {disp_floor:.1f}-{disp_ceiling:.1f} px disparity "
          f"(f={focal_px:.1f} px, B={baseline_mm:.1f} mm)")
    model_max_disp = net.max_disparity_at(image_size)
    if disp_ceiling > model_max_disp:
        closest_mm = focal_px * baseline_mm / model_max_disp
        print(f"NOTE: this model tops out at {model_max_disp:.0f} px disparity, "
              f"so it cannot see closer than ~{closest_mm:.0f} mm. Raise "
              "--min-depth-mm, or use middlebury_d400 (widest disparity range) "
              "at a larger input size.")

    if args.compare:
        from stereo.matching import make_matchers, make_wls_filter, compute_filtered_disparity
        left_matcher, right_matcher = make_matchers(mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
        wls_filter = make_wls_filter(left_matcher)

    cv.namedWindow("hitnet depth", cv.WINDOW_NORMAL)
    cv.setMouseCallback("hitnet depth", on_mouse)

    fps, frames, t0 = 0.0, 0, time.time()

    with StereoCamera(index=args.camera_index) as cam:
        while True:
            ok, left, right = cam.read()
            if not ok:
                print("Frame grab failed, retrying...")
                continue

            rect_l = cv.remap(left, map_lx, map_ly, cv.INTER_LINEAR)
            rect_r = cv.remap(right, map_rx, map_ry, cv.INTER_LINEAR)

            disparity, valid = net.compute(rect_l, rect_r)

            valid &= (disparity > disp_floor) & (disparity < disp_ceiling)

            points_3d = cv.reprojectImageTo3D(disparity, calib["Q"])
            points_3d[~valid] = np.nan

            depth_color = colorize_disparity(disparity, valid, max_disparity=disp_ceiling)
            annotate_click(depth_color, points_3d, baseline_mm, focal_px)

            frames += 1
            if frames % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - t0)
                t0 = now
            cv.putText(depth_color, f"{fps:.1f} FPS  {net.info.family} "
                                    f"{net.info.input_width}x{net.info.input_height}",
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv.imshow("rectified left", rect_l)
            cv.imshow("hitnet depth", depth_color)

            if args.compare:
                gray_l = cv.cvtColor(rect_l, cv.COLOR_BGR2GRAY)
                gray_r = cv.cvtColor(rect_r, cv.COLOR_BGR2GRAY)
                sgbm_disp, conf = compute_filtered_disparity(
                    left_matcher, right_matcher, wls_filter, gray_l, gray_r, rect_l
                )
                sgbm_valid = (conf > 100) & (sgbm_disp > 0)
                cv.imshow("sgbm depth",
                          colorize_disparity(sgbm_disp, sgbm_valid, max_disparity=disp_ceiling))

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
