"""Interactive stereo calibration from the AR0144 live feed.

Since both eyes come out of ONE hardware-synchronized frame, a single
capture pass gives us everything the temugeb tutorial needed three passes
for: per-eye intrinsics AND stereo extrinsics, from the same image set.

Usage:
    python calibrate.py --num-images 20 --square-size 25.0

Controls while the preview window is focused:
    c   - capture the current frame pair (only works when the checkerboard
          is detected in BOTH eyes)
    q   - stop capturing early and run calibration on what you have
"""

import argparse
import os

import cv2 as cv
import numpy as np

from stereo import config
from stereo.camera import StereoCamera
from stereo.calibration_io import save_calibration


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-images", type=int, default=20, help="target number of good pairs to capture")
    p.add_argument("--rows", type=int, default=config.CHESSBOARD_ROWS, help="inner corner rows")
    p.add_argument("--cols", type=int, default=config.CHESSBOARD_COLUMNS, help="inner corner columns")
    p.add_argument("--square-size", type=float, default=config.SQUARE_SIZE_MM,
                   help="real-world size of one checkerboard square, in mm")
    p.add_argument("--output", default=config.CALIB_FILE, help="output calibration YAML path")
    p.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    return p.parse_args()


def find_corners(gray, pattern_size, criteria):
    ok, corners = cv.findChessboardCorners(gray, pattern_size, None)
    if ok:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ok, corners


def capture_pairs(cam, pattern_size, num_images, criteria):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints, imgpoints_left, imgpoints_right = [], [], []

    print(f"Show the checkerboard to both cameras. Press 'c' to capture "
          f"(need {num_images} pairs), 'q' to stop early.")

    while len(objpoints) < num_images:
        ok, left, right = cam.read()
        if not ok:
            print("Frame grab failed, retrying...")
            continue

        gray_l = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
        gray_r = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
        found_l, corners_l = find_corners(gray_l, pattern_size, criteria)
        found_r, corners_r = find_corners(gray_r, pattern_size, criteria)

        vis_l, vis_r = left.copy(), right.copy()
        if found_l:
            cv.drawChessboardCorners(vis_l, pattern_size, corners_l, found_l)
        if found_r:
            cv.drawChessboardCorners(vis_r, pattern_size, corners_r, found_r)

        status = f"captured {len(objpoints)}/{num_images}  detected: L={found_l} R={found_r}"
        combined = np.hstack([vis_l, vis_r])
        cv.putText(combined, status, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.imshow("calibration (c=capture, q=quit)", combined)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and found_l and found_r:
            objpoints.append(objp.copy())
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            print(f"  captured pair {len(objpoints)}/{num_images}")

    cv.destroyAllWindows()
    return objpoints, imgpoints_left, imgpoints_right


def main():
    args = parse_args()
    pattern_size = (args.rows, args.cols)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    with StereoCamera(index=args.camera_index) as cam:
        objpoints, imgpoints_left, imgpoints_right = capture_pairs(
            cam, pattern_size, args.num_images, criteria
        )

    if len(objpoints) < 5:
        raise SystemExit(f"Only captured {len(objpoints)} pairs; need at least ~5-10 for a "
                          "usable calibration. Re-run and hold the board steady at varied "
                          "angles/distances/positions in the frame.")

    for p in objpoints:
        p *= args.square_size

    image_size = config.IMAGE_SIZE

    print("Calibrating left camera...")
    rmse_l, mtx_l, dist_l, _, _ = cv.calibrateCamera(objpoints, imgpoints_left, image_size, None, None)
    print(f"  left RMSE: {rmse_l:.4f} px")

    print("Calibrating right camera...")
    rmse_r, mtx_r, dist_r, _, _ = cv.calibrateCamera(objpoints, imgpoints_right, image_size, None, None)
    print(f"  right RMSE: {rmse_r:.4f} px")

    print("Running stereo calibration...")
    flags = cv.CALIB_FIX_INTRINSIC
    rmse_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l, mtx_r, dist_r, image_size,
        criteria=criteria, flags=flags,
    )
    print(f"  stereo RMSE: {rmse_stereo:.4f} px")
    # Per Teledyne's embedded stereo guide: "typical RMS error for good
    # calibration should be below 0.5 pixel".
    if rmse_stereo < 0.5:
        print(f"  calibration quality: GOOD (RMSE {rmse_stereo:.4f} px < 0.5 px)")
    else:
        print(f"  calibration quality: POOR (RMSE {rmse_stereo:.4f} px >= 0.5 px) -- "
              "recapture with more/varied checkerboard poses (corners, tilts, "
              "full frame coverage) before trusting depth from this calibration.")
    baseline = float(np.linalg.norm(T))
    print(f"  recovered baseline: {baseline:.2f} mm (datasheet: {config.BASELINE_MM} mm)")

    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T,
        flags=cv.CALIB_ZERO_DISPARITY, alpha=0,
    )

    data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "baseline_mm": baseline,
        "camera_matrix_left": mtx_l, "dist_coeffs_left": dist_l,
        "camera_matrix_right": mtx_r, "dist_coeffs_right": dist_r,
        "R": R, "T": T, "E": E, "F": F,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "rmse": rmse_stereo,
        "is_sample": 0,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_calibration(args.output, data)
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()
