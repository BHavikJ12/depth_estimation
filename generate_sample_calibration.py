"""Generate a NOMINAL calibration file from the AR0144 datasheet specs.

This is a placeholder, not a real calibration: it estimates the camera
matrix from focal length / pixel pitch, assumes zero distortion (the specs
claim <0.2%, close enough to bootstrap) and a pure-X baseline translation.
It lets live_depth.py run out of the box for testing, but depth values from
it should NOT be trusted -- run `calibrate.py` against a real checkerboard
and use calibration_data/stereo_calibration.yaml for anything quantitative.
"""

import numpy as np
import cv2 as cv

from stereo import config
from stereo.calibration_io import save_calibration

f_px = config.ESTIMATED_FOCAL_PX
cx = config.SINGLE_WIDTH / 2.0
cy = config.SINGLE_HEIGHT / 2.0

camera_matrix = np.array([
    [f_px, 0.0, cx],
    [0.0, f_px, cy],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
dist_coeffs = np.zeros((1, 5), dtype=np.float64)

R = np.eye(3, dtype=np.float64)
T = np.array([[-config.BASELINE_MM], [0.0], [0.0]], dtype=np.float64)

Tx = np.array([
    [0, -T[2, 0], T[1, 0]],
    [T[2, 0], 0, -T[0, 0]],
    [-T[1, 0], T[0, 0], 0],
], dtype=np.float64)
E = Tx @ R
F = np.linalg.inv(camera_matrix).T @ E @ np.linalg.inv(camera_matrix)

R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
    camera_matrix, dist_coeffs, camera_matrix, dist_coeffs,
    config.IMAGE_SIZE, R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0,
)

data = {
    "image_width": config.SINGLE_WIDTH,
    "image_height": config.SINGLE_HEIGHT,
    "baseline_mm": config.BASELINE_MM,
    "camera_matrix_left": camera_matrix,
    "dist_coeffs_left": dist_coeffs,
    "camera_matrix_right": camera_matrix,
    "dist_coeffs_right": dist_coeffs,
    "R": R, "T": T, "E": E, "F": F,
    "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
    "rmse": -1.0,
    "is_sample": 1,
}

import os
os.makedirs(config.CALIB_DIR, exist_ok=True)
save_calibration(config.SAMPLE_CALIB_FILE, data)
print(f"Wrote nominal/sample calibration to {config.SAMPLE_CALIB_FILE}")
print(f"  estimated focal length: {f_px:.1f} px  (from {config.FOCAL_LENGTH_MM} mm / "
      f"{config.SENSOR_PIXEL_SIZE_MM * 1000:.1f} um pixel pitch)")
print(f"  principal point: ({cx:.1f}, {cy:.1f})")
print(f"  baseline: {config.BASELINE_MM} mm")
print("This is NOT a real calibration -- run calibrate.py with a printed "
      "checkerboard for accurate depth.")
