"""Save/load stereo calibration data as a human-readable OpenCV YAML file."""

import cv2 as cv


FIELDS = [
    "image_width", "image_height", "baseline_mm",
    "camera_matrix_left", "dist_coeffs_left",
    "camera_matrix_right", "dist_coeffs_right",
    "R", "T", "E", "F",
    "R1", "R2", "P1", "P2", "Q",
    "rmse", "is_sample",
]


def save_calibration(path: str, data: dict):
    """data must contain (at least) all keys OpenCV matrices in FIELDS."""
    fs = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    for key in FIELDS:
        if key in data:
            fs.write(key, data[key])
    fs.release()


def load_calibration(path: str) -> dict:
    fs = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {path}")
    data = {}
    for key in FIELDS:
        node = fs.getNode(key)
        if node.empty():
            continue
        if node.isReal() or node.isInt():
            data[key] = node.real() if node.isReal() else int(node.real())
        else:
            data[key] = node.mat()
    fs.release()
    return data
