"""Shared calibration-loading and rectification helpers for the live viewers."""

import os

import cv2 as cv

from . import config
from .calibration_io import load_calibration


def resolve_calib_path(explicit=None):
    if explicit:
        return explicit, False
    if os.path.exists(config.CALIB_FILE):
        return config.CALIB_FILE, False
    if os.path.exists(config.SAMPLE_CALIB_FILE):
        return config.SAMPLE_CALIB_FILE, True
    raise SystemExit(
        "No calibration file found. Run `python generate_sample_calibration.py` "
        "for a quick nominal calibration, or `python calibrate.py` for a real one."
    )


def load_and_report(explicit=None):
    """Loads the calibration file, printing whether it's real or the nominal sample."""
    calib_path, is_sample = resolve_calib_path(explicit)
    calib = load_calibration(calib_path)
    if is_sample or calib.get("is_sample", 0):
        print(f"WARNING: using nominal sample calibration ({calib_path}). "
              "Depth values are approximate -- run calibrate.py for real numbers.")
    else:
        print(f"Loaded calibration: {calib_path} (stereo RMSE: {calib.get('rmse', -1):.4f} px)")
    return calib


def build_rectify_maps(calib, image_size):
    map_l = cv.initUndistortRectifyMap(
        calib["camera_matrix_left"], calib["dist_coeffs_left"],
        calib["R1"], calib["P1"], image_size, cv.CV_16SC2,
    )
    map_r = cv.initUndistortRectifyMap(
        calib["camera_matrix_right"], calib["dist_coeffs_right"],
        calib["R2"], calib["P2"], image_size, cv.CV_16SC2,
    )
    return map_l, map_r
