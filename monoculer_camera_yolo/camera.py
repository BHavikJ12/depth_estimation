"""Monocular camera model: whatever the user can supply -> focal length in pixels.

Range-from-size needs exactly one number, the focal length expressed in pixels
*of the frame the bounding box was measured in*. That number can come from
three places, in descending order of trustworthiness:

  1. A stereo/mono calibration YAML (this repo's calibration_data/*.yaml).
  2. An explicitly measured fx.
  3. The lens' horizontal field of view, from the datasheet.

If the frame is resized after calibration, fx scales with it, which `for_frame`
handles so the caller never has to think about it.
"""

import math

import cv2 as cv
import numpy as np


class CameraModel:
    """Pinhole intrinsics for a single camera, in pixels."""

    def __init__(self, fx, fy, cx, cy, width, height, dist_coeffs=None, source="manual"):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)
        self.dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).ravel()
        self.source = source

    # ---- constructors -------------------------------------------------

    @classmethod
    def from_fov(cls, width, height, hfov_deg, vfov_deg=None):
        """Intrinsics from the lens' field of view. Assumes square pixels if
        only the horizontal FOV is known."""
        if not 1.0 < hfov_deg < 179.0:
            raise ValueError(f"hfov_deg must be in (1, 179), got {hfov_deg}")
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        fy = fx if vfov_deg is None else (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        return cls(fx, fy, width / 2.0, height / 2.0, width, height,
                   source=f"hfov={hfov_deg:g}deg")

    @classmethod
    def from_fx(cls, width, height, fx, fy=None):
        return cls(fx, fy or fx, width / 2.0, height / 2.0, width, height, source="fx")

    @classmethod
    def from_calibration(cls, path, camera="left"):
        """Read an OpenCV FileStorage YAML as written by ../calibrate.py."""
        fs = cv.FileStorage(str(path), cv.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"cannot open calibration file: {path}")
        try:
            K = fs.getNode(f"camera_matrix_{camera}").mat()
            if K is None:
                K = fs.getNode("camera_matrix").mat()
            if K is None:
                raise ValueError(f"no camera_matrix_{camera} or camera_matrix in {path}")
            d_node = fs.getNode(f"dist_coeffs_{camera}")
            d = d_node.mat() if not d_node.empty() else fs.getNode("dist_coeffs").mat()
            w = int(fs.getNode("image_width").real() or 0)
            h = int(fs.getNode("image_height").real() or 0)
        finally:
            fs.release()
        if w == 0 or h == 0:
            # Fall back to the principal point, which sits near the centre.
            w, h = int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2))
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], w, h, d,
                   source=f"{path}:{camera}")

    # ---- use ----------------------------------------------------------

    def for_frame(self, width, height):
        """Rescale the intrinsics to a frame of a different resolution.

        Calibrating at 1280x720 and then running the detector on a 640x360
        stream silently halves every focal length; this makes that explicit.
        """
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return CameraModel(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                           width, height, self.dist_coeffs,
                           source=f"{self.source} scaled {self.width}x{self.height}->{width}x{height}")

    @property
    def hfov_deg(self):
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self):
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))

    def __repr__(self):
        return (f"CameraModel(fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, "
                f"cy={self.cy:.1f}, {self.width}x{self.height}, hfov={self.hfov_deg:.1f}deg, "
                f"source={self.source!r})")
