"""Pinhole intrinsics, back-projection, and PLY export.

A depth map on its own is a picture. It becomes geometry only once you know the
focal length, and there are four places that number can come from, in
descending order of trustworthiness:

  1. A calibration YAML from ../calibrate.py.
  2. A measured fx in pixels.
  3. The lens' horizontal FOV from the datasheet.
  4. Depth Anything V3's own per-frame prediction.

(4) is remarkable -- it needs nothing from the user and is usually within a few
percent -- but it is an estimate from image content, so it wanders as the scene
changes. For anything where the number matters, calibrate; V3's K is the
sensible default for a webcam nobody has measured.
"""

import math

import cv2 as cv
import numpy as np


class CameraModel:
    """Pinhole intrinsics in pixels, tied to one frame size."""

    def __init__(self, fx, fy, cx, cy, width, height, source="manual"):
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)
        self.width, self.height = int(width), int(height)
        self.source = source

    # ---- constructors -------------------------------------------------

    @classmethod
    def from_fov(cls, width, height, hfov_deg):
        if not 1.0 < hfov_deg < 179.0:
            raise ValueError(f"hfov must be in (1, 179), got {hfov_deg}")
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx, fx, width / 2.0, height / 2.0, width, height,
                   f"hfov={hfov_deg:g}deg")

    @classmethod
    def from_fx(cls, width, height, fx):
        return cls(fx, fx, width / 2.0, height / 2.0, width, height, "fx")

    @classmethod
    def from_matrix(cls, K, width, height, source="K"):
        K = np.asarray(K, np.float64).reshape(3, 3)
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height, source)

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
                raise ValueError(f"no camera_matrix in {path}")
            w = int(fs.getNode("image_width").real() or 0)
            h = int(fs.getNode("image_height").real() or 0)
        finally:
            fs.release()
        if w == 0 or h == 0:
            w, h = int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2))
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], w, h, f"{path}:{camera}")

    # ---- use ----------------------------------------------------------

    def for_frame(self, width, height):
        """Rescale to another resolution.

        The model runs at 504 px and the display frame is 1280 px wide; using
        one frame's intrinsics on the other silently scales every distance.
        """
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return CameraModel(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                           width, height,
                           f"{self.source} @{self.width}x{self.height}->{width}x{height}")

    @property
    def matrix(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], np.float64)

    @property
    def hfov_deg(self):
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self):
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))

    def bearing(self, u, v):
        """Pixel -> (azimuth, elevation) in degrees off boresight."""
        return (math.degrees(math.atan2((u - self.cx), self.fx)),
                math.degrees(math.atan2(-(v - self.cy), self.fy)))

    def __repr__(self):
        return (f"CameraModel(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"{self.width}x{self.height}, hfov={self.hfov_deg:.1f}deg, "
                f"src={self.source!r})")


def backproject(depth, cam, stride=1):
    """Depth map -> XYZ in the camera frame (+X right, +Y down, +Z forward).

    Returns an (N, 3) array of finite points, subsampled by `stride`. The image
    grid is built once per call; at 640x480 and stride 4 that is 19k points,
    which numpy does in about a millisecond.
    """
    d = depth[::stride, ::stride]
    h, w = d.shape
    us = (np.arange(w) * stride).astype(np.float32)
    vs = (np.arange(h) * stride).astype(np.float32)
    uu, vv = np.meshgrid(us, vs)
    z = d.astype(np.float32)
    x = (uu - cam.cx) * z / cam.fx
    y = (vv - cam.cy) * z / cam.fy
    pts = np.stack([x, y, z], -1).reshape(-1, 3)
    return pts[np.isfinite(pts).all(1) & (pts[:, 2] > 0)]


def backproject_rgb(depth, bgr, cam, stride=1, max_depth=None):
    """As `backproject`, but carries colour and an optional far clip."""
    d = depth[::stride, ::stride]
    c = bgr[::stride, ::stride]
    h, w = d.shape
    uu, vv = np.meshgrid((np.arange(w) * stride).astype(np.float32),
                         (np.arange(h) * stride).astype(np.float32))
    z = d.astype(np.float32)
    pts = np.stack([(uu - cam.cx) * z / cam.fx,
                    (vv - cam.cy) * z / cam.fy, z], -1).reshape(-1, 3)
    rgb = c.reshape(-1, 3)[:, ::-1]
    ok = np.isfinite(pts).all(1) & (pts[:, 2] > 0)
    if max_depth is not None:
        ok &= pts[:, 2] <= max_depth
    return pts[ok], rgb[ok]


def write_ply(path, pts, rgb=None):
    """Binary PLY, which every viewer reads and which is 4x smaller than ASCII."""
    n = len(pts)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if rgb is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header", ""]
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if rgb is not None:
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write("\n".join(header).encode())
        f.write(arr.tobytes())
    return n
