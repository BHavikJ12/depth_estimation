"""Turning relative depth into metres.

V3 predicts depth up to an unknown scale that it re-chooses on every frame.
Multiplying by a constant found once is therefore *not* enough for live video:
pan the camera and the constant is wrong again. Three strategies, from the one
that needs nothing to the one that actually holds up:

  none    leave it relative. Honest, and correct for "which of these is nearer".

  anchor  one point at a known distance fixes the scale. Simple, exact at the
          moment you set it, and it drifts with the model's own normalisation
          as the scene changes. Good for a static camera.

  ground  the camera's height above a flat floor is measured once with a tape,
          then re-derived from the depth map on *every* frame and the scale set
          so the two agree. The scene may change however it likes; as long as
          some floor is visible, the metre stays a metre. This is the mode to
          use on anything that moves.

  model   the V2-Metric networks output metres directly; nothing to do.

`ground` is a plane fit, so it needs intrinsics -- which V3 also predicts, so
on a V3 model the whole chain runs with no user input beyond the tape measure.
"""

import numpy as np

from geometry import backproject


class ScaleEstimator:
    """Maps a relative depth map to metres and keeps the factor steady."""

    def __init__(self, mode="none", camera_height=None, smooth=0.15,
                 ground_frac=0.45, max_tilt_deg=45.0):
        if mode not in ("none", "anchor", "ground", "model"):
            raise ValueError(f"unknown scale mode {mode!r}")
        self.mode = mode
        self.camera_height = camera_height
        self.smooth = float(smooth)     # EMA weight for a fresh estimate
        self.ground_frac = ground_frac  # bottom fraction of frame searched for floor
        self.max_tilt = max_tilt_deg
        self.scale = None               # metres per relative unit
        self.status = "relative"
        self.inliers = 0
        self.plane = None               # (normal, distance) in camera frame

        if mode == "ground" and not camera_height:
            raise ValueError("ground mode needs --camera-height in metres")
        if mode == "model":
            self.scale, self.status = 1.0, "metric (model)"

    # ---- anchoring ----------------------------------------------------

    def set_anchor(self, raw_depth_at_point, true_range_m):
        """Pin the scale from one pixel at a tape-measured distance."""
        if raw_depth_at_point <= 0 or true_range_m <= 0:
            return False
        self.scale = float(true_range_m) / float(raw_depth_at_point)
        self.status = f"anchored @{true_range_m:g}m"
        return True

    # ---- per-frame ground plane ---------------------------------------

    def update(self, depth, cam):
        """Re-derive the scale for this frame. No-op outside `ground` mode."""
        if self.mode != "ground":
            return self.scale
        s = self._scale_from_ground(depth, cam)
        if s is None:
            self.status = f"ground lost ({self.inliers} pts)"
            return self.scale
        # A fresh plane fit is noisy frame to frame; the EMA trades a little
        # lag for a number that does not flicker in the third digit.
        self.scale = s if self.scale is None else (
            (1 - self.smooth) * self.scale + self.smooth * s)
        self.status = f"ground {self.camera_height:g}m ({self.inliers} pts)"
        return self.scale

    def _scale_from_ground(self, depth, cam):
        """RANSAC a plane through the lower part of the frame, return h_true/h_raw.

        Only the bottom `ground_frac` is searched: on any camera pointed
        roughly forward the floor lives there, and excluding the top two thirds
        removes the walls and ceiling that otherwise win the vote in a corridor.
        """
        h = depth.shape[0]
        y0 = int(h * (1.0 - self.ground_frac))
        sub = depth[y0:]
        if sub.size < 400:
            return None

        cam_sub = cam.for_frame(depth.shape[1], depth.shape[0])
        pts = backproject(sub, _shift_cy(cam_sub, y0), stride=max(1, sub.shape[1] // 160))
        if len(pts) < 200:
            self.inliers = len(pts)
            return None

        normal, dist, inl = _ransac_plane(pts, thresh_frac=0.01, iters=60)
        self.inliers = int(inl.sum()) if inl is not None else 0
        if normal is None or self.inliers < 100:
            return None

        # The floor's normal must point up the camera's -Y axis. Anything more
        # than `max_tilt` off is a wall or a table edge, not the ground.
        if normal[1] > 0:
            normal, dist = -normal, -dist
        tilt = np.degrees(np.arccos(np.clip(-normal[1], -1, 1)))
        if tilt > self.max_tilt:
            return None

        h_raw = abs(dist)   # distance from the camera centre to the plane
        if h_raw <= 1e-6:
            return None
        self.plane = (normal, dist)
        return float(self.camera_height) / h_raw

    # ---- applying -----------------------------------------------------

    def to_metres(self, depth):
        """Scaled depth, or None when no scale is known."""
        if self.scale is None:
            return None
        if self.mode == "model":
            return depth
        return depth * self.scale

    @property
    def known(self):
        return self.scale is not None


def _shift_cy(cam, y0):
    """A crop's intrinsics: same focal length, principal point moved up by y0."""
    from geometry import CameraModel
    return CameraModel(cam.fx, cam.fy, cam.cx, cam.cy - y0,
                       cam.width, cam.height - y0, cam.source + f" +crop{y0}")


def _ransac_plane(pts, thresh_frac=0.01, iters=60, rng=None):
    """Plane through a point cloud, tolerant of the furniture standing on it.

    The inlier threshold is a fraction of the median depth rather than an
    absolute distance, because these points are in relative units whose
    magnitude changes every frame -- a fixed 2 cm threshold would mean
    everything on one frame and nothing on the next.
    """
    rng = rng or np.random.default_rng(0)
    n = len(pts)
    if n < 3:
        return None, None, None
    thresh = max(float(np.median(np.abs(pts[:, 2]))) * thresh_frac, 1e-9)

    best_n, best_d, best_inl, best_count = None, None, None, 0
    idx = rng.integers(0, n, size=(iters, 3))
    for tri in idx:
        p0, p1, p2 = pts[tri]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm)
        if norm < 1e-12:
            continue
        nrm = nrm / norm
        d = -float(nrm @ p0)
        inl = np.abs(pts @ nrm + d) < thresh
        c = int(inl.sum())
        if c > best_count:
            best_n, best_d, best_inl, best_count = nrm, d, inl, c

    if best_n is None or best_count < 3:
        return None, None, None

    # Least-squares refit on the inliers: RANSAC picked the right points from a
    # random triple, and the triple itself is a poor estimate of the plane.
    q = pts[best_inl]
    centroid = q.mean(0)
    _, _, vt = np.linalg.svd(q - centroid, full_matrices=False)
    nrm = vt[-1] / np.linalg.norm(vt[-1])
    d = -float(nrm @ centroid)
    inl = np.abs(pts @ nrm + d) < thresh
    return nrm, d, inl
