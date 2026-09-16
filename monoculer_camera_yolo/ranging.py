"""Range from a bounding box and a known target size (the pinhole size prior).

With one camera and no motion parallax there is no depth cue for a drone
against open sky, so range has to come from an assumed physical size:

    Z = f_px * S_real / s_px

`S_real` is the drone's real span (0.5 m by default, a typical motor-to-motor
diagonal) and `s_px` is that span as it appears in the image.

Two things about this that matter more than the formula:

* Error scales as 1/s_px. At 30 px the box is worth ~3% per pixel; at 8 px it
  is ~12% per pixel. Past a few hundred pixels of span the estimate is a guess
  with a decimal point on it, so `RangeEstimate` carries its own sigma and the
  overlay draws it.
* A quadcopter's *apparent* span shrinks with viewing angle, never grows. The
  bias is therefore one-sided: foreshortening makes the target look far, never
  near. `max(w, h)` is used by default because it is the least foreshortened of
  the two box dimensions.
"""

import math
from dataclasses import dataclass

DEFAULT_TARGET_SIZE_M = 0.5      # 500 mm motor-to-motor span
DEFAULT_BOX_SIGMA_PX = 2.0       # 1-sigma bounding box edge jitter
DEFAULT_SIZE_CV = 0.20           # 1-sigma spread of the size prior, as a fraction

SPAN_MODES = ("max", "width", "height", "diag")


@dataclass
class RangeEstimate:
    """Metric range plus where the target is in the camera frame."""
    range_m: float
    sigma_m: float            # 1-sigma, from box jitter and size-prior spread
    span_px: float            # apparent span actually used
    x_m: float                # camera frame: +X right, +Y down, +Z forward
    y_m: float
    z_m: float
    azimuth_deg: float        # + is right of boresight
    elevation_deg: float      # + is above boresight

    @property
    def range_lo(self):
        return max(0.0, self.range_m - self.sigma_m)

    @property
    def range_hi(self):
        return self.range_m + self.sigma_m


def apparent_span(box, mode="max"):
    """Pixel span of a box (x1, y1, x2, y2) under the chosen convention."""
    x1, y1, x2, y2 = box
    w, h = abs(x2 - x1), abs(y2 - y1)
    if mode == "width":
        return w
    if mode == "height":
        return h
    if mode == "diag":
        return math.hypot(w, h)
    if mode == "max":
        return max(w, h)
    raise ValueError(f"span mode must be one of {SPAN_MODES}, got {mode!r}")


class RangeEstimator:
    """Turns boxes into ranges for one target class of known size."""

    def __init__(self, camera, target_size_m=DEFAULT_TARGET_SIZE_M, span_mode="max",
                 box_sigma_px=DEFAULT_BOX_SIGMA_PX, size_cv=DEFAULT_SIZE_CV,
                 min_span_px=3.0):
        if target_size_m <= 0:
            raise ValueError("target_size_m must be positive")
        self.camera = camera
        self.target_size_m = float(target_size_m)
        self.span_mode = span_mode
        self.box_sigma_px = float(box_sigma_px)
        self.size_cv = float(size_cv)
        self.min_span_px = float(min_span_px)

    def estimate(self, box):
        """Range a single (x1, y1, x2, y2) box. Returns None if it is too small
        to carry any range information."""
        span = apparent_span(box, self.span_mode)
        if span < self.min_span_px:
            return None

        cam = self.camera
        f = cam.fx if self.span_mode == "width" else math.sqrt(cam.fx * cam.fy)
        z = f * self.target_size_m / span

        # Relative error adds in quadrature: how well we located the box edges,
        # and how well 0.5 m describes this particular airframe. The box term
        # uses sqrt(2) because both edges of the span are independently jittered.
        rel_box = (math.sqrt(2.0) * self.box_sigma_px) / span
        rel = math.hypot(rel_box, self.size_cv)
        sigma = z * rel

        u = (box[0] + box[2]) / 2.0
        v = (box[1] + box[3]) / 2.0
        x = (u - cam.cx) * z / cam.fx
        y = (v - cam.cy) * z / cam.fy

        return RangeEstimate(
            range_m=math.sqrt(x * x + y * y + z * z),
            sigma_m=sigma,
            span_px=span,
            x_m=x, y_m=y, z_m=z,
            azimuth_deg=math.degrees(math.atan2(u - cam.cx, cam.fx)),
            elevation_deg=math.degrees(math.atan2(-(v - cam.cy), cam.fy)),
        )

    def max_useful_range(self, rel_error=0.25):
        """Range at which box jitter alone costs `rel_error` — the point past
        which the number stops meaning anything."""
        budget = rel_error ** 2 - self.size_cv ** 2
        if budget <= 0:
            return float("inf")   # the size prior dominates at every range
        span = math.sqrt(2.0) * self.box_sigma_px / math.sqrt(budget)
        f = math.sqrt(self.camera.fx * self.camera.fy)
        return f * self.target_size_m / span


def solve_target_size(camera, box, known_range_m, span_mode="max"):
    """Back out the effective target size from one box measured at a known range.

    Point the camera at the drone, put a tape measure or an RTK fix on it, and
    feed the result back in as --target-size. This absorbs the airframe's real
    span, the detector's box-tightness bias, and the viewing angle in one
    number, which is worth far more than arguing about the true motor spacing.
    """
    span = apparent_span(box, span_mode)
    f = camera.fx if span_mode == "width" else math.sqrt(camera.fx * camera.fy)
    return known_range_m * span / f
