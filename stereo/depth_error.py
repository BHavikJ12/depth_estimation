"""Stereo depth-accuracy relationship.

From Teledyne's embedded stereo system guide:
https://www.teledynevisionsolutions.com/en-in/learn/learning-center/machine-vision/how-to-build-a-custom-embedded-stereo-system-for-depth-perception

    dZ = Z^2 / (B * f) * dd

Depth error grows with the SQUARE of range: a point twice as far away has
4x the depth uncertainty for the same pixel-level disparity error. This is
the fundamental reason stereo works well up close and gets unreliable at
distance, and why a wider baseline (B) or higher resolution/focal length
(f) both directly buy back accuracy.
"""


def depth_uncertainty_mm(depth_mm: float, baseline_mm: float, focal_px: float,
                          disparity_error_px: float = 0.5) -> float:
    """dZ: expected depth error (mm) at a given depth, per the stereo accuracy formula.

    disparity_error_px defaults to 0.5 -- a typical block-matching sub-pixel
    accuracy figure absent a per-pixel confidence estimate.
    """
    return (depth_mm ** 2) / (baseline_mm * focal_px) * disparity_error_px


def max_reliable_depth_mm(baseline_mm: float, focal_px: float,
                           disparity_error_px: float = 0.5,
                           max_relative_error: float = 0.05) -> float:
    """Depth at which the uncertainty exceeds `max_relative_error` of the
    reading itself (default 5%). Solves dZ/Z = max_relative_error for Z."""
    return baseline_mm * focal_px * max_relative_error / disparity_error_px
