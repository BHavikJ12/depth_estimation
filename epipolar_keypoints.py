"""Sparse keypoint depth viewer with REAL per-point epipolar lines.

An epipolar line is not a fixed horizontal grid line -- it is the specific
line, derived from the Fundamental matrix F, on which a point's true match
in the other image must lie. On raw (unrectified) images these lines are
generally slanted and converge toward the epipole; they only become
horizontal after a correct rectification. Drawing an arbitrary horizontal
grid (an earlier version of this script did that) is not the same thing and
doesn't prove anything about correspondence.

This version works on the RAW left/right frames (no rectification needed):

  1. Detect sparse, distinctive ORB keypoints in each eye and match them by
     descriptor similarity with a ratio test.
  2. Estimate the Fundamental matrix F fresh, every frame, from those matches
     via RANSAC (cv.findFundamentalMat). RANSAC's inlier mask is itself the
     correspondence filter -- a wrong match essentially never fits the same
     F as the rest, so this replaces the y-diff heuristic and does not
     depend on your calibration being accurate.
  3. For every inlier point in one image, compute its actual epipolar line
     in the other image (cv.computeCorrespondEpilines) and draw it -- the
     matched point should visibly sit on its own line.
  4. Triangulate the inlier pairs using the calibration's intrinsics/R/T
     (stereo/triangulation.triangulate_cv, which undistorts to normalized
     coordinates first) to get a stable per-point depth in mm.

Usage:
    python epipolar_keypoints.py [--calib ...] [--max-features 400]
"""

import argparse

import cv2 as cv
import numpy as np

from stereo import config
from stereo.camera import StereoCamera
from stereo.rectify import load_and_report
from stereo.triangulation import triangulate_cv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib", default=None, help="calibration YAML (defaults: real, else sample)")
    p.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--max-features", type=int, default=400, help="ORB keypoints per frame")
    p.add_argument("--ransac-thresh", type=float, default=2.0,
                   help="RANSAC epipolar-distance tolerance (px) when estimating F")
    p.add_argument("--min-depth-mm", type=float, default=50.0)
    p.add_argument("--max-depth-mm", type=float, default=8000.0)
    return p.parse_args()


def match_keypoints(orb, matcher, gray_l, gray_r):
    kp_l, des_l = orb.detectAndCompute(gray_l, None)
    kp_r, des_r = orb.detectAndCompute(gray_r, None)
    if des_l is None or des_r is None or len(kp_l) < 8 or len(kp_r) < 8:
        return kp_l, kp_r, []

    knn = matcher.knnMatch(des_l, des_r, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:      # Lowe's ratio test
            good.append(m)
    return kp_l, kp_r, good


def estimate_fundamental_inliers(kp_l, kp_r, matches, ransac_thresh):
    """Returns (pts_l, pts_r, F) for the RANSAC-inlier matches only."""
    if len(matches) < 8:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), None

    pts_l = np.float32([kp_l[m.queryIdx].pt for m in matches])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in matches])

    F, mask = cv.findFundamentalMat(pts_l, pts_r, cv.FM_RANSAC, ransac_thresh, 0.99)
    if F is None or mask is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), None

    mask = mask.ravel().astype(bool)
    return pts_l[mask], pts_r[mask], F


def line_endpoints(line, width):
    """line = (a, b, c) for a*x + b*y + c = 0. Returns two (x, y) endpoints
    spanning the full image width."""
    a, b, c = line
    if abs(b) > 1e-6:
        x0, y0 = 0, -c / b
        x1, y1 = width - 1, -(c + a * (width - 1)) / b
    else:
        x0 = x1 = -c / a if abs(a) > 1e-6 else 0
        y0, y1 = 0, 10_000
    return (int(x0), int(y0)), (int(x1), int(y1))


def draw_epilines_and_points(img, lines, points, colors):
    out = img.copy()
    w = out.shape[1]
    for line, pt, color in zip(lines, points, colors):
        p0, p1 = line_endpoints(line, w)
        cv.line(out, p0, p1, color, 1, cv.LINE_AA)
        cv.circle(out, (int(pt[0]), int(pt[1])), 4, color, -1, cv.LINE_AA)
    return out


def main():
    args = parse_args()
    calib = load_and_report(args.calib)

    orb = cv.ORB_create(nfeatures=args.max_features)
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    rng = np.random.default_rng(0)

    with StereoCamera(index=args.camera_index) as cam:
        while True:
            ok, left, right = cam.read()
            if not ok:
                print("Frame grab failed, retrying...")
                continue

            gray_l = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
            gray_r = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

            kp_l, kp_r, matches = match_keypoints(orb, matcher, gray_l, gray_r)
            pts_l, pts_r, F = estimate_fundamental_inliers(kp_l, kp_r, matches, args.ransac_thresh)

            vis_l, vis_r = left.copy(), right.copy()
            depths = []

            if F is not None and len(pts_l) > 0:
                # Epilines IN the right image FOR points FROM the left image (whichImage=1),
                # and vice versa -- this is OpenCV's convention for computeCorrespondEpilines.
                lines_r = cv.computeCorrespondEpilines(pts_l.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
                lines_l = cv.computeCorrespondEpilines(pts_r.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

                colors = rng.integers(0, 255, size=(len(pts_l), 3)).tolist()
                vis_l = draw_epilines_and_points(vis_l, lines_l, pts_l, colors)
                vis_r = draw_epilines_and_points(vis_r, lines_r, pts_r, colors)

                points_3d = triangulate_cv(
                    calib["camera_matrix_left"], calib["dist_coeffs_left"],
                    calib["camera_matrix_right"], calib["dist_coeffs_right"],
                    calib["R"], calib["T"], pts_l, pts_r,
                )
                for (x, y), p3d in zip(pts_l, points_3d):
                    depth = float(p3d[2])
                    if args.min_depth_mm <= depth <= args.max_depth_mm:
                        depths.append(depth)
                        cv.putText(vis_l, f"{depth:.0f}mm", (int(x) + 6, int(y) - 6),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

            canvas = np.hstack([vis_l, vis_r])
            summary = f"epipolar inliers: {len(pts_l)}/{len(matches)}"
            if depths:
                summary += f"  median depth: {np.median(depths):.0f} mm  std: {np.std(depths):.0f} mm"
            cv.putText(canvas, summary, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.putText(canvas, "each dot's true match must lie on its same-colored line in the other image",
                       (10, canvas.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv.imshow("epipolar lines + keypoint depth", canvas)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
