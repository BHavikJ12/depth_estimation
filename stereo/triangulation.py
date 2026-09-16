"""2D -> 3D triangulation from a pair of matched pixel observations.

Two implementations are provided:
  * `dlt`            - manual Direct Linear Transform via SVD (the classic
                        from-scratch approach, mirrors temugeb's tutorial).
  * `triangulate_cv` - the same result computed with cv.triangulatePoints,
                        which is faster and handles distortion via
                        cv.undistortPoints first. Prefer this one.
"""

import numpy as np
import cv2 as cv


def build_projection_matrices(cam_matrix_left, cam_matrix_right, R, T):
    """Left camera is the world origin; right camera is placed by (R, T)."""
    RT1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = cam_matrix_left @ RT1

    RT2 = np.hstack([R, T.reshape(3, 1)])
    P2 = cam_matrix_right @ RT2
    return P1, P2


def dlt(P1, P2, point1, point2):
    """Triangulate one point pair with a manual Direct Linear Transform."""
    A = np.array([
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :],
    ])
    B = A.T @ A
    _, _, Vh = np.linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]


def triangulate_points_dlt(P1, P2, points1, points2):
    """points1/points2: (N, 2) arrays of matched pixel coords. Returns (N, 3)."""
    return np.array([dlt(P1, P2, p1, p2) for p1, p2 in zip(points1, points2)])


def triangulate_cv(cam_matrix_left, dist_left, cam_matrix_right, dist_right,
                    R, T, points_left, points_right):
    """Recommended path: undistort to normalized coords, then cv.triangulatePoints.

    points_left/points_right: (N, 2) arrays of raw (distorted) pixel coords
    from the ORIGINAL (non-rectified) images.
    """
    points_left = np.asarray(points_left, dtype=np.float64).reshape(-1, 1, 2)
    points_right = np.asarray(points_right, dtype=np.float64).reshape(-1, 1, 2)

    norm_left = cv.undistortPoints(points_left, cam_matrix_left, dist_left)
    norm_right = cv.undistortPoints(points_right, cam_matrix_right, dist_right)

    RT1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    RT2 = np.hstack([R, T.reshape(3, 1)])

    points_4d = cv.triangulatePoints(RT1, RT2, norm_left, norm_right)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d
