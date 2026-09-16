"""Thin wrapper around the AR0144 stereo USB camera's combined video frame."""

import cv2 as cv

from . import config


class StereoCamera:
    """Opens the single UVC device and splits each frame into left/right eyes."""

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        fourcc: str = config.FOURCC,
        fps: int = config.FRAME_FPS,
    ):
        self.cap = cv.VideoCapture(index, cv.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {index}. "
                "Run `v4l2-ctl --list-devices` and update CAMERA_INDEX in stereo/config.py."
            )

        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv.CAP_PROP_FPS, fps)

        actual_w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (width, height):
            raise RuntimeError(
                f"Camera reports {actual_w}x{actual_h} instead of the requested "
                f"{width}x{height}. Check that this is the AR0144 stereo camera "
                "and that MJPG mode is supported at this resolution."
            )

    def read(self):
        """Returns (ok, left_bgr, right_bgr). Splits the combined frame in half."""
        ok, frame = self.cap.read()
        if not ok:
            return False, None, None
        half = frame.shape[1] // 2
        left = frame[:, :half]
        right = frame[:, half:]
        return True, left, right

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
