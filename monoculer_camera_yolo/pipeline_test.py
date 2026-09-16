#!/usr/bin/env python3
"""End-to-end test of run.py against a synthetic clip with known ground truth.

Renders a 0.5 m disc closing from 40 m to 7.5 m at a known focal length, then
drives the real CLI over it with the Roboflow call replaced by a stub that
finds the disc. That exercises everything except the network: frame decode,
tracking, ranging, the HUD, the video writer and the CSV.

The strict check is not the absolute error but whether the error stays inside
the `sigma_m` the tool reports. A range that is wrong and says so is usable; a
range that is wrong and claims precision is not.

    ./venv/bin/python pipeline_test.py
"""

import csv
import os
import sys
import tempfile

import cv2 as cv
import numpy as np

import detector as det_mod
import run as run_mod
from camera import CameraModel

W, H, N = 1280, 720, 60
HFOV = 70.0
TARGET_M = 0.5


def make_clip(path, camera):
    """A dark disc of known angular size against sky, closing at 0.55 m/frame."""
    truth = []
    vw = cv.VideoWriter(path, cv.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
    rng = np.random.default_rng(7)
    for i in range(N):
        z = 40.0 - 0.55 * i
        span = camera.fx * TARGET_M / z
        cx, cy = 400 + 7 * i, 250 + 1.5 * i
        truth.append((z, cx, cy, span))
        f = np.zeros((H, W, 3), np.uint8)
        f[:450] = (210, 160, 100)
        f[450:] = (70, 120, 80)
        f += rng.integers(0, 12, f.shape, dtype=np.uint8)
        cv.circle(f, (int(cx), int(cy)), max(2, int(span / 2)), (40, 40, 40), -1)
        vw.write(f)
    vw.release()
    return truth


class StubDetector:
    """Stands in for Roboflow. Finds the disc in whatever frame it is handed,
    plus 2 px of box jitter, so it stays correct at any --stride."""

    def __init__(self, *args, **kwargs):
        self.backend, self.model_id = "stub", "synthetic/1"
        self.weights, self.max_side, self.confidence = None, None, 0.4
        self.rng = np.random.default_rng(3)

    def describe(self):
        return "backend=stub  model=synthetic/1"

    def detect(self, frame):
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(cv.inRange(grey, 0, 70),
                                      cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        x, y, w, h = cv.boundingRect(max(contours, key=cv.contourArea))
        j = self.rng.normal(0, 2.0, 2)
        return [det_mod.Detection(x + j[0], y + j[1], x + w + j[0], y + h + j[1],
                                  0.87, "drone")]


def run_case(stride, tmp, save_frame=None):
    camera = CameraModel.from_fov(W, H, HFOV)
    clip = os.path.join(tmp, f"synthetic_{stride}.mp4")
    out_v = os.path.join(tmp, f"out_{stride}.mp4")
    out_c = os.path.join(tmp, f"out_{stride}.csv")
    truth = make_clip(clip, camera)

    real = run_mod.DroneDetector
    run_mod.DroneDetector = StubDetector
    try:
        rc = run_mod.main(["--source", clip, "--hfov", str(HFOV),
                           "--target-size", str(TARGET_M), "--stride", str(stride),
                           "--no-display", "--quiet", "--save", out_v,
                           "--save-csv", out_c])
    finally:
        run_mod.DroneDetector = real

    rows = list(csv.DictReader(open(out_c)))
    locked = [r for r in rows if r["locked"] == "1" and r["range_m"]]
    truth_of = lambda r: truth[int(r["frame"])][0]
    errs = [abs(float(r["range_m"]) - truth_of(r)) / truth_of(r) for r in locked]
    zs = [abs(float(r["range_m"]) - truth_of(r)) / float(r["sigma_m"]) for r in locked]
    near = [e for r, e in zip(locked, errs) if truth_of(r) <= 20]
    ids = {r["track_id"] for r in locked}

    print(f"\nstride {stride}")
    print(f"  {len(locked)}/{N} frames locked, track ids {sorted(ids)}")
    print(f"  mean error {100 * np.mean(errs):.2f}%   "
          f"inside 20 m: {100 * np.mean(near):.2f}%")
    print(f"  worst error is {max(zs):.2f} sigma of the reported uncertainty")
    print(f"  wrote {os.path.getsize(out_v)} bytes of video, {len(rows)} csv rows")

    problems = []
    if rc != 0:
        problems.append(f"run.py exited {rc}")
    if len(ids) != 1:
        problems.append(f"the lock changed identity: {sorted(ids)}")
    if len(locked) < N * 0.85:
        problems.append(f"only {len(locked)}/{N} frames locked")
    if np.mean(near) >= 0.06:
        problems.append(f"inside 20 m the error is {100 * np.mean(near):.1f}%")
    if max(zs) >= 2.0:
        problems.append(f"error reached {max(zs):.2f} sigma, beyond what it reports")

    if save_frame:
        cap = cv.VideoCapture(out_v)
        cap.set(cv.CAP_PROP_POS_FRAMES, 50)
        ok, frame = cap.read()
        cap.release()
        if ok:
            cv.imwrite(save_frame, frame)
            print(f"  wrote {save_frame}")
    return problems


def main():
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        problems += run_case(1, tmp, save_frame="selftest_pipeline.png")
        problems += run_case(3, tmp)
    print()
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1
    print("pipeline ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
