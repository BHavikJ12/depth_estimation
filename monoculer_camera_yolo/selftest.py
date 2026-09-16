#!/usr/bin/env python3
"""Offline checks for the geometry, the tracker and the HUD.

Runs without a Roboflow key and without a camera: the detector is the only part
that needs the network, so everything else is verified against synthetic data
with a known ground truth. Also writes selftest_hud.png so the overlay can be
eyeballed before pointing it at a real sky.
"""

import math
import sys

import cv2 as cv
import numpy as np

from camera import CameraModel
import hailo_model
import onnx_model
from detector import Detection
from overlay import annotate, status_lines
from ranging import RangeEstimator, apparent_span, solve_target_size
from tracker import DroneTracker

TARGET_M = 0.5
failures = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def box_for(camera, range_m, size_m=TARGET_M, center=None):
    """The box a `size_m` target would project to at `range_m`. Ground truth."""
    span = camera.fx * size_m / range_m
    cx, cy = center or (camera.cx, camera.cy)
    return (cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2)


print("camera model")
cam = CameraModel.from_fov(1280, 720, 70.0)
check("hfov round-trips", abs(cam.hfov_deg - 70.0) < 1e-6, f"fx={cam.fx:.2f}px")
check("fx matches the closed form",
      abs(cam.fx - (1280 / 2) / math.tan(math.radians(35))) < 1e-9)
half = cam.for_frame(640, 360)
check("intrinsics halve with the frame", abs(half.fx - cam.fx / 2) < 1e-9,
      f"{cam.fx:.1f} -> {half.fx:.1f}px")
check("FOV is unchanged by resizing", abs(half.hfov_deg - cam.hfov_deg) < 1e-6)

print("\nranging (fx=960 px, 0.5 m target)")
cal = CameraModel.from_fx(1280, 720, 960.0)
est = RangeEstimator(cal, TARGET_M, "max")
for truth in (5.0, 9.1, 25.0, 60.0):
    r = est.estimate(box_for(cal, truth))
    check(f"{truth:5.1f} m recovered", r is not None and abs(r.range_m - truth) < 1e-6,
          f"got {r.range_m:.3f} m, span {r.span_px:.1f}px, +/-{r.sigma_m:.2f} m"
          if r else "no estimate")

r10 = est.estimate(box_for(cal, 10.0))
r40 = est.estimate(box_for(cal, 40.0))
check("uncertainty grows with range", r40.sigma_m / r40.range_m > r10.sigma_m / r10.range_m,
      f"{100 * r10.sigma_m / r10.range_m:.1f}% at 10 m vs "
      f"{100 * r40.sigma_m / r40.range_m:.1f}% at 40 m")
check("sub-pixel targets are refused", est.estimate((100, 100, 102, 102)) is None)
check("max useful range is finite and sane",
      10.0 < est.max_useful_range(0.25) < 1000.0,
      f"{est.max_useful_range(0.25):.1f} m at 25% error")

off = est.estimate(box_for(cal, 20.0, center=(cal.cx + 300, cal.cy - 100)))
check("off-axis azimuth is signed correctly", off.azimuth_deg > 0 and off.elevation_deg > 0,
      f"az {off.azimuth_deg:+.2f} el {off.elevation_deg:+.2f}")
check("slant range exceeds forward range", off.range_m > off.z_m,
      f"range {off.range_m:.2f} m vs z {off.z_m:.2f} m")

print("\nsize calibration")
solved = solve_target_size(cal, box_for(cal, 12.0), 12.0)
check("solve_target_size recovers the prior", abs(solved - TARGET_M) < 1e-9,
      f"{solved * 1000:.1f} mm")

print("\nspan modes")
b = (0.0, 0.0, 40.0, 20.0)
check("max picks the larger side", apparent_span(b, "max") == 40.0)
check("height reads the shorter side", apparent_span(b, "height") == 20.0)
check("diag is the hypotenuse", abs(apparent_span(b, "diag") - math.hypot(40, 20)) < 1e-9)

print("\ntracker (target crossing the frame, 3 px/frame of box noise)")
rng = np.random.default_rng(0)
trk = DroneTracker(min_hits=3, max_age=15)
ids, ranges = set(), []
locked = None
for i in range(40):
    truth_range = 30.0 - 0.5 * i                    # closing at 0.5 m/frame
    span = cal.fx * TARGET_M / truth_range
    cx, cy = 300 + 14 * i, 360 + 4 * i              # fast, small, moving
    jitter = rng.normal(0, 3.0, 2)
    box = (cx - span / 2 + jitter[0], cy - span / 2 + jitter[1],
           cx + span / 2 + jitter[0], cy + span / 2 + jitter[1])
    tracks = trk.update([Detection(*box, 0.86)])
    for t in tracks:
        t.range_estimate = est.estimate(t.box)
    locked = trk.select_chase_target((720, 1280))
    if locked is not None:
        ids.add(locked.id)
        ranges.append((truth_range, locked.range_estimate.range_m))

check("the target is tracked", len(ranges) > 25, f"{len(ranges)} locked frames")
check("the lock never changes identity", len(ids) == 1, f"ids {sorted(ids)}")
err = [abs(a - b) / a for a, b in ranges[5:]]
check("smoothed range stays within 6% of truth", max(err) < 0.06,
      f"max {100 * max(err):.2f}%, mean {100 * float(np.mean(err)):.2f}%")
check("closing is detected", locked.closing > 0, f"d(span)/dt = {locked.closing:+.2f} px/frame")

raw = RangeEstimator(cal, TARGET_M).estimate(
    (300, 360, 300 + cal.fx * TARGET_M / 10.0, 360 + cal.fx * TARGET_M / 10.0))
check("a single box still ranges without a track", raw is not None and abs(raw.range_m - 10) < 1.0)

print("\nonnx decode (fake session, known box)")


class _FakeSession:
    """Emits a YOLO head containing one box, in the requested layout."""

    def __init__(self, layout, box_xywh, score=0.91, n_anchors=40, n_cls=1):
        self.layout, self.box, self.score = layout, box_xywh, score
        self.n_anchors, self.n_cls = n_anchors, n_cls

    def get_inputs(self):
        return [type("I", (), {"name": "images", "shape": [1, 3, 640, 640]})()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _outputs, _feed):
        if self.layout == "v8":
            a = np.zeros((self.n_anchors, 4 + self.n_cls), np.float32)
            a[0, :4] = self.box
            a[0, 4] = self.score
            return [a.T[None]]                       # (1, 4+C, anchors)
        a = np.zeros((self.n_anchors, 5 + self.n_cls), np.float32)
        a[0, :4] = self.box
        a[0, 4] = self.score                          # objectness
        a[0, 5] = 1.0                                 # class score
        return [a[None]]                              # (1, anchors, 5+C)


def _model(layout, box, letterbox):
    m = onnx_model.OnnxDetectionModel.__new__(onnx_model.OnnxDetectionModel)
    m.session = _FakeSession(layout, box)
    m.provider = "CPUExecutionProvider"
    m.input_name, m.in_w, m.in_h = "images", 640, 640
    m.env, m.classes, m.letterbox = {}, ["drone"], letterbox
    m.head = "v8" if layout == "v8" else "v5"
    return m


# A 1280x720 frame; put a 64x32 box at (400, 300) and check it comes back.
frame = np.zeros((720, 1280, 3), np.uint8)
want = (400.0, 300.0, 464.0, 332.0)

for layout in ("v8", "v5"):
    # stretch: x scales by 640/1280, y by 640/720
    sx, sy = 640 / 1280, 640 / 720
    net_box = ((want[0] + want[2]) / 2 * sx, (want[1] + want[3]) / 2 * sy,
               (want[2] - want[0]) * sx, (want[3] - want[1]) * sy)
    got = _model(layout, net_box, letterbox=False)(frame, 0.4, 0.45)
    ok = len(got) == 1 and max(abs(g - w) for g, w in zip(got[0][:4], want)) < 0.6
    check(f"{layout} head, stretch resize", ok,
          f"{tuple(round(v, 1) for v in got[0][:4])} conf {got[0][4]:.2f}" if got else "nothing")

    r = min(640 / 1280, 640 / 720)
    net_box = ((want[0] + want[2]) / 2 * r, (want[1] + want[3]) / 2 * r,
               (want[2] - want[0]) * r, (want[3] - want[1]) * r)
    got = _model(layout, net_box, letterbox=True)(frame, 0.4, 0.45)
    ok = len(got) == 1 and max(abs(g - w) for g, w in zip(got[0][:4], want)) < 0.6
    check(f"{layout} head, letterbox resize", ok,
          f"{tuple(round(v, 1) for v in got[0][:4])}" if got else "nothing")

check("a v5 head is not mistaken for v8",
      _model("v5", (320, 320, 64, 32), False)(frame, 0.4, 0.45)[0][4] > 0.8)
check("low-confidence output is dropped",
      _model("v8", (320, 320, 64, 32), False)(frame, 0.95, 0.45) == [])

print("\nhailo NMS decode (no hardware needed)")

# HailoRT returns, per class, an (n, 5) array of normalised
# (ymin, xmin, ymax, xmax, score). Check that unpacks to frame pixels.
m = hailo_model.HailoDetectionModel.__new__(hailo_model.HailoDetectionModel)
m.classes = ["0", "drone", "bird"]
frame_shape = (720, 1280, 3)
per_class = [
    np.zeros((0, 5), np.float32),                                   # class 0: none
    np.array([[300 / 720, 400 / 1280, 332 / 720, 464 / 1280, 0.91]], np.float32),
    np.array([[10 / 720, 10 / 1280, 20 / 720, 20 / 1280, 0.10]], np.float32),
]
got = m._decode_nms(per_class, frame_shape, 0.40)
check("one box survives the threshold", len(got) == 1, f"{len(got)} returned")
if got:
    x1, y1, x2, y2, score, cid = got[0]
    check("box maps back to frame pixels",
          max(abs(x1 - 400), abs(y1 - 300), abs(x2 - 464), abs(y2 - 332)) < 0.5,
          f"({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    check("class index follows the list position", cid == 1)
    check("score is carried through", abs(score - 0.91) < 1e-6)
    check("class name resolves", m.class_name(cid) == "drone", m.class_name(cid))
check("the low-confidence class is dropped",
      all(c != 2 for *_, c in got))
check("empty classes do not crash", m._decode_nms([np.zeros((0, 5), np.float32)],
                                                  frame_shape, 0.4) == [])
check("an NMS list is told apart from a dense tensor",
      hailo_model.HailoDetectionModel.is_nms_output([np.zeros((0, 5), np.float32)])
      and not hailo_model.HailoDetectionModel.is_nms_output(np.zeros((1, 8, 10), np.float32)))

print("\noverlay")
frame = np.zeros((720, 1280, 3), np.uint8)
frame[:400] = (200, 150, 90)          # sky
frame[400:] = (60, 110, 70)           # ground
tracks = trk.confirmed
annotate(frame, tracks, locked, cal,
         status_lines(type("D", (), {"backend": "hosted", "model_id": "drone-detection-rchy7/1"}),
                      cal, est, 24.0, len(tracks), locked))
cv.imwrite("selftest_hud.png", frame)
check("HUD renders", cv.imread("selftest_hud.png") is not None, "wrote selftest_hud.png")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks passed")
