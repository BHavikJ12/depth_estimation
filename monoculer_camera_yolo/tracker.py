"""Multi-target tracking, and the lock that picks the chase target.

A SORT-style constant-velocity Kalman filter, with two changes that matter for
a 15-pixel drone rather than a 200-pixel car:

* The filter's third state is the *span* the range estimate consumes, not SORT's
  area/aspect pair. Smoothing the span directly is what takes range from
  frame-to-frame noise to something a guidance loop can differentiate.
* Association falls back from IoU to gated centre distance. Two boxes 12 px
  across, moving 15 px between frames, overlap by nothing at all; pure IoU
  association breaks the track every time the target moves quickly.
"""

import numpy as np


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    """One tracked drone. State is [cx, cy, span, vx, vy, vspan]."""

    _next_id = 1

    def __init__(self, detection, process_noise=1.0, measurement_noise=4.0):
        x1, y1, x2, y2 = detection.box
        w, h = x2 - x1, y2 - y1
        span = max(w, h)

        self.id = Track._next_id
        Track._next_id += 1

        self.x = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, span, 0.0, 0.0, 0.0])
        self.P = np.diag([4.0, 4.0, 4.0, 100.0, 100.0, 25.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 2.0]) * process_noise
        self.R = np.diag([1.0, 1.0, 2.0]) * measurement_noise

        self.aspect = (w / h) if h > 0 else 1.0
        self.confidence = detection.confidence
        self.label = detection.label
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.range_estimate = None

    # ---- filter -------------------------------------------------------

    @staticmethod
    def _F(dt):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    _H = np.array([[1.0, 0, 0, 0, 0, 0],
                   [0, 1.0, 0, 0, 0, 0],
                   [0, 0, 1.0, 0, 0, 0]])

    def predict(self, dt=1.0):
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt
        self.x[2] = max(self.x[2], 1.0)      # a span can never go negative
        self.age += 1
        self.time_since_update += 1
        return self.box

    def update(self, detection):
        x1, y1, x2, y2 = detection.box
        w, h = x2 - x1, y2 - y1
        z = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(w, h)])

        H = self._H
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        # Floating-point drift can slowly make P numerically non-symmetric
        # over a long unattended run; re-symmetrize after every update so it
        # stays a valid covariance matrix indefinitely.
        self.P = (self.P + self.P.T) / 2.0

        if h > 0:
            self.aspect = 0.7 * self.aspect + 0.3 * (w / h)
        self.confidence = 0.6 * self.confidence + 0.4 * detection.confidence
        self.label = detection.label
        self.hits += 1
        self.time_since_update = 0

    # ---- geometry -----------------------------------------------------

    @property
    def box(self):
        """Smoothed box, rebuilt from span and the running aspect ratio."""
        cx, cy, s = self.x[0], self.x[1], max(self.x[2], 1.0)
        if self.aspect >= 1.0:          # wider than tall: span is the width
            w, h = s, s / max(self.aspect, 1e-6)
        else:
            w, h = s * self.aspect, s
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    @property
    def center(self):
        return (self.x[0], self.x[1])

    @property
    def pixel_velocity(self):
        return (self.x[3], self.x[4])

    @property
    def closing(self):
        """Span growing means the target is getting closer. Sign only."""
        return self.x[5]


class DroneTracker:
    """Greedy-association multi-target tracker plus a sticky chase lock."""

    def __init__(self, iou_threshold=0.2, max_age=15, min_hits=3,
                 distance_gate=2.5, process_noise=1.0, measurement_noise=4.0):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_gate = distance_gate   # in multiples of the track's span
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.tracks = []
        self.locked_id = None

    def _match(self, detections):
        """Greedy IoU, then gated centre distance for the leftovers."""
        pairs, used_d, used_t = [], set(), set()
        if detections and self.tracks:
            scores = []
            for ti, t in enumerate(self.tracks):
                tb = t.box
                for di, d in enumerate(detections):
                    v = iou(tb, d.box)
                    if v >= self.iou_threshold:
                        scores.append((v, ti, di))
            for _, ti, di in sorted(scores, reverse=True):
                if ti in used_t or di in used_d:
                    continue
                pairs.append((ti, di))
                used_t.add(ti)
                used_d.add(di)

            # Fast small targets can leave zero overlap between frames, so try
            # again on distance for whatever is still unmatched.
            dists = []
            for ti, t in enumerate(self.tracks):
                if ti in used_t:
                    continue
                tcx, tcy = t.center
                gate = self.distance_gate * max(t.x[2], 8.0)
                for di, d in enumerate(detections):
                    if di in used_d:
                        continue
                    dcx, dcy = d.center
                    dist = np.hypot(tcx - dcx, tcy - dcy)
                    if dist <= gate:
                        dists.append((dist, ti, di))
            for _, ti, di in sorted(dists):
                if ti in used_t or di in used_d:
                    continue
                pairs.append((ti, di))
                used_t.add(ti)
                used_d.add(di)

        unmatched_d = [i for i in range(len(detections)) if i not in used_d]
        unmatched_t = [i for i in range(len(self.tracks)) if i not in used_t]
        return pairs, unmatched_d, unmatched_t

    def update(self, detections, dt=1.0):
        for t in self.tracks:
            t.predict(dt)

        pairs, unmatched_d, _ = self._match(detections)
        for ti, di in pairs:
            self.tracks[ti].update(detections[di])
        for di in unmatched_d:
            self.tracks.append(Track(detections[di], self.process_noise,
                                     self.measurement_noise))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.confirmed

    @property
    def confirmed(self):
        """Tracks stable enough to act on. A track that has been seen enough
        times stays confirmed through a short dropout, which is what keeps the
        overlay from flickering when the target crosses a tree line."""
        return [t for t in self.tracks
                if t.hits >= self.min_hits and t.time_since_update <= self.max_age // 2]

    def select_chase_target(self, frame_shape):
        """Pick the one track to chase, with hysteresis so the lock does not
        ping-pong between two similar targets."""
        candidates = self.confirmed
        if not candidates:
            self.locked_id = None
            return None

        h, w = frame_shape[:2]
        cx0, cy0 = w / 2.0, h / 2.0
        diag = float(np.hypot(w, h))

        best, best_score = None, -1.0
        for t in candidates:
            cx, cy = t.center
            centrality = 1.0 - min(1.0, np.hypot(cx - cx0, cy - cy0) / (diag / 2.0))
            maturity = min(1.0, t.hits / 10.0)
            fresh = 1.0 / (1.0 + t.time_since_update)
            score = (0.45 * t.confidence + 0.25 * centrality
                     + 0.20 * maturity + 0.10 * fresh)
            if t.id == self.locked_id:
                score *= 1.25          # incumbent advantage
            if score > best_score:
                best, best_score = t, score

        self.locked_id = best.id if best else None
        return best
