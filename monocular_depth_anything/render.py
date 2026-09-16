"""Colourising a depth map, and the HUD that keeps it honest.

Two decisions here are worth stating, because both affect what the operator
believes:

**Robust normalisation.** A depth map's extremes are its least reliable pixels
-- one speck of sky or one blown-out reflection sets the far end and flattens
everything else to the same colour. The colour ramp is therefore stretched over
the 2nd..98th percentile, not min..max.

**Sticky range.** Renormalising per frame makes the colours breathe: a wall at
a fixed distance changes hue as somebody walks past. The range is carried
across frames with an EMA so the palette stays put, and in metric modes it can
be pinned to an absolute span so a colour means a distance.
"""

import cv2 as cv
import numpy as np

COLORMAPS = [
    ("turbo", cv.COLORMAP_TURBO),
    ("inferno", cv.COLORMAP_INFERNO),
    ("magma", cv.COLORMAP_MAGMA),
    ("viridis", cv.COLORMAP_VIRIDIS),
    ("gray", None),
]

VIEWS = ["depth", "split", "overlay", "confidence", "camera"]


class DepthPainter:
    """Depth -> BGR, with a range that does not jump between frames."""

    def __init__(self, cmap=0, smooth=0.1, fixed_range=None, invert=False):
        self.cmap = cmap
        self.smooth = smooth
        self.fixed_range = fixed_range   # (near, far) in metres, or None
        self.invert = invert             # near = warm by default
        self.lo = self.hi = None

    def cycle_cmap(self, step=1):
        self.cmap = (self.cmap + step) % len(COLORMAPS)
        return COLORMAPS[self.cmap][0]

    def range_of(self, depth):
        if self.fixed_range:
            return self.fixed_range
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo, hi = np.percentile(finite, (2.0, 98.0))
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        if self.lo is None:
            self.lo, self.hi = float(lo), float(hi)
        else:
            a = self.smooth
            self.lo = (1 - a) * self.lo + a * float(lo)
            self.hi = (1 - a) * self.hi + a * float(hi)
        return self.lo, self.hi

    def paint(self, depth, size=None, near_is_high=False):
        lo, hi = self.range_of(depth)
        norm = np.clip((depth - lo) / max(hi - lo, 1e-9), 0, 1)
        # Near should read as "hot". Depth grows away from the camera and
        # disparity shrinks, so only one of the two needs flipping; `invert`
        # is the operator's own toggle on top of that.
        if near_is_high == self.invert:
            norm = 1.0 - norm
        u8 = (norm * 255).astype(np.uint8)
        name, code = COLORMAPS[self.cmap]
        img = cv.cvtColor(u8, cv.COLOR_GRAY2BGR) if code is None else cv.applyColorMap(u8, code)
        if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
            img = cv.resize(img, tuple(size), interpolation=cv.INTER_LINEAR)
        return img


def paint_confidence(conf, size=None):
    """V3's confidence, stretched over its own range (it is not a probability)."""
    if conf is None:
        return None
    lo, hi = np.percentile(conf, (2, 98))
    u8 = (np.clip((conf - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    img = cv.applyColorMap(u8, cv.COLORMAP_BONE)
    if size is not None:
        img = cv.resize(img, tuple(size), interpolation=cv.INTER_LINEAR)
    return img


def compose(view, frame, depth_bgr, conf_bgr=None, alpha=0.6, split_x=None):
    """Lay the frame and the depth map out according to the current view."""
    h, w = frame.shape[:2]
    if view == "camera":
        return frame.copy()
    if view == "depth":
        return depth_bgr
    if view == "confidence":
        return conf_bgr if conf_bgr is not None else depth_bgr
    if view == "overlay":
        return cv.addWeighted(depth_bgr, alpha, frame, 1 - alpha, 0)
    if view == "split":
        # A wipe rather than two panes side by side: the same pixels line up
        # across the seam, which is how you see whether an edge in the depth
        # map is really on the object's edge.
        x = w // 2 if split_x is None else int(np.clip(split_x, 1, w - 1))
        out = frame.copy()
        out[:, x:] = depth_bgr[:, x:]
        cv.line(out, (x, 0), (x, h), (255, 255, 255), 1)
        return out
    return depth_bgr


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

_FONT = cv.FONT_HERSHEY_SIMPLEX


def _width(s, scale, weight=1):
    return cv.getTextSize(s, _FONT, scale, weight)[0][0]


def _dim(img, y0, y1, strength):
    """Darken a band in place, so HUD text sits on the picture without hiding it."""
    band = img[y0:y1]
    band[:] = (band.astype(np.float32) * (1.0 - strength)).astype(np.uint8)


def put_text(img, s, org, scale=0.5, color=(235, 235, 235), weight=1):
    """A light glyph over a dark shadow, legible on a white wall or a dark doorway.

    The shadow is drawn at the *same* thickness, one pixel down and right.
    OpenCV's Hershey advance widens with line thickness -- a thickness-3 stroke
    of the same string is ~6% longer than the thickness-1 glyph -- so the
    obvious "thick black underneath" trick leaves the stroke sticking out past
    the end of the text, which reads as doubled characters.
    """
    x, y = org
    cv.putText(img, s, (x + 1, y + 1), _FONT, scale, (0, 0, 0), weight, cv.LINE_AA)
    cv.putText(img, s, org, _FONT, scale, color, weight, cv.LINE_AA)


def draw_hud(img, *, model, device, fps, infer_ms, view, cmap, depth_range,
             scale_status, metric, probe=None, paused=False, recording=False,
             cam=None, note=None, range_units=None):
    h, w = img.shape[:2]
    _dim(img, 0, 54, 0.62)

    put_text(img, f"{model} | {device}", (10, 20), 0.55, (120, 235, 255))
    put_text(img, f"{fps:5.1f} fps   infer {infer_ms:5.1f} ms", (10, 42), 0.5)
    left_edge = 10 + max(_width(f"{model} | {device}", 0.55),
                         _width(f"{fps:5.1f} fps   infer {infer_ms:5.1f} ms", 0.5))

    unit = range_units or ("m" if metric else "rel")
    lo, hi = depth_range
    right = [(f"view {view}  cmap {cmap}  range {lo:.2f}-{hi:.2f} {unit}",
              (235, 235, 235)),
             (f"scale: {scale_status}",
              (150, 255, 150) if metric else (170, 170, 170))]
    # Right-aligned, and shrunk rather than overlapped when the frame is narrow
    # -- a 640-wide preview has no room for both blocks at full size.
    fs = 0.5
    while fs > 0.3 and left_edge + 16 + max(_width(t, fs) for t, _ in right) > w - 10:
        fs -= 0.02
    for i, (t, colour) in enumerate(right):
        x = max(left_edge + 16, w - 10 - _width(t, fs))
        put_text(img, t, (x, 20 + 22 * i), fs, colour)

    if cam is not None:
        put_text(img, f"fx {cam.fx:.0f}px  hfov {cam.hfov_deg:.0f}deg  [{cam.source}]",
              (10, h - 12), 0.45, (200, 200, 160))

    if probe is not None:
        _draw_probe(img, probe, metric)

    y = 74
    for flag, text, colour in ((paused, "PAUSED", (0, 210, 255)),
                              (recording, "REC", (0, 0, 255))):
        if flag:
            put_text(img, text, (10, y), 0.7, colour, 2)
            y += 26
    if note:
        put_text(img, note, (10, h - 34), 0.55, (0, 220, 255))
    return img


def _draw_probe(img, probe, metric):
    """Crosshair and readout at the pixel the mouse is over."""
    u, v, value, extra = probe
    h, w = img.shape[:2]
    cv.drawMarker(img, (int(u), int(v)), (255, 255, 255), cv.MARKER_CROSS, 18, 1, cv.LINE_AA)
    cv.circle(img, (int(u), int(v)), 10, (0, 0, 0), 1, cv.LINE_AA)
    label = f"{value:.2f} m" if metric else f"{value:.3f} rel"
    if extra:
        label += f"  {extra}"
    tw = cv.getTextSize(label, _FONT, 0.55, 2)[0][0]
    tx = int(np.clip(u + 14, 4, w - tw - 8))
    ty = int(np.clip(v - 12, 22, h - 8))
    cv.rectangle(img, (tx - 5, ty - 17), (tx + tw + 5, ty + 6), (0, 0, 0), -1)
    put_text(img, label, (tx, ty), 0.55, (120, 255, 200), 1)


def draw_pins(img, pins, metric):
    """Persistent measurement points dropped with the mouse."""
    for i, (u, v, value) in enumerate(pins):
        cv.circle(img, (int(u), int(v)), 5, (0, 240, 255), -1, cv.LINE_AA)
        cv.circle(img, (int(u), int(v)), 6, (0, 0, 0), 1, cv.LINE_AA)
        label = f"{i}: {value:.2f}m" if metric else f"{i}: {value:.3f}"
        put_text(img, label, (int(u) + 9, int(v) - 7), 0.45, (0, 240, 255))
    return img


HELP = [
    "q/ESC quit      SPACE pause      s save png+ply     r record",
    "c colormap      v view           i invert ramp      h help",
    "[ / ] net res   f freeze range   x clear pins",
    "LMB pin a point   RMB clear last   MMB set scale anchor",
]


def draw_help(img, bottom_margin=46):
    """The key list, parked above the permanent bottom status lines."""
    h, w = img.shape[:2]
    pad, lh = 12, 24
    box_h = lh * len(HELP) + pad * 2
    y1 = max(box_h, h - bottom_margin)
    y0 = y1 - box_h
    _dim(img, y0, y1, 0.75)
    for i, line in enumerate(HELP):
        put_text(img, line, (pad, y0 + pad + lh * (i + 1) - 6), 0.5)
    return img


# `_text` is the name the rest of this package grew up with.
_text = put_text
