"""HUD drawing: boxes, range readouts, and the chase-target callout."""

import cv2 as cv

RED = (0, 0, 255)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
AMBER = (0, 190, 255)
GREY = (170, 170, 170)
BLACK = (0, 0, 0)

FONT = cv.FONT_HERSHEY_DUPLEX


_OUTLINE_OFFSETS = ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0))


def _text(img, s, org, color, scale=0.5, thickness=1, shadow=True):
    """Text with a dark outline, so it survives a bright sky behind it.

    The outline is the same string at the same stroke weight, nudged one pixel
    in eight directions, rather than a single heavier pass. OpenCV 5 changes a
    Hershey glyph's advance width with its thickness, so a `thickness + 1`
    backing pass drifts out from under the text and reads as a double image.
    """
    if shadow:
        for dx, dy in _OUTLINE_OFFSETS:
            cv.putText(img, s, (org[0] + dx, org[1] + dy), FONT, scale, BLACK,
                       thickness, cv.LINE_AA)
    cv.putText(img, s, org, FONT, scale, color, thickness, cv.LINE_AA)


def _crosshair(img, center, color, size=9, gap=3):
    x, y = int(round(center[0])), int(round(center[1]))
    cv.line(img, (x - size, y), (x - gap, y), color, 1, cv.LINE_AA)
    cv.line(img, (x + gap, y), (x + size, y), color, 1, cv.LINE_AA)
    cv.line(img, (x, y - size), (x, y - gap), color, 1, cv.LINE_AA)
    cv.line(img, (x, y + gap), (x, y + size), color, 1, cv.LINE_AA)


def draw_track(img, track, locked=False, show_id=True):
    """One tracked drone: box, range, and the chase callout when locked."""
    x1, y1, x2, y2 = (int(round(v)) for v in track.box)
    color = RED if locked else AMBER
    cv.rectangle(img, (x1, y1), (x2, y2), color, 2 if locked else 1, cv.LINE_AA)

    est = track.range_estimate
    scale = 0.75 if locked else 0.45
    thick = 2 if locked else 1

    if est is not None:
        head = f"Dist: {est.range_m:.1f}m  Conf: {track.confidence:.2f}"
    else:
        head = f"Conf: {track.confidence:.2f}"

    (tw, th), _ = cv.getTextSize(head, FONT, scale, thick)
    hx = max(2, min(x1 + (x2 - x1) // 2 - tw // 2, img.shape[1] - tw - 2))
    hy = y1 - 10 if y1 - 10 > th else y2 + th + 10
    _text(img, head, (hx, hy), color, scale, thick)

    if locked:
        _crosshair(img, track.center, CYAN, size=12)
        label = "CHASE TARGET"
        (lw, _), _ = cv.getTextSize(label, FONT, 0.75, 2)
        lx = max(2, min(x1 + (x2 - x1) // 2 - lw // 2, img.shape[1] - lw - 2))
        _text(img, label, (lx, min(y2 + 34, img.shape[0] - 6)), RED, 0.75, 2)

        if est is not None:
            detail = (f"+/-{est.sigma_m:.1f}m   az {est.azimuth_deg:+.1f} "
                      f"el {est.elevation_deg:+.1f}   span {est.span_px:.0f}px")
            _text(img, detail, (lx, min(y2 + 54, img.shape[0] - 4)), CYAN, 0.45, 1)
    elif show_id:
        _text(img, f"#{track.id}", (x1, max(th + 2, y1 - 2)), color, 0.4, 1)


def draw_hud(img, lines, origin=(10, 24), spacing=20, color=CYAN, scale=0.5):
    x, y = origin
    for i, line in enumerate(lines):
        _text(img, line, (x, y + i * spacing), color, scale, 1)


def draw_boresight(img, camera, color=GREY):
    """Where the lens is actually pointing, which is not the frame centre once
    the principal point is calibrated."""
    _crosshair(img, (camera.cx, camera.cy), color, size=16, gap=6)


def annotate(img, tracks, locked, camera, hud_lines=(), draw_center=True):
    """Draw a whole frame's worth of overlay. Modifies `img` in place."""
    if draw_center:
        draw_boresight(img, camera)
    for t in tracks:
        draw_track(img, t, locked=(locked is not None and t.id == locked.id))
    if hud_lines:
        draw_hud(img, list(hud_lines))
    return img


def status_lines(detector, camera, estimator, fps, n_tracks, locked):
    """The telemetry block in the top-left corner."""
    lines = [
        f"{detector.backend}  {detector.model_id}   {fps:5.1f} fps",
        f"fx {camera.fx:.0f}px  hfov {camera.hfov_deg:.1f}deg  ({camera.source})",
        f"target span {estimator.target_size_m * 1000:.0f}mm  "
        f"usable to ~{estimator.max_useful_range(0.25):.0f}m @25% err",
        f"tracks {n_tracks}",
    ]
    if locked is not None and locked.range_estimate is not None:
        e = locked.range_estimate
        trend = "CLOSING" if locked.closing > 0.15 else (
            "OPENING" if locked.closing < -0.15 else "STEADY")
        lines.append(f"lock #{locked.id}  {e.range_m:.1f}m [{e.range_lo:.1f}-{e.range_hi:.1f}]  {trend}")
    return lines
