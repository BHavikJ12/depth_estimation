#!/usr/bin/env python3
"""Live monocular depth from one camera, using Depth Anything V3 (or V2).

    ./venv/bin/python run.py --source 0
    ./venv/bin/python run.py --source 0 --scale ground --camera-height 1.2
    ./venv/bin/python run.py --source clip.mp4 --model v2-metric-indoor

Capture, inference and display each run on their own thread. That matters more
than it sounds: the network takes ~50 ms on this GPU, and if the display waited
for it the preview would stutter at 20 fps with visible input lag. Instead the
camera is drained continuously, the network always picks up the *newest* frame
and drops whatever it missed, and the viewer redraws at capture rate over the
most recent depth map. The depth lags the picture by one inference, which is
the honest cost and is shown in the HUD as `age`.
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

import cv2 as cv
import numpy as np

import render
from render import put_text
from geometry import CameraModel, backproject_rgb, write_ply
from metric import ScaleEstimator
from models import MODELS, load_model


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

class FrameSource:
    """A camera, a video file, a stream URL, or a still image.

    Live sources are drained by a thread that keeps only the newest frame. A
    USB camera buffers internally, so reading at less than capture rate hands
    you progressively staler frames -- several seconds behind by the time the
    network has warmed up. Draining fixes that; for files it would drop frames,
    so those are read on demand instead.
    """

    def __init__(self, source, width=None, height=None, fps=None, loop=False):
        self.spec = source
        self.is_image = isinstance(source, str) and os.path.splitext(source)[1].lower() in (
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        self.loop = loop
        self.still = None
        self.cap = None
        self._frame = None
        self._seq = 0          # bumped per captured frame, so read() can block
        self._seen = -1        # for a new one instead of re-serving the last
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        if self.is_image:
            self.still = cv.imread(source, cv.IMREAD_COLOR)
            if self.still is None:
                raise SystemExit(f"cannot read image: {source}")
            self.live = False
            self.size = (self.still.shape[1], self.still.shape[0])
            self.src_fps = 0.0
            return

        idx = int(source) if str(source).isdigit() else source
        self.live = isinstance(idx, int) or str(source).startswith(("rtsp://", "http://", "udp://"))
        self.cap = cv.VideoCapture(idx, cv.CAP_V4L2) if isinstance(idx, int) else cv.VideoCapture(idx)
        if not self.cap.isOpened() and isinstance(idx, int):
            self.cap = cv.VideoCapture(idx)   # retry without forcing V4L2
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open source: {source}")

        if isinstance(idx, int):
            self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
            if width:
                self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                self.cap.set(cv.CAP_PROP_FPS, fps)
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self.size = (int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        self.src_fps = self.cap.get(cv.CAP_PROP_FPS) or 0.0

        if self.live:
            self._thread = threading.Thread(target=self._drain, daemon=True)
            self._thread.start()

    def _drain(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = f
                self._seq += 1

    def read(self):
        if self.is_image:
            return True, self.still.copy()
        if self.live:
            # Block until the camera actually produces a frame we have not seen.
            # Returning the previous one immediately would spin the display loop
            # at thousands of useless iterations per second -- which is also how
            # a `--no-display` run burns through `--max-frames` before the first
            # inference has even finished.
            deadline = time.time() + 5.0
            while time.time() < deadline and not self._stop.is_set():
                with self._lock:
                    if self._frame is not None and self._seq != self._seen:
                        self._seen = self._seq
                        return True, self._frame
                time.sleep(0.002)
            with self._lock:
                if self._frame is not None:
                    return True, self._frame      # camera stalled; show it anyway
            return False, None
        ok, f = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return ok, f

    def release(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
# inference thread
# ---------------------------------------------------------------------------

class DepthWorker:
    """Runs the network on the newest frame handed to it, forever."""

    def __init__(self, model):
        self.model = model
        self._in = None
        self._out = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self.error = None
        self.count = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame, stamp):
        with self._lock:
            self._in = (frame, stamp)
        self._new.set()

    def latest(self):
        with self._lock:
            return self._out

    def _loop(self):
        while not self._stop.is_set():
            if not self._new.wait(timeout=0.1):
                continue
            self._new.clear()
            with self._lock:
                job = self._in
                self._in = None
            if job is None:
                continue
            frame, stamp = job
            try:
                res = self.model.infer(frame)
            except Exception as e:                      # keep the UI alive
                self.error = f"{type(e).__name__}: {e}"
                continue
            with self._lock:
                self._out = (res, frame, stamp)
                self.count += 1

    def stop(self):
        self._stop.set()
        self._new.set()
        self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# video output
# ---------------------------------------------------------------------------

class FFmpegWriter:
    """Pipe raw frames to ffmpeg and get H.264 out.

    OpenCV's own mp4 writer here falls back to `mp4v` (MPEG-4 Part 2), because
    this build has no working H.264 encoder -- it tries a v4l2 hardware device
    that does not exist. An mp4v file plays in VLC and nothing else: browsers,
    most default desktop players and every web preview show a **black frame**.
    ffmpeg with libx264 is the difference between a clip you can send someone
    and one only you can watch.
    """

    def __init__(self, path, fps, size, crf=20, preset="veryfast"):
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{max(fps, 1.0):.3f}", "-i", "-",
             "-an",
             # yuv420p is what makes it play everywhere, and it needs even
             # dimensions -- a 1208x679 frame would otherwise fail outright.
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", str(crf), "-preset", preset, path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)
        self.path = path

    def write(self, frame):
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, ValueError):
            pass

    def release(self):
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.wait(timeout=30)
        if self.proc.returncode:
            err = self.proc.stderr.read().decode(errors="replace")[:400]
            print(f"[run] ffmpeg failed: {err}", file=sys.stderr)
        self.proc.stderr.close()


_FOURCC = {"mp4v": "mp4v", "xvid": "XVID", "mjpg": "MJPG"}


def make_writer(path, fps, size, codec="auto"):
    """Pick a video writer, preferring one whose output actually plays."""
    have_ffmpeg = shutil.which("ffmpeg") is not None
    if codec == "auto":
        codec = "h264" if have_ffmpeg else "mp4v"
    if codec == "h264":
        if not have_ffmpeg:
            raise SystemExit("--codec h264 needs ffmpeg on PATH (sudo apt install ffmpeg)")
        return FFmpegWriter(path, fps, size)

    w = cv.VideoWriter(path, cv.VideoWriter_fourcc(*_FOURCC[codec]), max(fps, 1.0), size)
    if not w.isOpened():
        raise SystemExit(f"cannot open {path} for writing with codec {codec!r}")
    if codec == "mp4v":
        print("[run] warning: writing mp4v -- VLC plays it, browsers and most "
              "default players show black. Install ffmpeg for H.264.")
    return w


# ---------------------------------------------------------------------------
# intrinsics
# ---------------------------------------------------------------------------

def resolve_camera(args, frame_size, result):
    """Pick the best intrinsics available for this frame, and say where from.

    Preference order is trust order: a calibration file beats a measured fx,
    which beats a datasheet FOV, which beats V3's own guess -- but the guess is
    still far better than nothing, and it is what makes `--scale ground` work
    on a camera nobody has ever calibrated.
    """
    w, h = frame_size
    if args.calib:
        return CameraModel.from_calibration(args.calib, args.calib_camera).for_frame(w, h)
    if args.fx:
        return CameraModel.from_fx(w, h, args.fx)
    if args.hfov:
        return CameraModel.from_fov(w, h, args.hfov)
    if result is not None and result.K is not None:
        nh, nw = result.raw.shape[:2]
        return CameraModel.from_matrix(result.K, nw, nh, "V3 predicted").for_frame(w, h)
    return CameraModel.from_fov(w, h, 60.0)     # a plausible webcam, flagged as a guess


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------

def save_capture(outdir, tag, view_img, frame, depth, metric, cam, ply_stride, max_depth):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, tag)
    cv.imwrite(base + "_view.png", view_img)
    cv.imwrite(base + "_rgb.png", frame)
    np.save(base + "_depth.npy", depth.astype(np.float32))
    if metric:
        # 16-bit millimetres: the convention every RGB-D tool already reads,
        # and lossless out to 65 m.
        mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        cv.imwrite(base + "_depth_mm.png", mm)
    pts, rgb = backproject_rgb(depth, frame, cam, stride=ply_stride,
                               max_depth=max_depth if metric else None)
    n = write_ply(base + "_cloud.ply", pts, rgb)
    return base, n


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _fps(stamps):
    """Display rate over the last N frames, not the reciprocal of one gap."""
    if len(stamps) < 2:
        return 0.0
    span = stamps[-1] - stamps[0]
    return (len(stamps) - 1) / span if span > 1e-9 else 0.0


def build_parser():
    p = argparse.ArgumentParser(
        description="Live monocular depth with Depth Anything V3/V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", default="0", help="camera index, video file, image, or stream URL")
    p.add_argument("--model", default="v3-small", help="see --list-models")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--res", type=int, default=None, help="network resolution (multiple of 14)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--loop", action="store_true", help="restart video files at EOF")
    p.add_argument("--sync", dest="sync", action="store_true", default=None,
                   help="run the network on every frame, blocking the display "
                        "(the default for files); use for offline processing")
    p.add_argument("--async", dest="sync", action="store_false",
                   help="never block: the display runs at capture rate over the "
                        "newest available depth map (the default for cameras)")

    g = p.add_argument_group("intrinsics")
    g.add_argument("--calib", help="calibration YAML from ../calibrate.py")
    g.add_argument("--calib-camera", default="left")
    g.add_argument("--fx", type=float, help="focal length in pixels at capture resolution")
    g.add_argument("--hfov", type=float, help="lens horizontal field of view, degrees")

    g = p.add_argument_group("metric scale")
    g.add_argument("--scale", default="auto",
                   choices=["auto", "none", "anchor", "ground", "model"])
    g.add_argument("--camera-height", type=float, help="camera height above the floor, metres")
    g.add_argument("--anchor-range", type=float, default=1.0,
                   help="distance in metres assigned to a middle-click")
    g.add_argument("--scale-smooth", type=float, default=0.15)

    g = p.add_argument_group("display")
    g.add_argument("--view", default="split", choices=render.VIEWS)
    g.add_argument("--cmap", default="turbo", choices=[c[0] for c in render.COLORMAPS])
    g.add_argument("--near", type=float, help="pin the colour ramp's near end")
    g.add_argument("--far", type=float, help="pin the colour ramp's far end")
    g.add_argument("--alpha", type=float, default=0.6, help="overlay blend")
    g.add_argument("--no-display", action="store_true")
    g.add_argument("--scale-view", type=float, default=1.0, help="resize the window")

    g = p.add_argument_group("output")
    g.add_argument("--save", help="write the displayed view to this mp4")
    g.add_argument("--codec", default="auto", choices=["auto", "h264", "mp4v", "xvid", "mjpg"],
                   help="auto uses ffmpeg/H.264 when available, which is the only "
                        "one that plays in browsers and most desktop players")
    g.add_argument("--save-csv", help="log pinned points to CSV")
    g.add_argument("--outdir", default="captures")
    g.add_argument("--ply-stride", type=int, default=2)
    g.add_argument("--max-depth", type=float, default=30.0, help="far clip for saved clouds, m")
    g.add_argument("--max-frames", type=int, default=0)
    g.add_argument("--snapshot", action="store_true",
                   help="save one capture set (view, rgb, depth, cloud) and exit; "
                        "the headless equivalent of pressing 's'")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_models:
        print(f"{'name':24} {'backend':8} {'output':9} {'res':>4}  repo")
        for m in MODELS.values():
            print(f"{m.name:24} {m.backend:8} {m.kind:9} {m.res:>4}  {m.repo}"
                  + (f"  ({m.note})" if m.note else ""))
        return 0

    src = FrameSource(args.source, args.width, args.height, args.fps, args.loop)
    ok, frame = src.read()
    if not ok:
        raise SystemExit(f"no frames from {args.source}")
    print(f"[run] source {args.source} at {frame.shape[1]}x{frame.shape[0]}"
          + (f" ({src.src_fps:.0f} fps)" if src.src_fps else ""))

    model = load_model(args.model, res=args.res, device=args.device)
    print(f"[run] {args.model} on {model.device} ({', '.join(model.providers)}), "
          f"net res {model.res}, output '{model.spec.kind}'")

    mode = args.scale
    if mode == "auto":
        mode = "model" if model.spec.kind == "metric" else (
            "ground" if args.camera_height else "none")
    if mode == "model" and model.spec.kind != "metric":
        raise SystemExit(f"--scale model needs a metric model; {args.model} outputs "
                         f"'{model.spec.kind}'. Try --model v2-metric-indoor.")
    scaler = ScaleEstimator(mode, args.camera_height, args.scale_smooth)
    print(f"[run] scale mode: {mode}")

    painter = render.DepthPainter(
        cmap=[c[0] for c in render.COLORMAPS].index(args.cmap),
        fixed_range=(args.near, args.far) if (args.near and args.far) else None)

    # A camera hands you frames whether or not you are ready, so the network
    # takes the newest one and the display never waits. A file is not going
    # anywhere, so every frame gets its own depth map instead.
    sync = (not src.live) if args.sync is None else args.sync
    print(f"[run] mode: {'synchronous (every frame)' if sync else 'live (newest frame)'}")
    worker = None if sync else DepthWorker(model)
    if worker:
        worker.submit(frame, time.time())
    n_depth = 0

    state = {"probe": None, "pins": [], "split_x": frame.shape[1] // 2, "anchor": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv.EVENT_MOUSEMOVE:
            state["probe"] = (x, y)
            if flags & cv.EVENT_FLAG_SHIFTKEY:
                state["split_x"] = x
        elif event == cv.EVENT_LBUTTONDOWN:
            state["pins"].append((x, y))
        elif event == cv.EVENT_RBUTTONDOWN and state["pins"]:
            state["pins"].pop()
        elif event == cv.EVENT_MBUTTONDOWN:
            state["anchor"] = (x, y)

    window = "Depth Anything - live monocular depth"
    if not args.no_display:
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        cv.resizeWindow(window, frame.shape[1], frame.shape[0])
        cv.setMouseCallback(window, on_mouse)

    writer = None
    csv = open(args.save_csv, "w") if args.save_csv else None
    if csv:
        csv.write("frame,t_s,pin,u,v,value,units,scale,fx_px\n")

    view, paused, show_help, recording = args.view, False, False, bool(args.save)
    stamps = deque(maxlen=30)   # display timestamps, for the fps readout
    t_start = time.time()
    n_frames, last_note, note_until = 0, None, 0.0
    depth_full = paint_full = None
    near_is_high = False
    cam = None

    try:
        while True:
            if not paused:
                ok, new = src.read()
                if not ok:
                    print("[run] source ended")
                    break
                frame = new
                if worker:
                    worker.submit(frame, time.time())

            if sync:
                try:
                    latest = (model.infer(frame), frame, time.time())
                    n_depth += 1
                except Exception as e:
                    print(f"[run] inference error: {type(e).__name__}: {e}", file=sys.stderr)
                    break
            else:
                latest = worker.latest()
                if worker.error:
                    print(f"[run] inference error: {worker.error}", file=sys.stderr)
                    break

            h, w = frame.shape[:2]
            infer_ms, age_ms = 0.0, 0.0
            conf_bgr = None

            if latest is not None:
                result, _, stamp = latest
                infer_ms, age_ms = result.latency_ms, (time.time() - stamp) * 1000.0
                cam = resolve_camera(args, (w, h), result)

                depth = result.depth()
                scaler.update(depth, cam.for_frame(depth.shape[1], depth.shape[0]))

                if state["anchor"] is not None:
                    au, av = state["anchor"]
                    state["anchor"] = None
                    sy, sx = depth.shape[0] / h, depth.shape[1] / w
                    raw = float(depth[int(np.clip(av * sy, 0, depth.shape[0] - 1)),
                                      int(np.clip(au * sx, 0, depth.shape[1] - 1))])
                    if scaler.mode in ("none", "anchor"):
                        scaler.mode = "anchor"
                        scaler.set_anchor(raw, args.anchor_range)
                        last_note = f"anchored that point at {args.anchor_range:g} m"
                    else:
                        last_note = f"anchor ignored in '{scaler.mode}' mode"
                    note_until = time.time() + 2.5

                metric_depth = scaler.to_metres(depth)
                depth_use = metric_depth if metric_depth is not None else depth
                depth_full = cv.resize(depth_use, (w, h), interpolation=cv.INTER_LINEAR)
                # Probes, point clouds and scaling all read `depth_full`; the
                # colour ramp reads whatever the model paints best in.
                paint_src, near_is_high = result.display()
                paint_full = (depth_full if not near_is_high else
                              cv.resize(paint_src, (w, h), interpolation=cv.INTER_LINEAR))
                conf_bgr = render.paint_confidence(result.confidence, (w, h))

            if depth_full is None:            # first inference not back yet
                out = frame.copy()
                put_text(out, "warming up...", (12, 30), 0.7, (0, 220, 255), 2)
            else:
                depth_bgr = painter.paint(paint_full, near_is_high=near_is_high)
                out = render.compose(view, frame, depth_bgr, conf_bgr,
                                     args.alpha, state["split_x"])

                probe = None
                if state["probe"]:
                    pu, pv = state["probe"]
                    if 0 <= pu < w and 0 <= pv < h:
                        val = float(depth_full[pv, pu])
                        az, el = cam.bearing(pu, pv)
                        probe = (pu, pv, val, f"az {az:+.1f} el {el:+.1f}")

                pins = [(u, v, float(depth_full[v, u])) for u, v in state["pins"]
                        if 0 <= u < w and 0 <= v < h]
                render.draw_pins(out, pins, scaler.known)
                render.draw_hud(
                    out, model=args.model, device=model.device,
                    fps=_fps(stamps),
                    infer_ms=infer_ms, view=view,
                    cmap=render.COLORMAPS[painter.cmap][0],
                    depth_range=painter.range_of(paint_full),
                    range_units=("disp" if near_is_high else None),
                    scale_status=scaler.status, metric=scaler.known,
                    probe=probe, paused=paused, recording=recording and writer is not None,
                    cam=cam,
                    note=(last_note if time.time() < note_until else
                          (f"age {age_ms:.0f} ms" if age_ms > 200 else None)))
                if show_help:
                    render.draw_help(out)

                if csv and pins:
                    t = time.time() - t_start
                    for i, (u, v, val) in enumerate(pins):
                        csv.write(f"{n_frames},{t:.3f},{i},{u},{v},{val:.4f},"
                                  f"{'m' if scaler.known else 'rel'},"
                                  f"{scaler.scale or 0:.6f},{cam.fx:.2f}\n")

            if args.scale_view != 1.0:
                out = cv.resize(out, None, fx=args.scale_view, fy=args.scale_view)

            if args.save:
                if writer is None and recording:
                    writer = make_writer(args.save, max(src.src_fps or 0, 20.0),
                                         (out.shape[1], out.shape[0]), args.codec)
                if writer is not None and recording:
                    writer.write(out)

            if not args.no_display:
                cv.imshow(window, out)
                key = cv.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("c"):
                    painter.cycle_cmap()
                elif key == ord("v"):
                    view = render.VIEWS[(render.VIEWS.index(view) + 1) % len(render.VIEWS)]
                elif key == ord("i"):
                    painter.invert = not painter.invert
                elif key == ord("h"):
                    show_help = not show_help
                elif key == ord("x"):
                    state["pins"].clear()
                elif key == ord("f"):
                    painter.fixed_range = None if painter.fixed_range else painter.range_of(paint_full)
                    last_note = f"ramp {'released' if not painter.fixed_range else 'frozen'}"
                    note_until = time.time() + 1.5
                elif key in (ord("["), ord("]")) and hasattr(model, "res"):
                    model.res = max(126, model.res + (14 if key == ord("]") else -14))
                    last_note = f"net res {model.res}"
                    note_until = time.time() + 1.5
                elif key == ord("r"):
                    recording = not recording
                    if not args.save and recording:
                        args.save = os.path.join(args.outdir, f"rec_{int(time.time())}.mp4")
                        os.makedirs(args.outdir, exist_ok=True)
                    last_note = f"recording {'on' if recording else 'off'}"
                    note_until = time.time() + 1.5
                elif key == ord("s") and depth_full is not None:
                    tag = time.strftime("%Y%m%d_%H%M%S")
                    base, n = save_capture(args.outdir, tag, out, frame, depth_full,
                                           scaler.known, cam, args.ply_stride, args.max_depth)
                    last_note = f"saved {os.path.basename(base)}_* ({n} points)"
                    note_until = time.time() + 2.5
                    print(f"[run] {last_note}")

            if args.snapshot and depth_full is not None:
                tag = time.strftime("%Y%m%d_%H%M%S")
                base, n = save_capture(args.outdir, tag, out, frame, depth_full,
                                       scaler.known, cam, args.ply_stride, args.max_depth)
                print(f"[run] saved {base}_* ({n} points)")
                break

            stamps.append(time.perf_counter())
            n_frames += 1
            if args.max_frames and n_frames >= args.max_frames:
                if not args.snapshot:
                    break
                # A snapshot run exists to produce one capture; stopping on the
                # frame count before the first depth map arrives produces none.
                print("[run] frame limit reached; waiting for the first depth map")
                args.max_frames = 0
            # A still image has nothing to display until the first depth map
            # arrives; exit on the map itself rather than on the worker's
            # counter, which can tick between the two reads.
            if src.is_image and args.no_display and depth_full is not None:
                break
            if src.is_image and not sync:
                time.sleep(0.01)      # nothing new to capture; do not spin
            if depth_full is None and not sync and time.time() - t_start > 120:
                raise SystemExit("no depth map after 120 s -- see selftest.py")
    except KeyboardInterrupt:
        print("\n[run] interrupted")
    finally:
        if worker:
            worker.stop()
        src.release()
        if writer:
            writer.release()
            print(f"[run] wrote {args.save}")
        if csv:
            csv.close()
            print(f"[run] wrote {args.save_csv}")
        if not args.no_display:
            cv.destroyAllWindows()
    print(f"[run] {n_frames} frames displayed, "
          f"{n_depth if sync else worker.count} depth maps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
