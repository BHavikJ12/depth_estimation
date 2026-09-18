"""One iterator over every kind of input: camera, video, stream, image, folder.

Keeping this in one place means run.py does not grow a branch per source type,
and it makes the awkward cases explicit — a folder of stills has no frame rate,
a live camera has no media clock, and a single image has neither.
"""

import glob
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import cv2 as cv

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_GLOB_CHARS = "*?["
CAMERA_STALL_TIMEOUT_S = 8.0


def _os_camera_names():
    """Human-readable device names, where the OS will tell us.

    OpenCV has no API for this: an index is all it offers. So ask the platform
    directly — v4l2 on Linux, PnP on Windows. Best effort; an empty result just
    means the listing falls back to resolution and a snapshot to tell cameras
    apart.
    """
    names = {}
    if sys.platform.startswith("linux"):
        for path in sorted(glob.glob("/sys/class/video4linux/video*")):
            try:
                idx = int(os.path.basename(path).replace("video", ""))
                with open(os.path.join(path, "name")) as fh:
                    names[idx] = fh.read().strip()
            except (OSError, ValueError):
                continue
    elif os.name == "nt":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_PnPEntity | "
                 "Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | "
                 "Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            # Windows does not expose which PnP device maps to which OpenCV
            # index, so these are listed in order as a hint, not a mapping.
            for i, line in enumerate(x for x in out.splitlines() if x.strip()):
                names[i] = line.strip()
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            pass
    return names


def list_cameras(max_index=8, snapshot_dir=None):
    """Probe camera indices and describe what is on each.

    Returns a list of dicts. If `snapshot_dir` is given, one frame from each
    working camera is saved there — with two identical-looking webcams that
    picture is usually the only way to tell which is which.
    """
    names = _os_camera_names()
    found = []
    # Probing an index that does not exist makes OpenCV print a multi-line C++
    # error for every miss, which buries the actual listing. Silence it for the
    # duration of the scan only.
    prev_log = None
    try:
        prev_log = cv.utils.logging.getLogLevel()
        cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass

    for idx in range(max_index + 1):
        cap = cv.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        info = {
            "index": idx,
            "name": names.get(idx, ""),
            "width": int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(cap.get(cv.CAP_PROP_FPS) or 0, 1),
            "backend": cap.getBackendName() if hasattr(cap, "getBackendName") else "",
            "readable": False,
            "brightness": None,
            "snapshot": None,
        }
        ok, frame = cap.read()
        if ok and frame is not None:
            info["readable"] = True
            info["height"], info["width"] = frame.shape[:2]
            # A lens cap, a privacy shutter or a disconnected sensor still opens
            # and still reads: it just returns near-black. Worth flagging.
            info["brightness"] = round(float(frame.mean()), 1)
            if snapshot_dir:
                os.makedirs(snapshot_dir, exist_ok=True)
                path = os.path.join(snapshot_dir, f"camera_{idx}.jpg")
                cv.imwrite(path, frame)
                info["snapshot"] = path
        cap.release()
        found.append(info)

    if prev_log is not None:
        try:
            cv.utils.logging.setLogLevel(prev_log)
        except AttributeError:
            pass
    return found


class FrameSource:
    """Iterable of BGR frames, with whatever timing information exists.

    kind is one of:
      camera   a webcam index; endless, no media clock
      video    a file or a stream URL (rtsp://, http://); has fps and timestamps
      image    one still
      images   a folder, or a glob, of stills in sorted order
    """

    def __init__(self, spec):
        self.spec = str(spec)
        self._cap = None
        self._paths = []
        self._camera_index = None
        self._frame_queue = None
        path = Path(self.spec)

        if self.spec.isdigit():
            self.kind = "camera"
            self._camera_index = int(self.spec)
            self._open(self._camera_index)
        elif path.is_dir():
            self.kind = "images"
            self._paths = sorted(p for p in path.iterdir()
                                 if p.suffix.lower() in IMAGE_SUFFIXES)
            if not self._paths:
                raise RuntimeError(f"no images in {path} "
                                   f"(looked for {', '.join(sorted(IMAGE_SUFFIXES))})")
        elif any(ch in self.spec for ch in _GLOB_CHARS):
            self.kind = "images"
            self._paths = sorted(Path(p) for p in glob.glob(self.spec)
                                 if Path(p).suffix.lower() in IMAGE_SUFFIXES)
            if not self._paths:
                raise RuntimeError(f"no images matched {self.spec!r}")
        elif path.suffix.lower() in IMAGE_SUFFIXES and path.exists():
            self.kind = "image"
            self._paths = [path]
        else:
            self.kind = "video"
            self._open(self.spec)

        if self.kind == "camera":
            self._start_reader()

        self.width, self.height = self._probe_size()

    def _open(self, target):
        self._cap = cv.VideoCapture(target)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"cannot open source {self.spec!r}. For a camera try another index "
                f"(0, 1, ...); for a file check the path; for a stream check the URL.")

    def _start_reader(self):
        """Reads frames on their own thread so a stalled USB camera (a real
        failure mode, not hypothetical) can never freeze the caller: cv2's
        VideoCapture.read() has no timeout and can block forever, but a
        caller blocked on Queue.get(timeout=...) can detect that and recover
        instead of hanging the whole pipeline with it."""
        self._frame_queue = queue.Queue(maxsize=1)
        threading.Thread(target=self._reader_loop, args=(self._cap,), daemon=True).start()

    def _reader_loop(self, cap):
        while True:
            ok, frame = cap.read()
            if not ok:
                return   # camera closed or errored; the consumer's timeout takes it from here
            try:
                self._frame_queue.get_nowait()   # drop any stale unread frame
            except queue.Empty:
                pass
            self._frame_queue.put(frame)

    def _reconnect_camera(self):
        """Called when no frame arrives within CAMERA_STALL_TIMEOUT_S. The old
        capture's reader thread may still be stuck inside a blocking read —
        there is no safe way to interrupt that from Python — so it is
        abandoned (it exits on its own if the read ever returns) and a fresh
        capture + reader thread takes over."""
        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = cv.VideoCapture(self._camera_index)
        self._start_reader()

    def _probe_size(self):
        if self.kind == "camera":
            w = int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            if w and h:
                return w, h
            # Some backends only know the size after a read. Pull it through
            # the same watchdog-protected queue the reader thread already
            # started (not a raw self._cap.read()) — otherwise a camera
            # that's stuck from the very first frame hangs at startup,
            # before the watchdog protecting steady-state iteration exists.
            try:
                frame = self._frame_queue.get(timeout=CAMERA_STALL_TIMEOUT_S)
            except queue.Empty:
                raise RuntimeError(
                    f"no frames from {self.spec!r} within "
                    f"{CAMERA_STALL_TIMEOUT_S:g}s of opening it")
            self._pending = frame
            return frame.shape[1], frame.shape[0]
        if self._cap is not None:
            w = int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            if w and h:
                return w, h
            ok, frame = self._cap.read()          # some backends only know after a read
            if not ok:
                raise RuntimeError(f"no frames from {self.spec!r}")
            self._pending = frame
            return frame.shape[1], frame.shape[0]
        frame = cv.imread(str(self._paths[0]))
        if frame is None:
            raise RuntimeError(f"cannot read image {self._paths[0]}")
        return frame.shape[1], frame.shape[0]

    # ---- properties ---------------------------------------------------

    @property
    def fps(self):
        if self._cap is None:
            return 0.0
        v = self._cap.get(cv.CAP_PROP_FPS)
        return v if v and v > 1 else 0.0

    @property
    def frame_count(self):
        if self._cap is None:
            return len(self._paths)
        return int(self._cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)

    @property
    def is_live(self):
        return self.kind == "camera"

    @property
    def is_single_image(self):
        return self.kind == "image"

    def media_time(self):
        """Seconds into the clip, or None when there is no media clock."""
        if self._cap is None:
            return None
        ms = self._cap.get(cv.CAP_PROP_POS_MSEC)
        return ms / 1000.0 if ms and ms > 0 else None

    def name(self, index):
        """A label for frame `index` — the filename for stills, else the index."""
        if self._paths and index < len(self._paths):
            return self._paths[index].name
        return f"{index:06d}"

    # ---- iteration ----------------------------------------------------

    def __iter__(self):
        if self.kind == "camera":
            pending = getattr(self, "_pending", None)
            if pending is not None:
                self._pending = None
                yield pending
            while True:
                try:
                    frame = self._frame_queue.get(timeout=CAMERA_STALL_TIMEOUT_S)
                except queue.Empty:
                    self._reconnect_camera()
                    continue
                yield frame
        elif self._cap is not None:
            pending = getattr(self, "_pending", None)
            if pending is not None:
                self._pending = None
                yield pending
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                yield frame
        else:
            for p in self._paths:
                frame = cv.imread(str(p))
                if frame is None:
                    continue                      # skip unreadable files, keep going
                yield frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self):
        bits = [f"{self.kind}", f"{self.width}x{self.height}"]
        if self.frame_count:
            bits.append(f"{self.frame_count} frames")
        if self.fps:
            bits.append(f"{self.fps:.1f} fps")
        return "  ".join(bits)
