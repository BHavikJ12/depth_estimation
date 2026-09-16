"""Annotated-video output that actually plays.

OpenCV's bundled FFmpeg cannot encode H.264 — every `avc1`/`H264` fourcc fails
to open — so its only working mp4 codec is `mp4v`, MPEG-4 Part 2. Plenty of
players (browsers, GNOME Videos without extra codecs) render that as a black
frame while reporting the right duration and frame count, which looks exactly
like the pipeline produced nothing.

So when a system ffmpeg with libx264 is on PATH, frames are piped to it and the
result is ordinary H.264. Otherwise it falls back to OpenCV, and says so.
"""

import shutil
import subprocess
import sys

import cv2 as cv


def _ffmpeg_with_x264():
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return exe if "libx264" in out else None


class AnnotatedVideoWriter:
    """Same three calls as cv.VideoWriter: construct, write(frame), release()."""

    def __init__(self, path, width, height, fps=25.0, codec="auto", crf=20):
        self.path = str(path)
        # H.264 needs even dimensions; odd ones fail or produce garbage.
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self.fps = fps if fps and fps > 1 else 25.0
        self.crf = crf
        self._proc = None
        self._cv = None

        exe = _ffmpeg_with_x264() if codec in ("auto", "h264") else None
        if exe:
            self.backend = "h264 (ffmpeg)"
            self._open_ffmpeg(exe)
        else:
            if codec == "h264":
                raise RuntimeError("no ffmpeg with libx264 on PATH for --codec h264")
            self.backend = "mp4v (opencv)"
            self._cv = cv.VideoWriter(self.path, cv.VideoWriter_fourcc(*"mp4v"),
                                      self.fps, (self.width, self.height))
            if not self._cv.isOpened():
                raise RuntimeError(f"cannot open {self.path} for writing")
            print("note: writing MPEG-4 Part 2 because no ffmpeg with libx264 was "
                  "found. Some players show this as a black frame; install ffmpeg "
                  "for ordinary H.264.", file=sys.stderr)

    def _open_ffmpeg(self, exe):
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", f"{self.fps}",
            "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",          # the profile every player accepts
            "-movflags", "+faststart",
            self.path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE)

    def write(self, frame):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = frame[:self.height, :self.width]
        if self._proc is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                err = self._proc.stderr.read().decode(errors="replace")[:400]
                raise RuntimeError(f"ffmpeg stopped while encoding: {err}") from None
        else:
            self._cv.write(frame)

    def release(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            self._proc.wait(timeout=60)
            if self._proc.returncode not in (0, None):
                err = self._proc.stderr.read().decode(errors="replace")[:400]
                print(f"ffmpeg exited {self._proc.returncode}: {err}", file=sys.stderr)
            self._proc.stderr.close()
            self._proc = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None
