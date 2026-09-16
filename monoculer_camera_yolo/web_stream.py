"""Minimal MJPEG HTTP server for viewing the annotated feed live, from any
device on the same network, with nothing beyond the standard library and
OpenCV (already a dependency).

    from web_stream import start_server
    server, broadcaster = start_server(8000)
    ...
    broadcaster.update(annotated_frame_bgr)   # call once per frame

Then open http://<this-device-ip>:8000/ in a browser on any device on the
same network (phone, laptop) — no player or extra software needed.
"""

import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2 as cv

_INDEX_HTML = b"""<!doctype html>
<title>drone range - live</title>
<body style="margin:0;background:#000">
<img src="/stream.mjpg" style="width:100%;height:100vh;object-fit:contain">
"""


class FrameBroadcaster:
    """Holds only the latest frame and wakes waiting clients when it changes,
    so viewers see it push-style instead of each polling in a loop."""

    def __init__(self, quality=80):
        self.quality = quality
        self._condition = threading.Condition()
        self._jpeg = None

    def update(self, frame_bgr):
        ok, buf = cv.imencode(".jpg", frame_bgr,
                              [int(cv.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self._condition:
            self._jpeg = buf.tobytes()
            self._condition.notify_all()

    def next_jpeg(self, timeout=5.0):
        """Blocks until a new frame arrives or timeout; returns None on
        timeout, which the caller uses as a keep-alive check."""
        with self._condition:
            if self._condition.wait(timeout=timeout):
                return self._jpeg
            return None


class _Handler(BaseHTTPRequestHandler):
    # Without this, a stalled client (locked phone screen, dropped wifi that
    # never sends a clean TCP close) leaves wfile.write() blocked forever —
    # the thread never exits, and repeated reconnects pile up stuck threads
    # until the Pi runs out of memory/file descriptors.
    timeout = 10

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_INDEX_HTML)))
            self.end_headers()
            self.wfile.write(_INDEX_HTML)
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    jpeg = self.server.broadcaster.next_jpeg()
                    if jpeg is None:
                        continue   # no frame yet / timed out waiting; keep the connection open
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass   # viewer closed the tab, or stalled past `timeout`; either way, done
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass   # one line per HTTP request would drown out run.py's own output


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server(port, host="0.0.0.0", quality=80):
    """Starts the server in a background thread and returns (server, broadcaster).
    Call broadcaster.update(frame) once per frame; nothing else to drive."""
    broadcaster = FrameBroadcaster(quality)
    server = _ThreadingHTTPServer((host, port), _Handler)
    server.broadcaster = broadcaster
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, broadcaster
