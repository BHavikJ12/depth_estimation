# Bugs found running this on real Hailo-8 hardware

This is the postmortem doc — what actually broke, why, and how it was fixed.
[SETUP_RPI_HAILO.md](SETUP_RPI_HAILO.md) has the commands needed to avoid
these; this doc has the reasoning, for anyone debugging something similar or
wondering why a step in the setup guide looks the way it does.

---

## Compiling: three packaging/config problems in Model Zoo 5.4.0

Hit while running `hailomz compile` for the first time (see
[SETUP_RPI_HAILO.md](SETUP_RPI_HAILO.md#34-compile) for the actual commands).

### a) Dynamic input shape

The ONNX reports its input as `[batch, 3, height, width]` with every dim
dynamic, and parsing fails with `Could not parse the model due to dynamic
shapes`. The `--tensor-shapes` flag the error message itself suggests does
not exist on `hailomz compile` in this version — passing it produces
`unrecognized arguments`. The error message was written for a different
Model Zoo release than the one installed here.

Fixed by resolving the shape on the ONNX itself, with `onnxsim` (already
installed as a Dataflow Compiler dependency), before compiling:

```bash
onnxsim weights.onnx weights_fixed.onnx --overwrite-input-shape images:1,3,640,640
```

### b) Missing NMS config file in the wheel

Compiling then failed later, during `_handle_classes_argument`, with:

```
FileNotFoundError: .../hailo_model_zoo/cfg/postprocess_config/yolov5s_nms_config.json
```

That whole `postprocess_config/` directory is absent from the
`hailo_model_zoo-5.4.0` wheel — checked directly, every `*_nms_config.json`
across the entire package is missing, not just this one model's. A genuine
packaging gap in that wheel, not a setup mistake.

Fixed by pulling the file from the matching GitHub release tag instead of
guessing its contents — it encodes real anchor box values, and wrong ones
would silently produce bad boxes rather than an error, which is a much worse
failure mode than a `FileNotFoundError`:

```bash
gh api "repos/hailo-ai/hailo_model_zoo/contents/hailo_model_zoo/cfg/postprocess_config/yolov5s_nms_config.json?ref=v5.4.0" \
    --jq '.content' | base64 -d > .../cfg/postprocess_config/yolov5s_nms_config.json
```

`?ref=` should match whatever `hailomz --version` actually reports, not
necessarily `v5.4.0`.

### c) Detection-head layer names don't match

Compiling then failed with:

```
HailoNNException: The layer named conv63 doesn't exist in the HN
```

The config's `bbox_decoders` hardcode the *reference* `yolov5s`'s internal
layer names (`conv55`/`conv63`/`conv70`) for its three detection heads. The
actual model here is narrower (fewer parameters than Hailo's reference
`yolov5s` — it's really a `yolov5v6n`, `yolov5s` was only used as the closest
available compile template), so its equivalent heads land at different
graph indices entirely.

Found the real ones by inspecting the `.har` that `hailomz compile` saves
even on a failing attempt:

```python
from hailo_sdk_client import ClientRunner
runner = ClientRunner(har="yolov5s.har")
hn = runner.get_hn_model()
print([l.name for l in hn.get_real_output_layers()])   # -> conv47, conv54, conv60
```

Those three, in that order, filled the same stride-8/16/32 roles the
reference names did. The anchor `w`/`h` values didn't need to change — only
each decoder's `encoded_layer` field, remapped from the reference names to
the real ones found above.

**If compiling a different model later:** expect (b) and (c) to recur under
different filenames/layer names — (b) is a wheel-wide gap, not
`yolov5s`-specific, and (c) recurs for any model narrower/wider than Hailo's
reference of the same name. An anchor-free architecture (YOLOv8, YOLOv11)
has a differently-shaped NMS config with no anchors at all, so (c)'s exact
patch doesn't carry over to a different architecture family — only the
*method* (inspect the `.har`, find the real output layers) does.

---

## The Hailo backend itself: two activation bugs

Found running a compiled `.hef` against real Hailo-8 hardware for the first
time (AI HAT+, HailoRT 4.23.0) — this code had never touched real hardware
before, only been offline-tested.

**Bug 1 — wrong order.** `hailo_model.py`'s `_ensure_open()` created and
entered the `InferVStreams` pipeline *before* activating the network group.
HailoRT requires the opposite order: the network group must already be
active before an inference pipeline is built on top of it. `close()` had the
matching bug in reverse — deactivating before closing the pipeline that was
still using it.

**Bug 2 — the real fix, less obvious.** Fixing the order alone did not fix
the symptom. The line was:

```python
self._activated = self.network_group.activate(self.ng_params).__enter__()
```

This chains everything into one expression, keeping only what `__enter__()`
*returns* — not the context-manager object itself, which is what actually
holds the "network group is active" state via RAII (its `__exit__`
deactivates it). Hailo's own official example code never binds this specific
`with` block to a variable at all — that's the tell that `__enter__()`
returns nothing useful here. With no reference to the real handle left
anywhere, Python's reference-counted garbage collector could free it —
deactivating the network group — almost immediately, before any inference
ever ran.

Fixed by keeping the handle itself in `self._activation_cm`, and calling
`__enter__()`/`__exit__()` on *that*, not on whatever it returns:

```python
def _ensure_open(self):
    if self._pipeline is None:
        self._activation_cm = self.network_group.activate(self.ng_params)
        self._activation_cm.__enter__()
        self._pipeline = InferVStreams(self.network_group, self.in_params,
                                       self.out_params).__enter__()

def close(self):
    if self._pipeline is not None:
        self._pipeline.__exit__(None, None, None)
        self._pipeline = None
    if self._activation_cm is not None:
        self._activation_cm.__exit__(None, None, None)
        self._activation_cm = None
```

Both bugs together produced the identical symptom —
`HailoRTNetworkGroupNotActivatedException` /
`HAILO_NETWORK_GROUP_NOT_ACTIVATED(69)` on every inference call. If you hit
this and the ordering already looks right, check for bug 2 specifically —
it's the one that's easy to miss.

**Still not verified even after these fixes:** the actual output format and
box positions. The `bbox_decoders` fix above (section c) used stock COCO
anchor values, not values checked against this specific trained model's real
anchors — auto-anchor training can shift those away from the defaults.
Before trusting any range/position number this backend reports, compare its
boxes against the same footage run through `--backend onnx` (known-good) and
look for a systematic offset or scale error, which would point at the
anchors, not a code bug.

---

## Running unattended: three bugs that only show up over hours

A quick test run in a terminal doesn't exercise any of these — they only
surfaced when actually trying to run this unattended, for hours, headless.

### The web viewer's thread leak

`web_stream.py`'s per-viewer HTTP handler had no socket timeout. A stalled
client — a locked phone screen, dropped wifi that never sends a clean TCP
close — left `wfile.write()` blocked *forever* inside that thread. Every
stall left one more stuck thread behind; over hours these accumulate and
exhaust the Pi's memory. This is a genuine "works fine at first, degrades
after some time" failure — not a crash you'd see in the first few minutes of
testing.

Fixed with a 10-second timeout on the handler:

```python
class _Handler(BaseHTTPRequestHandler):
    timeout = 10
```

and catching `TimeoutError` alongside the existing disconnect exceptions.
Verified with an actual test: a client that stops reading mid-stream gets
its thread cleaned up within the timeout window instead of hanging forever.

### The camera read can freeze the whole pipeline

`cv2.VideoCapture.read()` has no timeout. If the USB camera's connection
hiccups — a real failure mode on a long unattended run — that call can block
forever. Since it sat directly in `run.py`'s main loop, the *entire*
pipeline froze with it: no new detections, no new frames reaching the web
stream, while the process itself stayed alive and `ssh` kept working fine
the whole time. That specific combination — reachable over SSH, completely
unresponsive otherwise — is this bug's signature, distinct from the thread
leak above (which shows as climbing memory/thread count over time) and from
a genuine network/router failure (which takes SSH down too).

Fixed in `sources.py`: the camera is now read on its own background thread
into a 1-slot queue; the main loop waits on that queue with an 8-second
timeout instead of on the raw blocking read. No frame within that window
reopens the camera from scratch. The old stuck thread is abandoned, not
killed — there is no safe way to interrupt a blocked C call from Python — and
is left to exit on its own if the read ever returns. This also covers the
narrower case where the *very first* read (used to probe the frame size at
startup) stalls — that path used to bypass the watchdog entirely and hang at
startup with no error at all; it now fails fast with a clear message instead.

Verified with a fake camera object that simulates a permanent stall: the
pipeline recovers via reconnect during steady-state iteration, and fails
fast with a clear error (instead of hanging) if the camera never produces a
single frame at startup.

### Two lower-stakes long-run fixes

* `tracker.py`: the Kalman filter's covariance matrix is now re-symmetrized
  after every update (`self.P = (self.P + self.P.T) / 2.0`). Floating-point
  drift can slowly make it numerically non-symmetric over a very long run —
  standard defensive practice for a filter meant to run for hours, not a fix
  for an observed crash.
* `video_writer.py`: `release()` now catches `subprocess.TimeoutExpired`
  from a hung `ffmpeg` and kills it, instead of raising out of `run.py`'s
  `finally` block and silently skipping the CSV file close that came after
  it. Only matters when using `--save` for video output.
