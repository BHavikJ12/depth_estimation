#!/usr/bin/env python3
"""Offline checks: no camera, and (mostly) no downloaded weights.

    ./venv/bin/python selftest.py            # everything that runs in a second
    ./venv/bin/python selftest.py --model    # also run the network, if present
    ./venv/bin/python selftest.py --bench    # ... and time it at each resolution

The geometry tests are the interesting ones. They build a synthetic scene whose
true depth is known analytically -- a floor plane under a tilted camera -- and
check that the ground-plane scaler recovers metres from it. If that passes, the
one thing that can still be wrong on real video is the model's own depth, which
no unit test can check for you.
"""

import argparse
import math
import sys
import time

import cv2 as cv
import numpy as np

import geometry as G
import metric as M
import models
import render

PASS, FAIL = "  ok  ", " FAIL "
_results = []


def check(name, fn):
    try:
        detail = fn()
        _results.append((True, name, detail or ""))
        print(f"[{PASS}] {name}" + (f"  {detail}" if detail else ""))
        return True
    except Exception as e:
        _results.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[{FAIL}] {name}\n         {type(e).__name__}: {e}")
        return False


def approx(a, b, tol, what=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{what}: {a:.6g} != {b:.6g} (tol {tol:g})")


# ---------------------------------------------------------------------------
# synthetic scene
# ---------------------------------------------------------------------------

def floor_scene(cam, height=1.30, pitch_deg=12.0, far=60.0):
    """Exact depth of a flat floor `height` below a camera pitched down."""
    vv, uu = np.mgrid[0:cam.height, 0:cam.width].astype(np.float32)
    x = (uu - cam.cx) / cam.fx
    y = (vv - cam.cy) / cam.fy
    z = np.ones_like(x)
    c, s = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    y_w = c * y + s * z          # the floor sits at Y = +height in world axes
    with np.errstate(divide="ignore", invalid="ignore"):
        t = height / y_w
    Z = t * z
    Z[(y_w <= 1e-6) | (Z <= 0) | (Z > far)] = far
    return np.nan_to_num(Z, nan=far, posinf=far).astype(np.float32)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def t_preproc():
    for (w, h) in [(640, 480), (1280, 720), (1920, 1080), (480, 640), (777, 333)]:
        nw, nh = models.net_size_da3(w, h, 504)
        assert nw % 14 == 0 and nh % 14 == 0, f"da3 {w}x{h} -> {nw}x{nh} not /14"
        assert abs(max(nw, nh) - 504) <= 14, f"da3 longest side {max(nw, nh)}"
        nw, nh = models.net_size_dpt(w, h, 518)
        assert nw % 14 == 0 and nh % 14 == 0, f"dpt {w}x{h} -> {nw}x{nh} not /14"
    x, (nw, nh) = models.preprocess(np.zeros((480, 640, 3), np.uint8), 504, "da3")
    assert x.shape == (3, nh, nw), x.shape
    # A black frame normalises to -mean/std, which is how you catch a missing
    # or doubled /255.
    approx(float(x[0].mean()), -0.485 / 0.229, 1e-4, "R channel of a black frame")
    return f"da3 640x480->{models.net_size_da3(640, 480, 504)}, dpt->{models.net_size_dpt(640, 480, 518)}"


def t_result_kinds():
    d = np.linspace(0.5, 4.0, 64, dtype=np.float32).reshape(8, 8)
    r = models.DepthResult(d, "depth")
    assert np.allclose(r.depth(), d) and not r.is_metric
    inv = models.DepthResult(1.0 / d, "inverse")
    # Reciprocating inverse depth must put the ordering back, to within the
    # deliberate clip of the far tail.
    assert np.allclose(inv.depth(), d, rtol=2e-2), "inverse round-trip"
    assert np.corrcoef(inv.depth().ravel(), d.ravel())[0, 1] > 0.999, "ordering lost"
    hard = models.DepthResult(np.array([[0.0, -1.0, 2.0, 10.0]], np.float32), "inverse")
    out = hard.depth()
    assert np.isfinite(out).all(), f"zero/negative disparity produced {out}"
    assert out.max() < 1e4, f"sky pixel escaped the clip: {out}"
    return "depth, inverse round-trip, non-finite guard"


def t_camera():
    cam = G.CameraModel.from_fov(640, 480, 70.0)
    approx(cam.hfov_deg, 70.0, 1e-6, "hfov round-trip")
    approx(cam.fx, 320 / math.tan(math.radians(35)), 1e-6, "fx from fov")
    half = cam.for_frame(320, 240)
    approx(half.fx, cam.fx / 2, 1e-6, "fx rescale")
    approx(half.hfov_deg, cam.hfov_deg, 1e-6, "fov invariant under rescale")
    az, el = cam.bearing(cam.cx, cam.cy)
    approx(az, 0, 1e-9, "boresight az")
    approx(el, 0, 1e-9, "boresight el")
    az, _ = cam.bearing(639, cam.cy)
    approx(az, 34.9, 0.2, "right edge az ~ hfov/2")
    return f"fx={cam.fx:.1f}px, edge bearing {az:.1f}deg"


def t_backproject():
    cam = G.CameraModel.from_fov(640, 480, 70.0)
    Z = 3.0
    pts = G.backproject(np.full((480, 640), Z, np.float32), cam)
    approx(pts[:, 2].min(), Z, 1e-5, "depth preserved")
    span = pts[:, 0].max() - pts[:, 0].min()
    approx(span, 2 * Z * math.tan(math.radians(35)), 0.02, "width of a plane at 3 m")
    return f"{len(pts)} pts, plane width {span:.3f} m"


def t_ply(tmp="/tmp/_selftest.ply"):
    pts = np.random.rand(500, 3).astype(np.float32)
    rgb = (np.random.rand(500, 3) * 255).astype(np.uint8)
    n = G.write_ply(tmp, pts, rgb)
    head = open(tmp, "rb").read(200).split(b"end_header")[0]
    assert b"element vertex 500" in head and b"property uchar red" in head
    import os
    size = os.path.getsize(tmp)
    assert size == len(head) + len(b"end_header\n") + 500 * 15, f"bad size {size}"
    return f"{n} vertices, {size} bytes"


def t_ground_scale():
    """The headline claim: metres out of a relative depth map."""
    cam = G.CameraModel.from_fov(640, 480, 70.0)
    truth = floor_scene(cam, height=1.30, pitch_deg=12.0)
    worst = 0.0
    for unit in (1.0, 0.037, 14.2, 260.0):      # arbitrary per-frame scales
        est = M.ScaleEstimator("ground", camera_height=1.30, smooth=1.0)
        est.update(truth / unit, cam)
        assert est.known, f"no scale at unit {unit}"
        approx(est.scale, unit, unit * 0.02, f"scale at unit {unit}")
        err = np.abs(est.to_metres(truth / unit) - truth).max()
        worst = max(worst, err)
    assert worst < 0.05, f"max depth error {worst:.3f} m"
    return f"scale recovered over 4 decades, worst depth error {worst*1000:.1f} mm"


def t_ground_rejects_wall():
    """A camera facing a wall has no floor; the scaler must say so, not guess."""
    cam = G.CameraModel.from_fov(640, 480, 70.0)
    wall = np.full((480, 640), 2.0, np.float32)     # fronto-parallel, normal = -Z
    est = M.ScaleEstimator("ground", camera_height=1.30)
    est.update(wall, cam)
    assert not est.known, f"accepted a wall as the floor (scale {est.scale})"
    assert "lost" in est.status, est.status
    return f"rejected, status {est.status!r}"


def t_ground_survives_clutter():
    """Boxes on the floor must not drag the plane fit off."""
    cam = G.CameraModel.from_fov(640, 480, 70.0)
    truth = floor_scene(cam, height=1.30, pitch_deg=15.0)
    scene = truth.copy()
    rng = np.random.default_rng(3)
    for _ in range(6):                              # obstacles, 30% nearer
        y, x = rng.integers(300, 440), rng.integers(0, 560)
        scene[y:y + 40, x:x + 80] *= 0.7
    scene += rng.normal(0, 0.01, scene.shape).astype(np.float32)   # sensor noise
    est = M.ScaleEstimator("ground", camera_height=1.30, smooth=1.0)
    est.update(scene, cam)
    assert est.known, "lost the floor under clutter"
    approx(est.scale, 1.0, 0.05, "scale with clutter")
    return f"scale {est.scale:.4f} with 6 obstacles + noise ({est.inliers} inliers)"


def t_anchor():
    est = M.ScaleEstimator("anchor")
    assert not est.known
    assert est.set_anchor(0.25, 3.0)
    approx(est.scale, 12.0, 1e-9, "anchor scale")
    d = np.array([[0.25, 0.5]], np.float32)
    assert np.allclose(est.to_metres(d), [[3.0, 6.0]])
    assert not est.set_anchor(0.0, 3.0), "accepted a zero-depth anchor"
    return "3 m at 0.25 rel -> scale 12.0"


def t_painter():
    d = np.linspace(1, 8, 640, np.float32)[None].repeat(480, 0)
    p = render.DepthPainter(smooth=1.0)
    img = p.paint(d)
    assert img.shape == (480, 640, 3)
    lo, hi = p.range_of(d)
    # Percentile clipping, not min/max: 2% of 1..8 is ~1.14.
    assert 1.1 < lo < 1.2 and 7.8 < hi < 7.9, (lo, hi)
    # Near must be the warm end by default.
    near, far = img[240, 5].astype(int), img[240, 635].astype(int)
    assert near[2] > near[0] and far[0] > far[2], f"ramp direction near={near} far={far}"
    spike = d.copy()
    spike[0, 0] = 1e6
    lo2, hi2 = render.DepthPainter(smooth=1.0).range_of(spike)
    assert hi2 < 10, f"one outlier pixel blew the range out to {hi2}"
    return f"range {lo:.2f}-{hi:.2f}, outlier-resistant"


def t_views():
    frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    depth = np.linspace(1, 6, 640, np.float32)[None].repeat(480, 0)
    painter = render.DepthPainter()
    dbgr = painter.paint(depth)
    cbgr = render.paint_confidence(depth, (640, 480))
    for v in render.VIEWS:
        out = render.compose(v, frame, dbgr, cbgr, 0.6, 320)
        assert out.shape == frame.shape, (v, out.shape)
    return f"{len(render.VIEWS)} views"


def t_hud(path="selftest_hud.png"):
    frame = np.full((480, 854, 3), 60, np.uint8)
    cv.circle(frame, (600, 300), 120, (200, 190, 170), -1)
    depth = floor_scene(G.CameraModel.from_fov(854, 480, 70), 1.3, 14.0)
    painter = render.DepthPainter()
    out = render.compose("split", frame, painter.paint(depth), None, 0.6, 427)
    render.draw_pins(out, [(300, 300, 2.35), (700, 210, 5.1)], True)
    render.draw_hud(out, model="v3-small", device="cuda", fps=24.3, infer_ms=48.8,
                    view="split", cmap="turbo", depth_range=painter.range_of(depth),
                    scale_status="ground 1.3m (8640 pts)", metric=True,
                    probe=(500, 260, 3.42, "az +7.2 el -1.1"), recording=True,
                    cam=G.CameraModel.from_fov(854, 480, 70), note="saved capture")
    render.draw_help(out)
    cv.imwrite(path, out)
    return f"wrote {path}"


def t_registry():
    for name, spec in models.MODELS.items():
        assert spec.kind in ("depth", "inverse", "metric"), (name, spec.kind)
        assert spec.preproc in ("da3", "dpt"), (name, spec.preproc)
        assert spec.res % 14 == 0, (name, spec.res)
        if spec.backend == "onnx":
            assert spec.path.endswith(".onnx"), (name, spec.path)
    v3 = [n for n, s in models.MODELS.items() if s.kind == "depth"]
    met = [n for n, s in models.MODELS.items() if s.kind == "metric"]
    return f"{len(models.MODELS)} models ({len(v3)} V3, {len(met)} metric)"


def t_providers():
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" not in avail:
        # A deliberate CPU install (macOS, ARM, no NVIDIA card) lands here and
        # is not a failure -- say which it is rather than implying a mistake.
        import platform
        return (f"CPU only -- {ort.__name__} has no CUDA provider "
                f"({platform.system()}/{platform.machine()})")
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    # Being listed is not the same as loading: the CUDA provider is a separate
    # .so, and it fails at session creation when the CUDA/cuDNN pip wheels are
    # not preloaded -- the failure mode that silently halves your frame rate.
    import os
    probe = None
    for name, spec in models.MODELS.items():
        if spec.backend == "onnx" and os.path.exists(os.path.join("models", name, spec.path)):
            probe = os.path.join("models", name, spec.path)
            break
    if probe is None:
        return "CUDA provider listed; no weights on disk to load it with"
    s = ort.InferenceSession(probe, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if "CUDAExecutionProvider" not in s.get_providers():
        raise RuntimeError("CUDA provider listed but refused to load")
    return f"CUDA provider loads ({os.path.basename(os.path.dirname(os.path.dirname(probe)))})"


def t_model_infer(name, res=None):
    import os
    spec = models.MODELS[name]
    if spec.backend == "onnx" and not os.path.exists(os.path.join("models", name, spec.path)):
        return f"skipped: weights not downloaded (run.py fetches them)"
    m = models.load_model(name, res=res, verbose=False)
    img = cv.imread("/tmp/soh.png") if os.path.exists("/tmp/soh.png") else None
    if img is None:
        img = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    r = m.infer(img)
    assert r.raw.ndim == 2 and np.isfinite(r.raw).all(), "non-finite depth"
    assert r.raw.std() > 1e-6, "flat depth map"
    extra = ""
    if r.K is not None:
        cam = G.CameraModel.from_matrix(r.K, r.raw.shape[1], r.raw.shape[0])
        assert 10 < cam.hfov_deg < 170, f"absurd predicted hfov {cam.hfov_deg}"
        extra = f", predicted fx {cam.fx:.0f}px / {cam.hfov_deg:.0f}deg hfov"
    return (f"{m.device}, {r.raw.shape[1]}x{r.raw.shape[0]}, {r.latency_ms:.0f} ms, "
            f"range {r.raw.min():.2f}..{r.raw.max():.2f}{extra}")


def bench(name="v3-small", reps=12):
    import os
    spec = models.MODELS[name]
    if spec.backend == "onnx" and not os.path.exists(os.path.join("models", name, spec.path)):
        print(f"[skip] {name}: weights not downloaded")
        return
    img = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
    print(f"\n{'res':>5} {'net size':>12} {'ms':>8} {'fps':>7}")
    for res in (252, 308, 364, 420, 504, 616):
        m = models.load_model(name, res=res, verbose=False)
        m.infer(img)
        t0 = time.perf_counter()
        for _ in range(reps):
            m.infer(img)
        dt = (time.perf_counter() - t0) / reps
        nw, nh = models.net_size_da3(1280, 720, res) if spec.preproc == "da3" \
            else models.net_size_dpt(1280, 720, res)
        print(f"{res:>5} {f'{nw}x{nh}':>12} {dt*1000:>8.1f} {1/dt:>7.1f}")
        del m


def main():
    ap = argparse.ArgumentParser(description="offline checks for the depth pipeline")
    ap.add_argument("--model", nargs="?", const="v3-small", default=None,
                    help="also run this model on one frame")
    ap.add_argument("--res", type=int)
    ap.add_argument("--bench", action="store_true", help="time the model at each resolution")
    args = ap.parse_args()

    print("=== preprocessing and model plumbing ===")
    check("preprocess: patch alignment and normalisation", t_preproc)
    check("DepthResult: depth / inverse / metric", t_result_kinds)
    check("model registry", t_registry)

    print("\n=== geometry ===")
    check("camera model", t_camera)
    check("back-projection", t_backproject)
    check("PLY export", t_ply)

    print("\n=== metric scale ===")
    check("ground plane -> metres", t_ground_scale)
    check("ground plane rejects a wall", t_ground_rejects_wall)
    check("ground plane survives clutter", t_ground_survives_clutter)
    check("click anchor", t_anchor)

    print("\n=== rendering ===")
    check("colour ramp", t_painter)
    check("view composition", t_views)
    check("HUD", t_hud)

    print("\n=== runtime ===")
    check("onnxruntime providers", t_providers)

    if args.model or args.bench:
        name = args.model or "v3-small"
        print(f"\n=== {name} ===")
        check(f"{name}: one forward pass", lambda: t_model_infer(name, args.res))
        if args.bench:
            bench(name)

    bad = [r for r in _results if not r[0]]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} passed")
    if bad:
        for _, name, detail in bad:
            print(f"  FAILED: {name} -- {detail}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
