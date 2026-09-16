"""Verify the HITNET setup and benchmark each model on this machine.

Checks, in order: onnxruntime and its execution providers, the ONNX model
files, the calibration file, and the camera. Then times every installed
model on synthetic frames so you can pick one that actually keeps up.

Usage:
    python check_hitnet_setup.py                 # checks + benchmark
    python check_hitnet_setup.py --no-bench      # checks only
    python check_hitnet_setup.py --runs 20
"""

import argparse
import gc
import glob
import os
import time

import numpy as np

OK, BAD, WARN = "  [ok] ", "  [!!] ", "  [--] "


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-bench", action="store_true", help="skip the timing pass")
    p.add_argument("--runs", type=int, default=10, help="timed iterations per model")
    p.add_argument("--eye-width", type=int, default=1280)
    p.add_argument("--eye-height", type=int, default=720)
    return p.parse_args()


def check_runtime():
    print("onnxruntime")
    try:
        import onnxruntime as ort
    except ImportError:
        print(BAD + "not installed -- pip install 'onnxruntime-gpu[cuda,cudnn]'")
        return False, []

    providers = ort.get_available_providers()
    print(OK + f"version {ort.__version__}")
    print(OK + f"providers: {', '.join(providers)}")
    if "CUDAExecutionProvider" not in providers:
        print(WARN + "no CUDA provider -- inference will run on CPU (slow)")
    return True, providers


def check_models():
    print("\nmodels")
    paths = sorted(glob.glob("models/hitnet/*/*/model_float32.onnx"))
    if not paths:
        print(BAD + "none found -- run ./download_hitnet_models.sh")
        return []
    for path in paths:
        size_mb = os.path.getsize(path) / 1e6
        print(OK + f"{path}  ({size_mb:.1f} MB)")
    return paths


def check_calibration():
    print("\ncalibration")
    from stereo import config
    if os.path.exists(config.CALIB_FILE):
        print(OK + f"{config.CALIB_FILE} (real calibration)")
        return config.CALIB_FILE
    if os.path.exists(config.SAMPLE_CALIB_FILE):
        print(WARN + f"only the nominal sample ({config.SAMPLE_CALIB_FILE}) is present. "
                     "Depth will be approximate -- run `python calibrate.py`.")
        return config.SAMPLE_CALIB_FILE
    print(BAD + "no calibration file -- run `python generate_sample_calibration.py` "
                "then `python calibrate.py`")
    return None


def check_camera():
    print("\ncamera")
    import cv2 as cv
    from stereo import config
    cap = cv.VideoCapture(config.CAMERA_INDEX, cv.CAP_V4L2)
    if not cap.isOpened():
        print(BAD + f"could not open /dev/video{config.CAMERA_INDEX} -- "
                    "check `v4l2-ctl --list-devices` and CAMERA_INDEX in stereo/config.py")
        return False
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*config.FOURCC))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(BAD + "camera opened but returned no frame")
        return False
    h, w = frame.shape[:2]
    if (w, h) != (config.FRAME_WIDTH, config.FRAME_HEIGHT):
        print(BAD + f"got {w}x{h}, expected {config.FRAME_WIDTH}x{config.FRAME_HEIGHT} "
                    "(the combined side-by-side stereo frame)")
        return False
    print(OK + f"{w}x{h} combined frame -> two {w // 2}x{h} eyes")
    return True


def benchmark(paths, runs, eye_w, eye_h):
    from stereo.hitnet import HitNet

    print(f"\nbenchmark ({eye_w}x{eye_h} per eye, {runs} runs each)")
    # Noise rather than flat grey: HITNET's cost volume is data dependent, so
    # a textureless input would time faster than any real scene.
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (eye_h, eye_w, 3), dtype=np.uint8)
    right = np.roll(left, -12, axis=1)

    print(f"  {'model':<44}{'ms':>9}{'FPS':>8}")
    for path in paths:
        label = "/".join(path.split(os.sep)[2:4])
        net = None
        try:
            net = HitNet(path)
            net.compute(left, right)          # warm-up: first run pays for allocation
            start = time.perf_counter()
            for _ in range(runs):
                net.compute(left, right)
            ms = (time.perf_counter() - start) / runs * 1000
            print(f"  {label:<44}{ms:>9.1f}{1000 / ms:>8.1f}")
        except Exception as exc:                                  # noqa: BLE001
            reason = str(exc)
            if ("ALLOC_FAILED" in reason or "Failed to allocate memory" in reason
                    or "out of memory" in reason.lower()):
                reason = "out of GPU memory for this input size"
            else:
                reason = f"{type(exc).__name__}: {reason.splitlines()[0][:90]}"
            print(f"  {label:<44}{'--':>9}{'--':>8}   {reason}")
        finally:
            # Each session reserves its own CUDA arena. Without dropping it
            # here the later (bigger) models benchmark against a card that
            # earlier models are still holding, and spuriously fail.
            del net
            gc.collect()


def main():
    args = parse_args()
    runtime_ok, _ = check_runtime()
    paths = check_models() if runtime_ok else []
    check_calibration()
    check_camera()
    if runtime_ok and paths and not args.no_bench:
        benchmark(paths, args.runs, args.eye_width, args.eye_height)
    print("\nRun the live viewer with:  python live_depth_hitnet.py")


if __name__ == "__main__":
    main()
