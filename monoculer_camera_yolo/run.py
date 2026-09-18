#!/usr/bin/env python3
"""Monocular drone detection with range from a known target size.

    export ROBOFLOW_API_KEY=...
    python run.py --source 0 --hfov 70
    python run.py --source clip.mp4 --calib ../calibration_data/sample_calibration.yaml \
                  --save out.mp4 --save-csv track.csv --no-display
    python run.py --source 0 --endpoint http://localhost:9001 --hfov 70

Everything hangs off one assumption: the target is `--target-size` metres
across. Change the airframe and that number has to change with it, or every
range in the output is wrong by the same ratio.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2 as cv

from camera import CameraModel
from detector import DEFAULT_MODEL_ID, DroneDetector, list_versions
from overlay import annotate, status_lines
from ranging import (DEFAULT_SIZE_CV, DEFAULT_TARGET_SIZE_M, SPAN_MODES,
                     RangeEstimator, solve_target_size)
from sources import FrameSource
from tracker import DroneTracker
from video_writer import AnnotatedVideoWriter

WINDOW_TITLE = "monocular drone range"


def gui_available():
    """Whether opening a window will work here.

    This has to be decided *before* calling imshow, not by catching its error:
    with no display, OpenCV's Qt backend prints "no Qt platform plugin could be
    initialized" and aborts the process outright. There is no exception to
    catch, and a run that was writing a video loses it.

    Windows and macOS always have a window server. On Linux, a session with
    neither DISPLAY nor WAYLAND_DISPLAY is headless — an SSH session to a Pi,
    a systemd service, a container.
    """
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def build_camera(args, width, height):
    if args.calib:
        cam = CameraModel.from_calibration(args.calib, args.calib_camera)
        return cam.for_frame(width, height)
    if args.fx:
        return CameraModel.from_fx(width, height, args.fx, args.fy)
    return CameraModel.from_fov(width, height, args.hfov, args.vfov)


def process(frame, detector, tracker, estimator, dt=1.0, run_detector=True):
    """One frame: detect (or coast), track, range. Returns (tracks, locked).

    `dt` is in frames, not seconds. The tracker's only notion of time is the
    interval between measurements, which is one frame whether the clip is being
    chewed through at 400 fps or a webcam is dribbling out 12 — tying it to the
    wall clock would make the filter's lag depend on how busy the machine is.
    """
    if run_detector:
        tracks = tracker.update(detector.detect(frame), dt)
    else:
        for t in tracker.tracks:
            t.predict(dt)
        tracks = tracker.confirmed

    for t in tracks:
        t.range_estimate = estimator.estimate(t.box)
    return tracks, tracker.select_chase_target(frame.shape)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_argument_group("input")
    src.add_argument("--source", default="0",
                     help="webcam index (0, 1, ...), video file, stream URL, "
                          "image file, or a folder/glob of images (default: 0)")
    src.add_argument("--step", action="store_true",
                     help="wait for a keypress between frames (a single image "
                          "always waits)")
    src.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    src.add_argument("--stride", type=int, default=1,
                     help="run the detector every Nth frame; the tracker coasts "
                          "in between. Use 2-5 with --backend hosted (default: 1)")

    cam = p.add_argument_group("camera (pick one; --hfov is the fallback)")
    cam.add_argument("--calib", help="OpenCV calibration YAML, e.g. ../calibration_data/*.yaml")
    cam.add_argument("--calib-camera", default="left", choices=("left", "right"))
    cam.add_argument("--fx", type=float, help="focal length in pixels")
    cam.add_argument("--fy", type=float, help="vertical focal length (defaults to fx)")
    cam.add_argument("--hfov", type=float, default=70.0,
                     help="horizontal field of view in degrees (default: 70)")
    cam.add_argument("--vfov", type=float, help="vertical FOV; assumes square pixels if unset")

    tgt = p.add_argument_group("target")
    tgt.add_argument("--target-size", type=float, default=DEFAULT_TARGET_SIZE_M,
                     help=f"real span in metres (default: {DEFAULT_TARGET_SIZE_M} = 500 mm)")
    tgt.add_argument("--span-mode", default="max", choices=SPAN_MODES,
                     help="which box dimension is the span (default: max)")
    tgt.add_argument("--size-cv", type=float, default=DEFAULT_SIZE_CV,
                     help="1-sigma spread of the size prior, as a fraction (default: 0.20)")
    tgt.add_argument("--solve-size", type=float, metavar="RANGE_M",
                     help="calibration mode: the target in this frame is at RANGE_M "
                          "metres; print the --target-size that makes it so, and exit")

    det = p.add_argument_group("detector")
    det.add_argument("--backend", default="auto",
                     choices=("auto", "onnx", "hailo", "hosted", "server", "local", "ultralytics"),
                     help="onnx = the model's weights run locally with onnxruntime "
                          "(default); hailo = a .hef on a Hailo accelerator via "
                          "--weights; hosted = Roboflow cloud; server = a Roboflow "
                          "container on this machine (see --endpoint); ultralytics "
                          "= --weights; local = the `inference` package if installed")
    det.add_argument("--endpoint", help="base URL of a local Roboflow inference "
                                        "server, e.g. http://localhost:9001; "
                                        "implies --backend server")
    det.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                     help=f"Roboflow model id (default: {DEFAULT_MODEL_ID})")
    det.add_argument("--api-key", help="Roboflow key; defaults to $ROBOFLOW_API_KEY")
    det.add_argument("--weights", help="local .pt/.onnx for --backend ultralytics, "
                                       "or a .hef for --backend hailo")
    det.add_argument("--conf", type=float, default=0.40)
    det.add_argument("--overlap", type=float, default=0.30)
    det.add_argument("--max-side", type=int, default=0,
                     help="downscale the longest frame side before inference (0 = off)")
    det.add_argument("--keep-classes", help="comma-separated class ids to keep, "
                                            "e.g. 1 for this model's aircraft class")
    det.add_argument("--class-names", help="comma-separated names for the class ids")
    det.add_argument("--list-versions", action="store_true",
                     help="list the Roboflow project's trained versions and exit")

    trk = p.add_argument_group("tracker")
    trk.add_argument("--min-hits", type=int, default=3)
    trk.add_argument("--max-age", type=int, default=15)
    trk.add_argument("--iou", type=float, default=0.2)

    out = p.add_argument_group("output")
    out.add_argument("--save", help="write the annotated result here: a video "
                                    "file for video/camera input, an image file "
                                    "for one image, a folder for many images")
    out.add_argument("--save-csv", help="write per-frame range telemetry here")
    out.add_argument("--codec", default="auto", choices=("auto", "h264", "mp4v"),
                     help="video codec. auto/h264 use the system ffmpeg (libx264) "
                          "and produce a file every player can open; mp4v is "
                          "OpenCV's own, which some players render black")
    out.add_argument("--no-display", action="store_true",
                     help="do not open a preview window. Implied anyway on a "
                          "machine with no GUI, where the window is skipped "
                          "with a warning rather than crashing")
    out.add_argument("--quiet", action="store_true", help="no per-frame console output")
    out.add_argument("--web-port", type=int, default=0,
                     help="serve the annotated feed as MJPEG on this port; open "
                          "http://<this-device-ip>:PORT/ from any device on the "
                          "same network (0 = disabled)")

    args = p.parse_args(argv)

    if args.list_versions:
        info = list_versions(args.api_key)
        print(f"{info['name']}  (project-wide annotation counts: {info['classes']})\n")
        for v in info["versions"]:
            print(f"  --model-id {v['version']}   {v['name']}   "
                  f"images={v['images']}  mAP={v['map']}")
            if v["classes"]:
                names = ",".join(v["classes"])
                idx = "  ".join(f"{i}={c}" for i, c in enumerate(v["classes"]))
                print(f"      classes: {idx}")
                print(f"      --class-names '{names}'")
            else:
                print("      classes: not reported for this version")
        print("\nThe class index is the position in that list, and it differs "
              "between versions.\nPassing --keep-classes for an index a version "
              "does not have matches nothing at all.")
        return 0

    source = FrameSource(args.source)
    w, h = source.width, source.height
    camera = build_camera(args, w, h)
    estimator = RangeEstimator(camera, args.target_size, args.span_mode,
                               size_cv=args.size_cv)
    detector = DroneDetector(args.backend, args.model_id, args.api_key, args.conf,
                             args.overlap, args.weights, args.max_side or None,
                             args.endpoint,
                             [int(c) for c in args.keep_classes.split(",")] if args.keep_classes else None,
                             args.class_names.split(",") if args.class_names else None)
    tracker = DroneTracker(args.iou, args.max_age, args.min_hits)

    print(f"source   {args.source}  {source.describe()}")
    print(f"camera   {camera}")
    print(f"detector {detector.describe()}")
    print(f"target   {args.target_size * 1000:.0f}mm span, {args.span_mode} dimension; "
          f"~{estimator.max_useful_range(0.25):.0f}m at 25% error")
    if not (args.calib or args.fx):
        print("warning: no --calib or --fx, using --hfov "
              f"{args.hfov:g}deg. Every range scales linearly with this number.")

    # --- one-shot size calibration -------------------------------------
    if args.solve_size:
        frame = next(iter(source), None)
        if frame is None:
            print("no frames to calibrate against", file=sys.stderr)
            return 1
        dets = detector.detect(frame)
        if not dets:
            print("no detection in this frame; nothing to calibrate against", file=sys.stderr)
            return 1
        best = max(dets, key=lambda d: d.confidence)
        size = solve_target_size(camera, best.box, args.solve_size, args.span_mode)
        print(f"\nbox {tuple(round(v, 1) for v in best.box)} conf {best.confidence:.2f} "
              f"at {args.solve_size:g}m")
        print(f"--target-size {size:.4f}   ({size * 1000:.0f} mm effective span)")
        return 0

    writer = None
    csv_file = csv_writer = None
    if args.save_csv:
        csv_file = open(args.save_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "t_s", "track_id", "locked", "conf", "range_m",
                             "sigma_m", "x_m", "y_m", "z_m", "az_deg", "el_deg",
                             "span_px"])

    save_dir = None
    if args.save and source.kind == "images":
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)

    show_window = not args.no_display
    if show_window and not gui_available():
        show_window = False
        print("no display detected (no DISPLAY/WAYLAND_DISPLAY), so no preview "
              "window." + ("" if args.web_port else " Use --web-port to watch the "
              "annotated feed in a browser, or --save to write a video."),
              file=sys.stderr)
    web_broadcaster = None
    if args.web_port:
        from web_stream import start_server
        _, web_broadcaster = start_server(args.web_port)
        print(f"web stream: http://<this-device-ip>:{args.web_port}/")

    frame_idx = 0
    t_start = time.time()
    fps = 0.0
    try:
        for frame in source:
            now = time.time()

            run_det = (frame_idx % max(1, args.stride) == 0)
            tracks, locked = process(frame, detector, tracker, estimator, 1.0, run_det)

            inst = 1.0 / max(1e-6, time.time() - now)
            fps = inst if frame_idx == 0 else 0.9 * fps + 0.1 * inst

            annotate(frame, tracks, locked, camera,
                     status_lines(detector, camera, estimator, fps, len(tracks), locked))

            if web_broadcaster is not None:
                web_broadcaster.update(frame)

            if show_window:
                try:
                    cv.imshow(WINDOW_TITLE, frame)
                    wait = 0 if (args.step or source.is_single_image) else 1
                    if cv.waitKey(wait) & 0xFF in (27, ord("q")):
                        break
                except cv.error as exc:      # a GUI-less OpenCV build
                    show_window = False
                    print(f"cannot open a window, continuing without one ({exc})",
                          file=sys.stderr)

            if csv_writer:
                # A recorded clip's own timeline, so the log is replayable;
                # wall clock only for a live camera, which has no other.
                media = source.media_time()
                t_rel = media if media is not None else now - t_start
                for t in tracks:
                    e = t.range_estimate
                    csv_writer.writerow([
                        frame_idx, f"{t_rel:.3f}", t.id,
                        int(locked is not None and t.id == locked.id),
                        f"{t.confidence:.3f}",
                        *( [f"{e.range_m:.3f}", f"{e.sigma_m:.3f}", f"{e.x_m:.3f}",
                            f"{e.y_m:.3f}", f"{e.z_m:.3f}", f"{e.azimuth_deg:.2f}",
                            f"{e.elevation_deg:.2f}", f"{e.span_px:.1f}"]
                           if e else [""] * 8),
                    ])

            if args.save:
                if save_dir is not None:
                    cv.imwrite(str(save_dir / source.name(frame_idx)), frame)
                elif source.is_single_image:
                    cv.imwrite(args.save, frame)
                else:
                    if writer is None:
                        writer = AnnotatedVideoWriter(args.save, w, h, source.fps,
                                                      args.codec)
                        print(f"writing {args.save} as {writer.backend}")
                    writer.write(frame)

            if not args.quiet and locked is not None and locked.range_estimate:
                e = locked.range_estimate
                print(f"[{frame_idx:05d}] lock #{locked.id} conf {locked.confidence:.2f}  "
                      f"{e.range_m:6.2f} +/- {e.sigma_m:4.2f} m  "
                      f"az {e.azimuth_deg:+6.2f} el {e.elevation_deg:+6.2f}  "
                      f"span {e.span_px:5.1f}px")

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        source.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if show_window:
            cv.destroyAllWindows()

    print(f"\n{frame_idx} frame(s) in {time.time() - t_start:.1f}s")
    if args.save:
        print(f"wrote {args.save}")
    if args.save_csv:
        print(f"wrote {args.save_csv}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except RuntimeError as exc:
        # Missing key, unreachable endpoint, unopenable source: all expected
        # ways to be wrong, and none of them are worth a traceback.
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
