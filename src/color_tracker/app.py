"""Command-line entry point and the interactive capture loop.

Controls (in the video window):
    left click  sample the color under the cursor and start tracking it
    c           clear the motion trail
    m           toggle the mask window
    q / Esc     quit
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2

from color_tracker import __version__
from color_tracker.colors import PRESETS, HSVRange, ranges_from_sample
from color_tracker.detection import DEFAULT_MIN_AREA, to_hsv, track_frame
from color_tracker.overlay import draw_overlay

WINDOW = "Color Tracker"
MASK_WINDOW = "Mask"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="color-tracker",
        description="Track a color in real time through your webcam, entirely on-device.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--color", choices=sorted(PRESETS), help="preset color to track")
    target.add_argument(
        "--hsv",
        nargs=6,
        type=int,
        metavar=("HL", "SL", "VL", "HU", "SU", "VU"),
        help="custom HSV range: lower H S V then upper H S V (H is 0-179)",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="camera index or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--min-area", type=float, default=DEFAULT_MIN_AREA, help="minimum blob area in pixels"
    )
    parser.add_argument("--trail", type=int, default=32, help="length of the motion trail")
    parser.add_argument(
        "--no-mirror", action="store_true", help="don't flip the image horizontally"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def initial_target(args: argparse.Namespace) -> tuple[list[HSVRange] | None, str]:
    if args.color:
        return PRESETS[args.color], args.color
    if args.hsv:
        return [(tuple(args.hsv[:3]), tuple(args.hsv[3:]))], "custom HSV"
    return None, ""


def open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open video source {source!r}. Is the camera connected and allowed?"
        )
    return cap


def run(args: argparse.Namespace) -> None:
    ranges, label = initial_target(args)
    trail: deque = deque(maxlen=args.trail)
    latest_frame = None
    show_mask = False

    def on_mouse(event, x, y, _flags, _param):
        nonlocal ranges, label
        if event == cv2.EVENT_LBUTTONDOWN and latest_frame is not None:
            pixel = to_hsv(latest_frame)[y, x]
            ranges = ranges_from_sample(pixel)
            label = f"picked HSV {tuple(int(p) for p in pixel)}"
            trail.clear()

    cap = open_source(args.source)
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    last_tick, fps = time.perf_counter(), None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            latest_frame = frame.copy()

            detection, mask = None, None
            if ranges:
                detection, mask = track_frame(frame, ranges, args.min_area)
                trail.appendleft(detection.center if detection else None)

            now = time.perf_counter()
            instant = 1.0 / max(now - last_tick, 1e-6)
            fps = instant if fps is None else 0.9 * fps + 0.1 * instant
            last_tick = now

            draw_overlay(frame, detection, trail, label, fps)
            cv2.imshow(WINDOW, frame)
            if show_mask and mask is not None:
                cv2.imshow(MASK_WINDOW, mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                trail.clear()
            elif key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow(MASK_WINDOW)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
