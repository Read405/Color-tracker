"""Color Tracker - track a given color through your webcam, entirely locally.

Usage examples:
    python color_tracker.py --color blue
    python color_tracker.py --hsv 35 80 60 85 255 255
    python color_tracker.py            # then click on an object to track its color

Controls (in the video window):
    left click  sample the color under the cursor and start tracking it
    c           clear the motion trail
    m           toggle the mask window
    q / Esc     quit
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# HSV ranges for OpenCV (H: 0-179, S: 0-255, V: 0-255).
# Red wraps around the hue circle, so it needs two ranges.
PRESETS: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "orange": [((10, 120, 70), (22, 255, 255))],
    "yellow": [((22, 100, 100), (35, 255, 255))],
    "green": [((36, 80, 50), (85, 255, 255))],
    "blue": [((90, 100, 50), (130, 255, 255))],
    "purple": [((130, 80, 50), (160, 255, 255))],
    "pink": [((160, 60, 100), (170, 255, 255))],
}

MIN_AREA = 300  # ignore blobs smaller than this many pixels


@dataclass
class Detection:
    center: tuple[int, int]
    radius: int
    area: float
    bbox: tuple[int, int, int, int]


def build_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    """Return a cleaned binary mask of pixels that fall inside any of the HSV ranges."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_target(mask: np.ndarray, min_area: float = MIN_AREA) -> Detection | None:
    """Find the largest blob in the mask, or None if nothing big enough is there."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    _, radius = cv2.minEnclosingCircle(largest)
    return Detection((cx, cy), int(radius), area, cv2.boundingRect(largest))


def track_frame(frame_bgr: np.ndarray, ranges, min_area: float = MIN_AREA):
    """Run the full pipeline on one BGR frame. Returns (detection or None, mask)."""
    blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, ranges)
    return find_target(mask, min_area), mask


def ranges_from_sample(hsv_pixel, h_tol: int = 10, s_tol: int = 60, v_tol: int = 60):
    """Build HSV range(s) around a sampled pixel, handling hue wraparound."""
    h, s, v = (int(x) for x in hsv_pixel)
    s_lo, s_hi = max(s - s_tol, 40), min(s + s_tol, 255)
    v_lo, v_hi = max(v - v_tol, 40), min(v + v_tol, 255)
    lo_h, hi_h = h - h_tol, h + h_tol
    if lo_h < 0:
        return [((0, s_lo, v_lo), (hi_h, s_hi, v_hi)), ((180 + lo_h, s_lo, v_lo), (179, s_hi, v_hi))]
    if hi_h > 179:
        return [((lo_h, s_lo, v_lo), (179, s_hi, v_hi)), ((0, s_lo, v_lo), (hi_h - 180, s_hi, v_hi))]
    return [((lo_h, s_lo, v_lo), (hi_h, s_hi, v_hi))]


def draw_overlay(frame, detection: Detection | None, trail: deque, label: str) -> None:
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        thickness = int(np.sqrt(len(trail) / float(i + 1)) * 2.5)
        cv2.line(frame, trail[i - 1], trail[i], (0, 0, 255), thickness)

    if detection:
        cv2.circle(frame, detection.center, detection.radius, (0, 255, 255), 2)
        cv2.circle(frame, detection.center, 5, (0, 0, 255), -1)
        x, y = detection.center
        cv2.putText(frame, f"({x}, {y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    status = f"Tracking: {label}" if label else "Click an object to track its color"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Track a color through your webcam, locally.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--color", choices=sorted(PRESETS), help="preset color to track")
    group.add_argument("--hsv", nargs=6, type=int, metavar=("HL", "SL", "VL", "HU", "SU", "VU"),
                       help="custom HSV range: lower H S V then upper H S V (H is 0-179)")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--min-area", type=float, default=MIN_AREA, help="minimum blob area in pixels")
    parser.add_argument("--trail", type=int, default=32, help="length of the motion trail")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    state = {"ranges": None, "label": ""}
    if args.color:
        state["ranges"], state["label"] = PRESETS[args.color], args.color
    elif args.hsv:
        state["ranges"] = [(tuple(args.hsv[:3]), tuple(args.hsv[3:]))]
        state["label"] = "custom HSV"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Is it plugged in / allowed?")

    window = "Color Tracker"
    cv2.namedWindow(window)
    latest = {"frame": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and latest["frame"] is not None:
            hsv = cv2.cvtColor(cv2.GaussianBlur(latest["frame"], (11, 11), 0), cv2.COLOR_BGR2HSV)
            pixel = hsv[y, x]
            state["ranges"] = ranges_from_sample(pixel)
            state["label"] = f"picked HSV {tuple(int(p) for p in pixel)}"
            trail.clear()

    cv2.setMouseCallback(window, on_mouse)
    trail: deque = deque(maxlen=args.trail)
    show_mask = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            frame = cv2.flip(frame, 1)  # mirror so it feels natural
            latest["frame"] = frame.copy()

            detection, mask = None, None
            if state["ranges"]:
                detection, mask = track_frame(frame, state["ranges"], args.min_area)
                trail.appendleft(detection.center if detection else None)

            draw_overlay(frame, detection, trail, state["label"])
            cv2.imshow(window, frame)
            if show_mask and mask is not None:
                cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                trail.clear()
            if key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("Mask")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
