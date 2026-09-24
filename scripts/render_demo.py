"""Render docs/demo.png: the real tracking pipeline and HUD run on a synthetic scene.

Run from the repo root:  python scripts/render_demo.py
"""

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from color_tracker import PRESETS, track_frame
from color_tracker.overlay import draw_overlay

OUT = Path(__file__).resolve().parents[1] / "docs" / "demo.png"
W, H = 640, 360


def scene(ball_center):
    """A gray-gradient backdrop with distractor shapes and a blue ball."""
    frame = np.tile(np.linspace(70, 140, W, dtype=np.uint8), (H, 1))
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame, (60, 230), (160, 320), (40, 170, 40), -1)  # green box
    cv2.circle(frame, (560, 90), 35, (30, 30, 210), -1)  # red ball
    cv2.circle(frame, ball_center, 32, (210, 110, 20), -1)  # blue ball (target)
    return frame


def main():
    path = [(90 + 5 * t, int(200 - 110 * np.sin(t / 22))) for t in range(0, 71, 2)]
    trail: deque = deque(maxlen=32)
    frame, detection = None, None
    for center in path:
        frame = scene(center)
        detection, mask = track_frame(frame, PRESETS["blue"])
        trail.appendleft(detection.center if detection else None)

    draw_overlay(frame, detection, trail, "blue")
    mask_small = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (W // 4, H // 4))
    cv2.rectangle(mask_small, (0, 0), (W // 4 - 1, H // 4 - 1), (255, 255, 255), 1)
    frame[H - H // 4 - 10 : H - 10, W - W // 4 - 10 : W - 10] = mask_small
    cv2.putText(
        frame,
        "mask",
        (W - W // 4 - 6, H - H // 4 - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    OUT.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT), frame)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
