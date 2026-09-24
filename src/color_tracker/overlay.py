"""Drawing helpers for the on-screen HUD."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from color_tracker.detection import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX
YELLOW, RED, WHITE, GREEN = (0, 255, 255), (0, 0, 255), (255, 255, 255), (0, 200, 0)


def draw_trail(frame: np.ndarray, trail: Sequence[tuple[int, int] | None]) -> None:
    """Draw a trail that thins out as points get older (index 0 is newest)."""
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        thickness = max(1, int(np.sqrt(len(trail) / float(i + 1)) * 2.5))
        cv2.line(frame, trail[i - 1], trail[i], RED, thickness)


def draw_detection(frame: np.ndarray, detection: Detection) -> None:
    x, y = detection.center
    cv2.circle(frame, detection.center, detection.radius, YELLOW, 2)
    cv2.circle(frame, detection.center, 5, RED, -1)
    cv2.putText(frame, f"({x}, {y})", (x + 10, y - 10), FONT, 0.5, WHITE, 2)


def draw_status(frame: np.ndarray, label: str, fps: float | None = None) -> None:
    text = f"Tracking: {label}" if label else "Click an object to track its color"
    if fps is not None:
        text += f"  |  {fps:.0f} FPS"
    cv2.putText(frame, text, (10, 25), FONT, 0.6, GREEN, 2)


def draw_overlay(
    frame: np.ndarray,
    detection: Detection | None,
    trail: Sequence[tuple[int, int] | None],
    label: str,
    fps: float | None = None,
) -> None:
    """Draw the full HUD (trail, target marker, status line) onto the frame in place."""
    draw_trail(frame, trail)
    if detection:
        draw_detection(frame, detection)
    draw_status(frame, label, fps)
