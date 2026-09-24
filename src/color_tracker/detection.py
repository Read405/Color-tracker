"""Frame processing pipeline: blur -> HSV -> mask -> largest blob."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from color_tracker.colors import HSVRange

DEFAULT_MIN_AREA = 300  # pixels; smaller blobs are treated as noise
_KERNEL = np.ones((5, 5), np.uint8)


@dataclass(frozen=True)
class Detection:
    """The tracked object in a single frame."""

    center: tuple[int, int]
    radius: int
    area: float
    bbox: tuple[int, int, int, int]


def to_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """Blur to suppress sensor noise, then convert BGR to HSV."""
    return cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (11, 11), 0), cv2.COLOR_BGR2HSV)


def build_mask(hsv: np.ndarray, ranges: Iterable[HSVRange]) -> np.ndarray:
    """Binary mask of pixels inside any of the ranges, with speckles removed."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, iterations=2)


def find_target(mask: np.ndarray, min_area: float = DEFAULT_MIN_AREA) -> Detection | None:
    """Return the largest blob in the mask, or None if nothing is big enough."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    moments = cv2.moments(largest)
    if area < min_area or moments["m00"] == 0:
        return None

    center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    _, radius = cv2.minEnclosingCircle(largest)
    return Detection(center, int(radius), area, cv2.boundingRect(largest))


def track_frame(
    frame_bgr: np.ndarray, ranges: Iterable[HSVRange], min_area: float = DEFAULT_MIN_AREA
) -> tuple[Detection | None, np.ndarray]:
    """Run the full pipeline on one BGR frame. Returns (detection, mask)."""
    mask = build_mask(to_hsv(frame_bgr), ranges)
    return find_target(mask, min_area), mask
