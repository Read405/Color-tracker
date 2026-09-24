import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from color_tracker import PRESETS, ranges_from_sample, track_frame  # noqa: E402


def frame_with_circle(bgr, center=(200, 150), radius=40):
    frame = np.full((300, 400, 3), 128, dtype=np.uint8)  # gray background
    cv2.circle(frame, center, radius, bgr, -1)
    return frame


def test_finds_blue_circle():
    detection, _ = track_frame(frame_with_circle((255, 0, 0)), PRESETS["blue"])
    assert detection is not None
    assert abs(detection.center[0] - 200) <= 2 and abs(detection.center[1] - 150) <= 2


def test_finds_red_across_hue_wraparound():
    detection, _ = track_frame(frame_with_circle((0, 0, 255), center=(100, 100)), PRESETS["red"])
    assert detection is not None
    assert abs(detection.center[0] - 100) <= 2


def test_ignores_wrong_color():
    detection, _ = track_frame(frame_with_circle((0, 255, 0)), PRESETS["blue"])
    assert detection is None


def test_ignores_tiny_blobs():
    detection, _ = track_frame(frame_with_circle((255, 0, 0), radius=5), PRESETS["blue"])
    assert detection is None


def test_sampled_range_wraps_hue():
    ranges = ranges_from_sample((3, 200, 200))
    assert len(ranges) == 2
    ranges = ranges_from_sample((60, 200, 200))
    assert len(ranges) == 1
