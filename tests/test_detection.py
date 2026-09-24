import cv2
import numpy as np
import pytest

from color_tracker.colors import PRESETS
from color_tracker.detection import track_frame

BGR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}


def frame_with_circle(bgr, center=(200, 150), radius=40):
    frame = np.full((300, 400, 3), 128, dtype=np.uint8)  # neutral gray background
    cv2.circle(frame, center, radius, bgr, -1)
    return frame


@pytest.mark.parametrize("color", sorted(BGR))
def test_finds_circle_of_each_color(color):
    detection, _ = track_frame(frame_with_circle(BGR[color]), PRESETS[color])
    assert detection is not None
    assert detection.center == pytest.approx((200, 150), abs=2)
    assert detection.radius == pytest.approx(40, abs=3)


def test_ignores_wrong_color():
    detection, _ = track_frame(frame_with_circle(BGR["green"]), PRESETS["blue"])
    assert detection is None


def test_ignores_blobs_below_min_area():
    detection, _ = track_frame(frame_with_circle(BGR["blue"], radius=5), PRESETS["blue"])
    assert detection is None


def test_picks_largest_blob():
    frame = frame_with_circle(BGR["blue"], center=(100, 100), radius=20)
    cv2.circle(frame, (300, 200), 50, BGR["blue"], -1)
    detection, _ = track_frame(frame, PRESETS["blue"])
    assert detection.center == pytest.approx((300, 200), abs=2)


def test_mask_matches_frame_size():
    frame = frame_with_circle(BGR["blue"])
    _, mask = track_frame(frame, PRESETS["blue"])
    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.uint8
