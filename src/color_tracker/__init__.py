"""Color Tracker: real-time, on-device color tracking with OpenCV."""

from color_tracker.colors import PRESETS, HSVRange, ranges_from_sample
from color_tracker.detection import Detection, build_mask, find_target, track_frame

__all__ = [
    "PRESETS",
    "Detection",
    "HSVRange",
    "build_mask",
    "find_target",
    "ranges_from_sample",
    "track_frame",
]

__version__ = "1.0.0"
