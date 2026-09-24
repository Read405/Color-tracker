"""Color definitions and helpers for building HSV ranges.

OpenCV uses H in 0-179, S and V in 0-255. Hue is circular, so colors near the
wrap point (red) are represented as two ranges.
"""

from __future__ import annotations

HSV = tuple[int, int, int]
HSVRange = tuple[HSV, HSV]

PRESETS: dict[str, list[HSVRange]] = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "orange": [((10, 120, 70), (22, 255, 255))],
    "yellow": [((22, 100, 100), (35, 255, 255))],
    "green": [((36, 80, 50), (85, 255, 255))],
    "blue": [((90, 100, 50), (130, 255, 255))],
    "purple": [((130, 80, 50), (160, 255, 255))],
    "pink": [((160, 60, 100), (170, 255, 255))],
}

H_MAX = 179
MIN_SV = 40  # floor for sampled S/V so we never match near-gray or near-black


def ranges_from_sample(
    hsv_pixel, h_tol: int = 10, s_tol: int = 60, v_tol: int = 60
) -> list[HSVRange]:
    """Build HSV range(s) centered on a sampled pixel, handling hue wraparound."""
    h, s, v = (int(x) for x in hsv_pixel)
    s_lo, s_hi = max(s - s_tol, MIN_SV), min(s + s_tol, 255)
    v_lo, v_hi = max(v - v_tol, MIN_SV), min(v + v_tol, 255)
    lo_h, hi_h = h - h_tol, h + h_tol

    if lo_h < 0:
        return [
            ((0, s_lo, v_lo), (hi_h, s_hi, v_hi)),
            ((H_MAX + 1 + lo_h, s_lo, v_lo), (H_MAX, s_hi, v_hi)),
        ]
    if hi_h > H_MAX:
        return [
            ((lo_h, s_lo, v_lo), (H_MAX, s_hi, v_hi)),
            ((0, s_lo, v_lo), (hi_h - H_MAX - 1, s_hi, v_hi)),
        ]
    return [((lo_h, s_lo, v_lo), (hi_h, s_hi, v_hi))]
