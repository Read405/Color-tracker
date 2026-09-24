import pytest

from color_tracker.colors import H_MAX, PRESETS, ranges_from_sample


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_are_valid_hsv(name):
    for lower, upper in PRESETS[name]:
        assert 0 <= lower[0] <= upper[0] <= H_MAX
        assert all(0 <= lo <= hi <= 255 for lo, hi in zip(lower[1:], upper[1:], strict=True))


def test_sample_in_middle_of_hue_gives_one_range():
    assert ranges_from_sample((60, 200, 200)) == [((50, 140, 140), (70, 255, 255))]


def test_sample_near_zero_wraps_to_top():
    ranges = ranges_from_sample((3, 200, 200))
    hue_spans = sorted((lo[0], hi[0]) for lo, hi in ranges)
    assert hue_spans == [(0, 13), (173, 179)]


def test_sample_near_top_wraps_to_zero():
    ranges = ranges_from_sample((175, 200, 200))
    hue_spans = sorted((lo[0], hi[0]) for lo, hi in ranges)
    assert hue_spans == [(0, 5), (165, 179)]


def test_sample_never_accepts_near_black():
    ((lower, _),) = ranges_from_sample((60, 10, 10))
    assert lower[1] >= 40 and lower[2] >= 40
