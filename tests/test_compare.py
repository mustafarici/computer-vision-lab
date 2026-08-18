import numpy as np
import pytest

from modules.compare import (
    compute_difference,
    is_comparable,
    summarize_difference,
)

# ==================================================
# ELIGIBILITY
# ==================================================


def test_a_result_of_the_same_size_is_comparable(color_image):
    result = color_image.copy()

    assert is_comparable(color_image, result)


def test_a_grayscale_result_is_still_comparable(color_image, grayscale_image):
    """Channel counts may differ; only the pixel grid has to match."""

    assert is_comparable(color_image, grayscale_image)


def test_a_wider_composite_is_not_comparable(color_image):
    """
    The RGB/HSV channel stages return three frames side by side. There
    is no meaningful "before" to lay next to that, which is exactly
    what this flag exists to catch.
    """

    composite = np.hstack([color_image, color_image, color_image])

    assert not is_comparable(color_image, composite)


def test_a_figure_is_not_comparable(color_image):
    """The histogram stages return a matplotlib figure, not an array."""

    assert not is_comparable(color_image, object())


# ==================================================
# DIFFERENCE
# ==================================================


def test_difference_of_an_unchanged_image_is_black(color_image):
    difference = compute_difference(color_image, color_image.copy())

    assert difference.shape == color_image.shape
    assert difference.max() == 0


def test_difference_isolates_the_changed_region(color_image):
    modified = color_image.copy()
    modified[10:20, 10:20] = 255 - modified[10:20, 10:20]

    difference = compute_difference(color_image, modified)

    assert difference[10:20, 10:20].max() > 0
    # Everything the stage didn't touch stays black.
    assert difference[40:, 40:].max() == 0


def test_difference_lifts_a_grayscale_result_to_rgb(
    color_image, grayscale_image
):
    difference = compute_difference(color_image, grayscale_image)

    assert difference.shape == (*grayscale_image.shape, 3)


def test_difference_rejects_mismatched_sizes(color_image):
    composite = np.hstack([color_image, color_image])

    with pytest.raises(ValueError, match="same height and width"):
        compute_difference(color_image, composite)


# ==================================================
# SUMMARY
# ==================================================


def test_summary_reports_no_change_when_there_is_none():
    difference = np.zeros((10, 10, 3), dtype=np.uint8)

    assert "completely unchanged" in summarize_difference(difference)


def test_summary_reports_the_changed_share():
    difference = np.zeros((10, 10, 3), dtype=np.uint8)
    # A quarter of the pixels changed.
    difference[:5, :5] = 40

    summary = summarize_difference(difference)

    assert "25.0%" in summary
    # Mean over the whole image: 40 * 0.25.
    assert "10.0" in summary


def test_summary_handles_a_single_channel_difference():
    difference = np.zeros((10, 10), dtype=np.uint8)
    difference[:5, :] = 255

    assert "50.0%" in summarize_difference(difference)
