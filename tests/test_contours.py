import cv2
import numpy as np

from modules.contours import apply_contour_detection


def test_finds_expected_number_of_squares():
    binary = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(binary, (10, 10), (30, 30), 255, -1)  # area ~ 21*21 = 441
    cv2.rectangle(binary, (50, 50), (52, 52), 255, -1)  # tiny 3x3 = 9

    canvas, count = apply_contour_detection(
        binary, binary, "External", "Simple", min_area=50
    )

    # Only the larger square survives the area filter.
    assert count == 1
    assert canvas.shape == (100, 100, 3)


def test_min_area_filter_can_exclude_everything():
    binary = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(binary, (10, 10), (20, 20), 255, -1)

    _, count = apply_contour_detection(
        binary, binary, "External", "Simple", min_area=100_000
    )

    assert count == 0
