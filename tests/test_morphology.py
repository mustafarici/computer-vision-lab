import cv2
import numpy as np

from modules.morphology import (
    apply_closing,
    apply_dilation,
    apply_erosion,
    apply_opening,
)


def _white_square():
    image = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (40, 40), 255, -1)
    return image


def test_erosion_shrinks_foreground():
    binary = _white_square()

    eroded = apply_erosion(binary, 5, 1)

    assert np.count_nonzero(eroded) < np.count_nonzero(binary)


def test_dilation_grows_foreground():
    binary = _white_square()

    dilated = apply_dilation(binary, 5, 1)

    assert np.count_nonzero(dilated) > np.count_nonzero(binary)


def test_opening_removes_small_noise():
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[5, 5] = 255  # isolated single-pixel speck

    opened = apply_opening(binary, 3, 1)

    assert np.count_nonzero(opened) == 0


def test_closing_fills_small_hole():
    binary = _white_square()
    binary[29, 29] = 0  # single-pixel hole inside the square

    closed = apply_closing(binary, 3, 1)

    assert closed[29, 29] == 255
