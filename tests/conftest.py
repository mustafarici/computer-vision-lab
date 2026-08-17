"""Shared fixtures for the modules/ unit tests.

The fixtures build small synthetic images instead of depending on
sample files on disk, so the test suite is self-contained and fast.
"""

import cv2
import numpy as np
import pytest


@pytest.fixture
def color_image():
    """
    64x64 RGB test image (channel order matches what load_image()
    produces: R, G, B):

    - a white square at (10,10)-(30,30), useful for
      threshold/contour/morphology tests
    - a red patch at (40,40)-(55,55), useful for HSV/color
      threshold tests
    """

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (30, 30), (255, 255, 255), -1)
    cv2.rectangle(image, (40, 40), (55, 55), (255, 0, 0), -1)
    return image


@pytest.fixture
def grayscale_image(color_image):
    """The color_image fixture, converted to single-channel grayscale."""

    return cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)


@pytest.fixture
def grayscale_only_image():
    """
    A genuinely single-channel (L-mode) image, as produced by
    load_image() for grayscale-only uploads (ndim == 2, no color
    information at all — distinct from grayscale_image, which is a
    3-channel image converted to gray).
    """

    image = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (30, 30), 255, -1)
    return image
