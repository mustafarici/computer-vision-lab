import numpy as np

from modules.basic_ops import (
    apply_gaussian_blur,
    apply_threshold,
    convert_to_grayscale,
)


def test_convert_to_grayscale_passthrough_for_2d(grayscale_only_image):
    result = convert_to_grayscale(grayscale_only_image)

    assert result.ndim == 2
    assert np.array_equal(result, grayscale_only_image)


def test_convert_to_grayscale_rgb(color_image):
    result = convert_to_grayscale(color_image)

    assert result.ndim == 2
    assert result.shape == color_image.shape[:2]


def test_convert_to_grayscale_rgba(color_image):
    alpha = np.full(color_image.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack([color_image, alpha])

    result = convert_to_grayscale(rgba)

    assert result.ndim == 2
    assert result.shape == color_image.shape[:2]


def test_apply_threshold_is_binary(grayscale_image):
    binary = apply_threshold(grayscale_image, 127)

    assert set(np.unique(binary)).issubset({0, 255})
    assert binary.shape == grayscale_image.shape


def test_apply_threshold_respects_boundary():
    # cv2.THRESH_BINARY: dst = maxval if src > thresh else 0
    grayscale = np.array([[0, 100, 127, 128, 255]], dtype=np.uint8)

    binary = apply_threshold(grayscale, 127)

    assert list(binary[0]) == [0, 0, 0, 255, 255]


def test_apply_gaussian_blur_smooths(grayscale_image):
    blurred = apply_gaussian_blur(grayscale_image, 5)

    assert blurred.shape == grayscale_image.shape
    assert blurred.dtype == grayscale_image.dtype
    # A blurred sharp-edged image should have less variance, not more.
    assert blurred.var() <= grayscale_image.var()
