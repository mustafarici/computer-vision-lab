import numpy as np
import pytest

from modules.filters import apply_bilateral_filter, apply_median_filter


def _salt_and_pepper(rng):
    image = np.full((60, 60), 128, dtype=np.uint8)
    noise = rng.random((60, 60))
    image[noise < 0.05] = 0
    image[noise > 0.95] = 255
    return image


def test_median_removes_salt_and_pepper_noise():
    rng = np.random.default_rng(0)
    noisy = _salt_and_pepper(rng)

    filtered = apply_median_filter(noisy, 3)

    # The speckles are the only extreme values in a flat grey image, so
    # a working median filter leaves far fewer of them behind.
    def extremes(image):
        return int(((image == 0) | (image == 255)).sum())

    assert extremes(filtered) < extremes(noisy) / 10


def test_median_preserves_shape_and_channels(color_image):
    filtered = apply_median_filter(color_image, 5)

    assert filtered.shape == color_image.shape


def test_median_drops_alpha(color_image):
    alpha = np.full(color_image.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack([color_image, alpha])

    filtered = apply_median_filter(rgba, 3)

    assert filtered.shape[2] == 3


@pytest.mark.parametrize("kernel_size", [2, 4, 0, -1])
def test_median_rejects_invalid_kernel(color_image, kernel_size):
    with pytest.raises(ValueError, match="odd positive"):
        apply_median_filter(color_image, kernel_size)


def test_bilateral_preserves_edges_better_than_it_smooths_them():
    """
    A hard edge between two flat regions should survive bilateral
    filtering nearly intact, which is what separates it from a plain
    blur.
    """

    image = np.zeros((60, 60), dtype=np.uint8)
    image[:, 30:] = 255

    filtered = apply_bilateral_filter(image, 9, 75, 75)

    # Well away from the boundary the regions stay flat...
    assert filtered[30, 5] < 10
    assert filtered[30, 55] > 245
    # ...and the step across the boundary is still essentially full.
    assert int(filtered[30, 31]) - int(filtered[30, 28]) > 200


def test_bilateral_preserves_shape(color_image):
    filtered = apply_bilateral_filter(color_image, 9, 75, 75)

    assert filtered.shape == color_image.shape
