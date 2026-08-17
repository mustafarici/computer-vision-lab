import numpy as np

from modules.enhancement import apply_clahe, apply_histogram_equalization


def _low_contrast_image():
    """A noisy image squeezed into a narrow intensity band."""

    rng = np.random.default_rng(0)
    return rng.integers(100, 140, (80, 80), dtype=np.uint8)


def test_equalization_widens_the_intensity_range():
    image = _low_contrast_image()

    equalized = apply_histogram_equalization(image)

    original_spread = int(image.max()) - int(image.min())
    new_spread = int(equalized.max()) - int(equalized.min())

    assert new_spread > original_spread
    assert equalized.shape == image.shape


def test_equalization_increases_standard_deviation():
    image = _low_contrast_image()

    assert apply_histogram_equalization(image).std() > image.std()


def test_clahe_also_raises_contrast():
    image = _low_contrast_image()

    result = apply_clahe(image, 2.0, 8)

    assert result.shape == image.shape
    assert result.std() > image.std()


def test_higher_clip_limit_gives_more_contrast():
    image = _low_contrast_image()

    gentle = apply_clahe(image, 1.0, 8)
    aggressive = apply_clahe(image, 10.0, 8)

    assert aggressive.std() >= gentle.std()
