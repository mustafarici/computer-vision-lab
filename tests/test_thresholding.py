import numpy as np
import pytest

from modules.thresholding import apply_adaptive_threshold, apply_otsu_threshold


def test_otsu_lands_between_two_intensity_clusters():
    """
    Two noisy clusters, one dark and one bright. Otsu should pick a
    threshold in the valley between them.

    (The clusters need real spread: given an image containing only two
    exact values, every threshold between them separates the classes
    equally well and OpenCV returns the lowest one, which says nothing
    about whether the method works.)
    """

    rng = np.random.default_rng(0)

    image = np.zeros((60, 60), dtype=np.uint8)
    image[:, :30] = rng.integers(40, 80, (60, 30))
    image[:, 30:] = rng.integers(170, 210, (60, 30))

    binary, chosen = apply_otsu_threshold(image)

    # Derived from the data rather than hard-coded, so the assertion
    # states the property: the threshold separates the clusters cleanly.
    dark_peak = image[:, :30].max()
    bright_peak = image[:, 30:].min()

    assert dark_peak <= chosen < bright_peak
    assert set(np.unique(binary)) == {0, 255}
    assert binary[0, 0] == 0
    assert binary[0, 59] == 255


def test_otsu_output_matches_input_shape(grayscale_image):
    binary, _ = apply_otsu_threshold(grayscale_image)

    assert binary.shape == grayscale_image.shape


def test_adaptive_threshold_handles_a_lighting_gradient():
    """
    A global threshold can't binarize an image whose background
    brightness ramps across the frame; a local one can. This is the
    whole reason adaptive thresholding exists.
    """

    gradient = np.tile(
        np.linspace(0, 255, 100, dtype=np.uint8), (100, 1)
    )
    # A constant-contrast square sitting on top of the ramp.
    gradient[40:60, 40:60] = np.clip(
        gradient[40:60, 40:60].astype(int) + 40, 0, 255
    ).astype(np.uint8)

    result = apply_adaptive_threshold(gradient, "Gaussian", 11, 2)

    assert result.shape == gradient.shape
    assert set(np.unique(result)).issubset({0, 255})
    # The square must survive as foreground somewhere.
    assert result[40:60, 40:60].max() == 255


@pytest.mark.parametrize("method", ["Mean", "Gaussian"])
def test_both_adaptive_methods_run(grayscale_image, method):
    result = apply_adaptive_threshold(grayscale_image, method, 11, 2)

    assert result.shape == grayscale_image.shape


def test_unknown_adaptive_method_raises(grayscale_image):
    with pytest.raises(ValueError, match="Unknown adaptive method"):
        apply_adaptive_threshold(grayscale_image, "Bilinear", 11, 2)


@pytest.mark.parametrize("block_size", [2, 10, 1, 0])
def test_invalid_block_size_raises(grayscale_image, block_size):
    """OpenCV would abort with a bare C++ assertion; we want a message."""

    with pytest.raises(ValueError, match="odd number"):
        apply_adaptive_threshold(grayscale_image, "Mean", block_size, 2)
