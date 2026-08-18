import numpy as np

from modules.image_utils import same_frame_size, to_rgb


def test_to_rgb_expands_a_single_channel_image():
    grayscale = np.zeros((8, 8), dtype=np.uint8)

    assert to_rgb(grayscale).shape == (8, 8, 3)


def test_to_rgb_repeats_the_intensity_across_channels():
    grayscale = np.full((4, 4), 120, dtype=np.uint8)

    rgb = to_rgb(grayscale)

    assert (rgb[:, :, 0] == rgb[:, :, 1]).all()
    assert (rgb[:, :, 1] == rgb[:, :, 2]).all()
    assert rgb[0, 0, 0] == 120


def test_to_rgb_drops_an_alpha_channel():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)

    assert to_rgb(rgba).shape == (8, 8, 3)


def test_to_rgb_leaves_rgb_alone(color_image):
    assert np.array_equal(to_rgb(color_image), color_image)


def test_same_frame_size_ignores_channel_count():
    assert same_frame_size(
        np.zeros((10, 20, 3), dtype=np.uint8),
        np.zeros((10, 20), dtype=np.uint8),
    )


def test_different_dimensions_are_not_the_same_frame():
    assert not same_frame_size(
        np.zeros((10, 20), dtype=np.uint8),
        np.zeros((10, 40), dtype=np.uint8),
    )


def test_a_non_array_is_never_the_same_frame():
    assert not same_frame_size(np.zeros((10, 10)), "not an image")
