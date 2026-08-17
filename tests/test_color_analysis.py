from modules.color_analysis import (
    build_hsv_channel_composite,
    build_rgb_channel_composite,
)


def test_rgb_composite_shape(color_image):
    composite = build_rgb_channel_composite(color_image)

    height, width = color_image.shape[:2]
    assert composite.shape == (height, width * 3, 3)


def test_rgb_composite_none_for_grayscale(grayscale_only_image):
    assert build_rgb_channel_composite(grayscale_only_image) is None


def test_hsv_composite_shape(color_image):
    composite = build_hsv_channel_composite(color_image)

    height, width = color_image.shape[:2]
    assert composite.shape == (height, width * 3, 3)


def test_hsv_composite_none_for_grayscale(grayscale_only_image):
    assert build_hsv_channel_composite(grayscale_only_image) is None
