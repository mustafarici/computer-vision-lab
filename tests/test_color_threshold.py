from modules.color_threshold import apply_color_threshold


def test_masks_red_patch(color_image):
    mask, result = apply_color_threshold(
        color_image,
        hue_range=(0, 10),
        saturation_range=(100, 255),
        value_range=(100, 255),
    )

    # The red patch at (40:55, 40:55) should fall inside the mask...
    assert mask[47, 47].max() > 0
    # ...while the black background should not.
    assert mask[0, 0].max() == 0
    assert result.shape == color_image.shape


def test_none_for_grayscale(grayscale_only_image):
    mask, result = apply_color_threshold(
        grayscale_only_image, (0, 179), (0, 255), (0, 255)
    )

    assert mask is None
    assert result is None
