import pytest

from modules.object_detection import OBJECT_CASCADES, apply_object_detection


def test_unknown_object_type_raises(grayscale_image, color_image):
    with pytest.raises(ValueError):
        apply_object_detection(
            color_image, grayscale_image, "Unicorn", 1.1, 5, 20
        )


def test_invalid_scale_factor_raises(grayscale_image, color_image):
    with pytest.raises(ValueError):
        apply_object_detection(
            color_image, grayscale_image, "Eyes", 1.0, 5, 20
        )


def test_runs_for_every_bundled_cascade(grayscale_image, color_image):
    for object_type in OBJECT_CASCADES:
        canvas, count = apply_object_detection(
            color_image, grayscale_image, object_type, 1.1, 5, 20
        )

        assert canvas.shape == color_image.shape
        assert count >= 0
