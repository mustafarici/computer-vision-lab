import pytest

from modules.face_detection import apply_face_detection


def test_invalid_scale_factor_raises(grayscale_image, color_image):
    with pytest.raises(ValueError):
        apply_face_detection(color_image, grayscale_image, 1.0, 5, 30)


def test_invalid_min_neighbors_raises(grayscale_image, color_image):
    with pytest.raises(ValueError):
        apply_face_detection(color_image, grayscale_image, 1.1, -1, 30)


def test_invalid_min_size_raises(grayscale_image, color_image):
    with pytest.raises(ValueError):
        apply_face_detection(color_image, grayscale_image, 1.1, 5, 0)


def test_non_2d_grayscale_raises(color_image):
    with pytest.raises(ValueError):
        apply_face_detection(color_image, color_image, 1.1, 5, 30)


def test_runs_on_synthetic_image_without_faces(grayscale_image, color_image):
    canvas, count = apply_face_detection(
        color_image, grayscale_image, 1.1, 5, 30
    )

    assert canvas.shape == color_image.shape
    assert count == 0


def test_return_boxes_flag(grayscale_image, color_image):
    canvas, count, boxes = apply_face_detection(
        color_image, grayscale_image, 1.1, 5, 30, return_boxes=True
    )

    assert isinstance(boxes, list)
    assert count == len(boxes)
