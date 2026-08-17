import numpy as np

from modules.feature_detection import (
    apply_harris_corners,
    apply_orb_keypoints,
    compute_harris_response,
)


def test_harris_response_shape(grayscale_image):
    response = compute_harris_response(grayscale_image, 2, 3, 0.04)

    assert response.shape == grayscale_image.shape


def test_harris_corners_marks_pixels_at_low_threshold(grayscale_image):
    response = compute_harris_response(grayscale_image, 2, 3, 0.04)

    canvas = apply_harris_corners(grayscale_image, response, 0.01)

    red_pixels = np.all(canvas == [255, 0, 0], axis=-1)
    assert red_pixels.any()


def test_harris_corners_none_at_impossible_threshold(grayscale_image):
    response = compute_harris_response(grayscale_image, 2, 3, 0.04)

    # threshold_ratio > 1 means nothing can exceed
    # threshold_ratio * response.max().
    canvas = apply_harris_corners(grayscale_image, response, 1.5)

    red_pixels = np.all(canvas == [255, 0, 0], axis=-1)
    assert not red_pixels.any()


def test_orb_keypoints_returns_count_and_canvas(grayscale_image):
    canvas, count = apply_orb_keypoints(grayscale_image, 300)

    assert isinstance(count, int)
    assert count >= 0
    assert canvas.shape == (*grayscale_image.shape, 3)
