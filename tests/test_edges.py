import numpy as np

from modules.edges import apply_canny, apply_laplacian, apply_sobel


def test_apply_canny_is_binary_edges(grayscale_image):
    edges = apply_canny(grayscale_image, 50, 150)

    assert edges.shape == grayscale_image.shape
    assert set(np.unique(edges)).issubset({0, 255})


def test_apply_sobel_shape_and_dtype(grayscale_image):
    sobel = apply_sobel(grayscale_image, 3)

    assert sobel.shape == grayscale_image.shape
    assert sobel.dtype == np.uint8


def test_apply_laplacian_shape_and_dtype(grayscale_image):
    laplacian = apply_laplacian(grayscale_image, 3)

    assert laplacian.shape == grayscale_image.shape
    assert laplacian.dtype == np.uint8
