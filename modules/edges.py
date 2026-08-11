"""Canny, Sobel and Laplacian edge detection."""

import cv2
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def apply_canny(blurred: np.ndarray, lower: int, upper: int) -> np.ndarray:
    return cv2.Canny(blurred, lower, upper)


@st.cache_data(show_spinner=False)
def apply_sobel(grayscale: np.ndarray, ksize: int) -> np.ndarray:
    """Combined Sobel gradient magnitude (x and y), normalized to uint8."""

    sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = cv2.magnitude(sobel_x, sobel_y)

    return cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)


@st.cache_data(show_spinner=False)
def apply_laplacian(grayscale: np.ndarray, ksize: int) -> np.ndarray:
    """Laplacian edge response, absolute value and normalized to uint8."""

    laplacian = cv2.Laplacian(grayscale, cv2.CV_64F, ksize=ksize)

    return cv2.convertScaleAbs(laplacian)
