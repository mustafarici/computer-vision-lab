"""Grayscale conversion, thresholding and blurring."""

import cv2
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def convert_to_grayscale(image_np: np.ndarray) -> np.ndarray:
    """Convert an L / RGB / RGBA image to single-channel grayscale."""

    if image_np.ndim == 2:
        return image_np

    channels = image_np.shape[2]

    if channels == 4:
        return cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)

    if channels == 3:
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Fallback for unexpected channel counts.
    return image_np[:, :, 0]


@st.cache_data(show_spinner=False)
def apply_threshold(grayscale: np.ndarray, threshold_value: int) -> np.ndarray:
    _, binary = cv2.threshold(
        grayscale,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )
    return binary


@st.cache_data(show_spinner=False)
def apply_gaussian_blur(grayscale: np.ndarray, kernel_size: int) -> np.ndarray:
    return cv2.GaussianBlur(
        grayscale,
        (kernel_size, kernel_size),
        0
    )
