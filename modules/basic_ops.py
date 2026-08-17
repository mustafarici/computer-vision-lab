"""Grayscale conversion, thresholding and blurring.

Deliberately NOT cached. @st.cache_data has to hash the full input
array (~6 MB for a 1920x1080 image) and deep-copy the output on every
call, which measures at ~1.5-2 ms — more than these operations cost to
just recompute (0.1-0.6 ms each). Caching them was a net slowdown, and
it also grew unboundedly: sweeping the 0-255 threshold slider cached a
separate full-size image per position (~500 MB).

See modules/edges.py for the operations where caching does pay off.
"""

import cv2
import numpy as np


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


def apply_threshold(grayscale: np.ndarray, threshold_value: int) -> np.ndarray:
    _, binary = cv2.threshold(
        grayscale,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )
    return binary


def apply_gaussian_blur(grayscale: np.ndarray, kernel_size: int) -> np.ndarray:
    return cv2.GaussianBlur(
        grayscale,
        (kernel_size, kernel_size),
        0
    )
