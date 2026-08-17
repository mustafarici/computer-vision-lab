"""
Noise-reduction filters that aren't Gaussian blur.

The Gaussian blur in basic_ops.py smooths everything equally, edges
included. These two don't:

- Median replaces each pixel with the median of its neighbourhood, so
  salt-and-pepper speckles vanish outright rather than being smeared
  into grey smudges.
- Bilateral averages only over neighbours that are BOTH nearby in space
  and similar in intensity, so flat regions get smoothed while edges
  survive. That's why it's the usual choice for skin/texture cleanup.

Both operate on the image as uploaded (colour when available), since
that's where their difference is easiest to see.
"""

import cv2
import numpy as np
import streamlit as st

MAX_CACHE_ENTRIES = 6


def _as_processable(image_np: np.ndarray) -> np.ndarray:
    """Drop an alpha channel if present; leave grayscale alone."""

    if image_np.ndim == 3:
        return image_np[:, :, :3]

    return image_np


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def apply_median_filter(image_np: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Median blur. kernel_size must be odd; OpenCV also requires it to be
    <= 5 for anything other than 8-bit input, which is all we pass.
    """

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be an odd positive number, got {kernel_size}."
        )

    return cv2.medianBlur(_as_processable(image_np), kernel_size)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def apply_bilateral_filter(
    image_np: np.ndarray,
    diameter: int,
    sigma_color: int,
    sigma_space: int,
) -> np.ndarray:
    """
    Edge-preserving smoothing.

    sigma_color controls how different two intensities may be and still
    be averaged together; sigma_space controls how far apart they may
    be. Larger diameter means a wider neighbourhood and a much slower
    filter, which is why this one is cached.
    """

    return cv2.bilateralFilter(
        _as_processable(image_np),
        diameter,
        sigma_color,
        sigma_space,
    )
