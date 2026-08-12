"""HSV range-based color thresholding."""

import cv2
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def apply_color_threshold(
    image_np: np.ndarray,
    hue_range: tuple,
    saturation_range: tuple,
    value_range: tuple
):
    """
    Build an HSV in-range mask and apply it to the original image.

    Returns (mask_as_rgb, masked_result). Both are None for
    grayscale-only input, since there is no color/hue information
    to threshold on.
    """

    if image_np.ndim == 2:
        return None, None

    rgb = image_np[:, :, :3]  # drop alpha if present
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower_bound = np.array(
        [hue_range[0], saturation_range[0], value_range[0]]
    )
    upper_bound = np.array(
        [hue_range[1], saturation_range[1], value_range[1]]
    )

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    masked_result = cv2.bitwise_and(rgb, rgb, mask=mask)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    return mask_rgb, masked_result
