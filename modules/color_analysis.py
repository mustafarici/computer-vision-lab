"""RGB channel and HSV color space analysis."""

import cv2
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def build_rgb_channel_composite(image_np: np.ndarray):
    """
    Isolate the R, G and B channels into three color images
    (each channel shown in its own color, others zeroed out)
    and lay them side by side. Returns None for grayscale-only
    input, since there are no color channels to split.
    """

    if image_np.ndim == 2:
        return None

    rgb = image_np[:, :, :3]  # drop alpha if present
    r, g, b = cv2.split(rgb)
    zeros = np.zeros_like(r)

    red_only = cv2.merge([r, zeros, zeros])
    green_only = cv2.merge([zeros, g, zeros])
    blue_only = cv2.merge([zeros, zeros, b])

    return np.hstack([red_only, green_only, blue_only])


@st.cache_data(show_spinner=False)
def build_hsv_channel_composite(image_np: np.ndarray):
    """
    Convert to HSV and lay the H, S and V channels side by side.
    Hue is colorized (it directly represents color), while
    Saturation and Value are shown as grayscale intensity maps.
    Returns None for grayscale-only input.
    """

    if image_np.ndim == 2:
        return None

    rgb = image_np[:, :, :3]  # drop alpha if present
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    hue_colorized = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
    hue_colorized = cv2.cvtColor(hue_colorized, cv2.COLOR_BGR2RGB)

    saturation_vis = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    value_vis = cv2.cvtColor(v, cv2.COLOR_GRAY2RGB)

    return np.hstack([hue_colorized, saturation_vis, value_vis])
