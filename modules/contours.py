"""Contour detection and visualization."""

import cv2
import numpy as np
import streamlit as st

RETRIEVAL_MODES = {
    "External": cv2.RETR_EXTERNAL,
    "List": cv2.RETR_LIST,
    "Tree": cv2.RETR_TREE,
    "CComp": cv2.RETR_CCOMP,
}

APPROX_METHODS = {
    "Simple": cv2.CHAIN_APPROX_SIMPLE,
    "None": cv2.CHAIN_APPROX_NONE,
}


# Bounded so sweeping the "Minimum Contour Area" slider can't keep a
# full-size canvas alive for every position it passes through.
MAX_CACHE_ENTRIES = 8


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def apply_contour_detection(
    binary: np.ndarray,
    grayscale: np.ndarray,
    retrieval_mode: str,
    approx_method: str,
    min_area: int
):
    """
    Find contours on the binary image and draw them on a grayscale
    canvas. Contours are returned together with the drawn image so
    both stay in sync and only one value crosses the cache boundary.
    """

    contours, _ = cv2.findContours(
        binary,
        RETRIEVAL_MODES[retrieval_mode],
        APPROX_METHODS[approx_method]
    )

    filtered_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]

    canvas = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)

    cv2.drawContours(
        canvas,
        filtered_contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=2
    )

    return canvas, len(filtered_contours)
