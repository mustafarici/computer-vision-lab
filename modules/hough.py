"""
Hough transform: finding lines and circles.

The idea behind both is the same. Every edge pixel votes for all the
shapes that could pass through it, votes accumulate in a parameter
space, and the peaks are the shapes actually present. It's how you get
from "these pixels are edges" to "there is a line from here to there".
"""

import cv2
import numpy as np
import streamlit as st

MAX_CACHE_ENTRIES = 8


def _color_canvas(grayscale: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def apply_hough_lines(
    grayscale: np.ndarray,
    canny_lower: int,
    canny_upper: int,
    threshold: int,
    min_line_length: int,
    max_line_gap: int,
):
    """
    Detect straight line segments with the probabilistic Hough
    transform, drawn over the source image.

    Returns (image_with_lines, line_count).
    """

    edges = cv2.Canny(grayscale, canny_lower, canny_upper)

    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    canvas = _color_canvas(grayscale)

    if segments is None:
        return canvas, 0

    for segment in segments:
        x1, y1, x2, y2 = segment[0]
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2,
                 lineType=cv2.LINE_AA)

    return canvas, len(segments)


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def apply_hough_circles(
    grayscale: np.ndarray,
    min_distance: int,
    accumulator_threshold: int,
    min_radius: int,
    max_radius: int,
):
    """
    Detect circles with the Hough gradient method.

    Returns (image_with_circles, circle_count). The input is blurred
    first because HoughCircles is unusually sensitive to noise — it
    relies on gradient directions, which noise scrambles.
    """

    blurred = cv2.medianBlur(grayscale, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_distance,
        param1=100,  # upper Canny threshold used internally
        param2=accumulator_threshold,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    canvas = _color_canvas(grayscale)

    if circles is None:
        return canvas, 0

    circles = np.uint16(np.around(circles))

    for x, y, radius in circles[0, :]:
        cv2.circle(canvas, (int(x), int(y)), int(radius), (0, 255, 0), 2,
                   lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(x), int(y)), 2, (255, 0, 0), 3)

    return canvas, circles.shape[1]
