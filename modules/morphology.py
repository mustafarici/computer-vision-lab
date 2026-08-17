"""Erosion, dilation, opening and closing.

Not cached: each of these runs in well under 1 ms on a 1920x1080
image, which is faster than the ~1.5-2 ms @st.cache_data spends
hashing the input array and copying the result back out. See the note
at the top of modules/basic_ops.py.
"""

import cv2
import numpy as np


def apply_erosion(binary: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(binary, kernel, iterations=iterations)


def apply_dilation(binary: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(binary, kernel, iterations=iterations)


def apply_opening(binary: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """Erosion followed by dilation — removes small noise/specks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)


def apply_closing(binary: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """Dilation followed by erosion — closes small holes/gaps."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
