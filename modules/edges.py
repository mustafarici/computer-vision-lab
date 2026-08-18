"""Canny, Sobel and Laplacian edge detection."""

import cv2
import numpy as np

from modules.caching import cache_data

# Caches are bounded so that sweeping a slider can't grow memory
# without limit — each distinct parameter combination would otherwise
# keep its own full-size result image alive forever. Streamlit evicts
# the least recently used entry once the limit is reached.
MAX_CACHE_ENTRIES = 8


@cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def apply_canny(blurred: np.ndarray, lower: int, upper: int) -> np.ndarray:
    """Cached: ~21 ms per run, comfortably above the ~2 ms cache overhead."""

    return cv2.Canny(blurred, lower, upper)


@cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def apply_sobel(grayscale: np.ndarray, ksize: int) -> np.ndarray:
    """Combined Sobel gradient magnitude (x and y), normalized to uint8.

    Cached: ~16 ms per run (two float64 passes plus a magnitude).
    """

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


def apply_laplacian(grayscale: np.ndarray, ksize: int) -> np.ndarray:
    """Laplacian edge response, absolute value and normalized to uint8.

    Not cached: ~2 ms per run, which is about what the cache lookup
    itself costs, so caching would only add memory pressure.
    """

    laplacian = cv2.Laplacian(grayscale, cv2.CV_64F, ksize=ksize)

    return cv2.convertScaleAbs(laplacian)
