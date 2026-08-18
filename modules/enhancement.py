"""
Contrast enhancement: global histogram equalization and CLAHE.

Equalization stretches the intensity histogram so the full 0-255 range
gets used, which lifts detail out of a flat, low-contrast image. Its
weakness is that it works from one histogram for the whole image, so a
bright region can wash out while a dark one is over-amplified. CLAHE
does the same thing per tile and clips the histogram first, which keeps
local contrast without amplifying noise as aggressively.
"""

import cv2
import numpy as np

from modules.caching import cache_data

MAX_CACHE_ENTRIES = 8


def apply_histogram_equalization(grayscale: np.ndarray) -> np.ndarray:
    """Not cached: ~1 ms, cheaper than a cache lookup."""

    return cv2.equalizeHist(grayscale)


@cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def apply_clahe(
    grayscale: np.ndarray,
    clip_limit: float,
    tile_grid_size: int,
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.

    clip_limit caps how much any one histogram bin may contribute,
    which is what stops flat regions from turning into noise.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )

    return clahe.apply(grayscale)
