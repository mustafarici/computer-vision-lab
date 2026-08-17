"""
Automatic and locally adaptive thresholding.

Where the fixed threshold in basic_ops.py makes you pick a number,
these two pick it for you: Otsu by choosing the value that best
separates the image's two intensity clusters, and adaptive
thresholding by computing a different threshold for every
neighbourhood — which is what rescues images with uneven lighting,
where no single global value works anywhere.
"""

import cv2
import numpy as np

ADAPTIVE_METHODS = {
    "Mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    "Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}


def apply_otsu_threshold(grayscale: np.ndarray):
    """
    Binarize using Otsu's method, which derives the threshold from the
    image histogram instead of taking one as input.

    Returns (binary, chosen_threshold) so the UI can report the value
    Otsu settled on — that number is the interesting part.
    """

    chosen, binary = cv2.threshold(
        grayscale,
        0,  # ignored when THRESH_OTSU is set
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return binary, chosen


def apply_adaptive_threshold(
    grayscale: np.ndarray,
    method: str,
    block_size: int,
    constant: int,
) -> np.ndarray:
    """
    Threshold each pixel against a statistic of its own neighbourhood.

    block_size must be odd and >= 3; the sliders enforce that, but so
    does this, because OpenCV raises a bare assertion otherwise.
    """

    if method not in ADAPTIVE_METHODS:
        raise ValueError(
            f"Unknown adaptive method '{method}'. "
            f"Must be one of: {list(ADAPTIVE_METHODS)}."
        )

    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(
            f"block_size must be an odd number >= 3, got {block_size}."
        )

    return cv2.adaptiveThreshold(
        grayscale,
        255,
        ADAPTIVE_METHODS[method],
        cv2.THRESH_BINARY,
        block_size,
        constant,
    )
