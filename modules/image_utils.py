"""
Small shape/representation helpers shared across modules.

Stages hand each other arrays that may be single-channel (grayscale,
binary), three-channel RGB, or four-channel RGBA. Anything that has to
combine two of them — the comparison view, the difference view, the
pipeline builder, video encoding — needs them in one common
representation first. Rather than each of those re-implementing the
same conversion, they call in here.
"""

import cv2
import numpy as np


def to_rgb(image_np: np.ndarray) -> np.ndarray:
    """Normalize any supported input to a 3-channel RGB array."""

    if image_np.ndim == 2:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    return image_np[:, :, :3]


def same_frame_size(first: np.ndarray, second: np.ndarray) -> bool:
    """
    True when two arrays cover the same pixel grid.

    Channel counts may differ — a grayscale result and its RGB original
    still describe the same frame. What disqualifies a pair is a
    different height or width, which is what the channel-composite
    stages produce (RGB Channels is three frames wide).
    """

    if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
        return False

    return first.shape[:2] == second.shape[:2]
