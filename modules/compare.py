"""
Before/after comparison between the input image and a stage's result.

The app used to show the processed image on its own, which meant
judging what a stage actually did came down to remembering what the
photo looked like a click ago. Two views fix that:

- Side by side: the original and the result next to each other.
- Difference: |original - result| per pixel, which answers "what
  exactly changed, and where" far more precisely than two images the
  eye has to shuttle between.

Not every stage can be compared. The channel-composite stages (RGB
Channels, HSV Channels) return an image three frames wide, and the
histogram stages return a matplotlib figure rather than an array —
`is_comparable` is what the app checks before offering the option.
"""

import cv2
import numpy as np

from modules.caching import cache_data
from modules.image_utils import same_frame_size, to_rgb

# Difference images are cheap to compute (~1 ms) but this is called on
# every rerun while a slider is being dragged, and the result is a
# full-size array. A small bounded cache keeps the common case — the
# user toggling between views without changing parameters — free.
MAX_CACHE_ENTRIES = 4


def is_comparable(original, result) -> bool:
    """
    True when `result` can be placed next to `original` as a
    before/after pair: both are arrays covering the same pixel grid.
    """

    return same_frame_size(original, result)


@cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def compute_difference(original: np.ndarray, result: np.ndarray) -> np.ndarray:
    """
    Absolute per-pixel difference between the original and the result.

    Both sides are lifted to RGB first so a grayscale or binary result
    can still be differenced against a color original. Black means
    "this stage left this pixel alone".
    """

    if not same_frame_size(original, result):
        raise ValueError(
            "Difference needs two images of the same height and width, "
            f"got {original.shape[:2]} and {result.shape[:2]}."
        )

    return cv2.absdiff(to_rgb(original), to_rgb(result))


def summarize_difference(difference: np.ndarray) -> str:
    """
    One line describing how much of the image the stage changed.

    Reported as the share of pixels that moved at all, plus the mean
    magnitude of the change, because those two say different things: a
    contrast tweak nudges every pixel a little, while a threshold
    rewrites most of them completely.
    """

    if difference.size == 0:
        return "Nothing to compare."

    per_pixel = difference.max(axis=2) if difference.ndim == 3 else difference

    changed_share = float((per_pixel > 0).mean())
    mean_change = float(per_pixel.mean())

    if changed_share == 0.0:
        return "This stage left the image completely unchanged."

    return (
        f"**{changed_share:.1%}** of pixels changed, by "
        f"**{mean_change:.1f}** intensity levels on average "
        "(brighter means a larger change)."
    )
