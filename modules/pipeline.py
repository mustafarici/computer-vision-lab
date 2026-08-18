"""
User-composed processing chains.

Every other stage in this app starts from the original image: pick
"Canny Edge Detection" and you get Canny applied to the uploaded photo,
with a fixed blur in front of it. That is the right shape for learning
one operation at a time, and the wrong shape for the moment you want to
ask "what happens if I equalize the contrast first, *then* threshold,
*then* close the gaps?" — which is what real pipelines are made of.

This module supplies the operations that can be strung together and the
runner that applies them in order. The operations themselves are the
same functions the fixed stages call; nothing here re-implements image
processing.

Two rules keep any ordering the user picks from being an error:

- An operation that needs a single channel gets a grayscale conversion
  inserted in front of it automatically.
- An operation that needs a binary mask (the morphology operations,
  contour detection) gets a threshold inserted in front of it, unless
  what arrived is already binary.

Both insertions are recorded in the trace the runner returns, so the
chain that actually ran is shown to the user rather than guessed at.
"""

from collections.abc import Callable
from typing import NamedTuple

import cv2
import numpy as np

from modules.basic_ops import (
    apply_gaussian_blur,
    apply_threshold,
    convert_to_grayscale,
)
from modules.contours import apply_contour_detection
from modules.edges import apply_canny, apply_laplacian, apply_sobel
from modules.enhancement import apply_clahe, apply_histogram_equalization
from modules.filters import apply_bilateral_filter, apply_median_filter
from modules.morphology import (
    apply_closing,
    apply_dilation,
    apply_erosion,
    apply_opening,
)
from modules.thresholding import apply_adaptive_threshold, apply_otsu_threshold


class Operation(NamedTuple):
    """One step that can appear in a user-composed chain."""

    apply: Callable[[np.ndarray, dict], np.ndarray]
    # Shown in the trace under the result.
    summary: str
    # The operation only accepts one channel (cv2.equalizeHist and
    # cv2.Canny reject a 3-channel array outright), so the runner
    # converts to grayscale first when needed.
    single_channel: bool = False
    # The operation is defined on a binary mask, so the runner
    # thresholds first when what arrived isn't already one.
    binary_input: bool = False


def _is_binary(image_np: np.ndarray) -> bool:
    """True if every pixel is already either 0 or 255."""

    return bool(((image_np == 0) | (image_np == 255)).all())


def _make_morphology_step(morph_fn):
    def step(image_np, params):
        return morph_fn(
            image_np,
            params["morph_kernel_size"],
            params["morph_iterations"],
        )

    return step


def _contours_step(image_np, params):
    # apply_contour_detection draws onto a grayscale canvas; at this
    # point in a chain the mask itself is the most honest backdrop,
    # since anything earlier in the chain has already been folded in.
    result, _count = apply_contour_detection(
        image_np,
        image_np,
        params["contour_retrieval_mode"],
        params["contour_approx_method"],
        params["min_contour_area"],
    )

    return result


# Insertion order is the order the steps are offered in the sidebar,
# so it runs roughly from "cleaning up the image" through to
# "extracting something from it".
OPERATIONS: dict[str, Operation] = {
    "Grayscale": Operation(
        apply=lambda image, params: convert_to_grayscale(image),
        summary="collapse colour to one intensity channel",
    ),
    "Gaussian Blur": Operation(
        apply=lambda image, params: apply_gaussian_blur(
            image, params["kernel_size"]
        ),
        summary="smooth with a Gaussian kernel",
    ),
    "Median Filter": Operation(
        apply=lambda image, params: apply_median_filter(
            image, params["median_kernel_size"]
        ),
        summary="remove speckle noise",
    ),
    "Bilateral Filter": Operation(
        apply=lambda image, params: apply_bilateral_filter(
            image,
            params["bilateral_diameter"],
            params["bilateral_sigma_color"],
            params["bilateral_sigma_space"],
        ),
        summary="smooth without softening edges",
    ),
    "Histogram Equalization": Operation(
        apply=lambda image, params: apply_histogram_equalization(image),
        summary="spread intensities across the full range",
        single_channel=True,
    ),
    "CLAHE": Operation(
        apply=lambda image, params: apply_clahe(
            image, params["clahe_clip_limit"], params["clahe_tile_size"]
        ),
        summary="raise local contrast, tile by tile",
        single_channel=True,
    ),
    "Invert": Operation(
        apply=lambda image, params: cv2.bitwise_not(image),
        summary="swap light and dark",
    ),
    "Fixed Threshold": Operation(
        apply=lambda image, params: apply_threshold(
            image, params["threshold_value"]
        ),
        summary="binarize at the chosen intensity",
        single_channel=True,
    ),
    "Otsu Threshold": Operation(
        apply=lambda image, params: apply_otsu_threshold(image)[0],
        summary="binarize at a threshold read off the histogram",
        single_channel=True,
    ),
    "Adaptive Threshold": Operation(
        apply=lambda image, params: apply_adaptive_threshold(
            image,
            params["adaptive_method"],
            params["adaptive_block_size"],
            params["adaptive_constant"],
        ),
        summary="binarize against each pixel's own neighbourhood",
        single_channel=True,
    ),
    "Canny Edges": Operation(
        apply=lambda image, params: apply_canny(
            image,
            min(params["lower_threshold"], params["upper_threshold"]),
            max(params["lower_threshold"], params["upper_threshold"]),
        ),
        summary="trace edges with hysteresis thresholding",
        single_channel=True,
    ),
    "Sobel Edges": Operation(
        apply=lambda image, params: apply_sobel(
            image, params["sobel_kernel_size"]
        ),
        summary="measure the intensity gradient",
        single_channel=True,
    ),
    "Laplacian Edges": Operation(
        apply=lambda image, params: apply_laplacian(
            image, params["laplacian_kernel_size"]
        ),
        summary="find rapid intensity change in every direction",
        single_channel=True,
    ),
    "Erosion": Operation(
        apply=_make_morphology_step(apply_erosion),
        summary="shrink the white regions",
        single_channel=True,
        binary_input=True,
    ),
    "Dilation": Operation(
        apply=_make_morphology_step(apply_dilation),
        summary="grow the white regions",
        single_channel=True,
        binary_input=True,
    ),
    "Opening": Operation(
        apply=_make_morphology_step(apply_opening),
        summary="erode then dilate, clearing small specks",
        single_channel=True,
        binary_input=True,
    ),
    "Closing": Operation(
        apply=_make_morphology_step(apply_closing),
        summary="dilate then erode, filling small holes",
        single_channel=True,
        binary_input=True,
    ),
    "Contour Detection": Operation(
        apply=_contours_step,
        summary="outline the connected white regions",
        single_channel=True,
        binary_input=True,
    ),
}


def run_pipeline(image_np: np.ndarray, steps: list, params: dict):
    """
    Apply `steps` to `image_np` in order.

    Returns (result, trace), where trace lists what actually ran —
    including any grayscale conversion or threshold the runner had to
    insert to make a step's input valid.

    Raises ValueError for a step name that isn't in OPERATIONS, rather
    than skipping it silently and returning a result that doesn't match
    what was asked for.
    """

    current = image_np
    trace: list = []

    for name in steps:
        if name not in OPERATIONS:
            raise ValueError(
                f"Unknown pipeline step '{name}'. "
                f"Must be one of: {list(OPERATIONS)}."
            )

        operation = OPERATIONS[name]

        if operation.single_channel and current.ndim == 3:
            current = convert_to_grayscale(current)
            trace.append("Grayscale *(inserted automatically)*")

        if operation.binary_input and not _is_binary(current):
            current = apply_threshold(current, params["threshold_value"])
            trace.append("Fixed Threshold *(inserted automatically)*")

        current = operation.apply(current, params)
        trace.append(name)

    return current, trace
