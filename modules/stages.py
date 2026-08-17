"""
Stage registry for the Computer Vision Lab pipeline.

Each entry in STAGES bundles a stage's sidebar-facing metadata
(category, description, pipeline breakdown, whether it needs a color
image) together with the handler that actually computes its result.

Why this exists: previously app.py carried this same information
split across three places (a STAGES metadata dict, a giant
if/elif chain, and the sidebar layout). Adding or changing a stage
meant touching all three and keeping them in sync by hand. Now a new
stage is a single new entry here plus (if it needs new sliders) a
small addition to the sidebar in app.py — the metadata and the
handler that computes it live next to each other.

Handlers all share the same signature:

    handler(image_np, grayscale, params) -> StageResult

- image_np: the original (possibly RGB/RGBA/grayscale) uploaded image.
- grayscale: convert_to_grayscale(image_np), computed once by app.py
  before dispatch (cheap: cached by @st.cache_data).
- params: dict of the current sidebar slider/selectbox values, keyed
  by the names used below (see app.py's `params = {...}` block).
"""

from typing import Callable, NamedTuple, Optional

from modules.basic_ops import apply_gaussian_blur, apply_threshold
from modules.color_analysis import (
    build_hsv_channel_composite,
    build_rgb_channel_composite,
)
from modules.color_threshold import apply_color_threshold
from modules.contours import apply_contour_detection
from modules.edges import apply_canny, apply_laplacian, apply_sobel
from modules.face_detection import apply_face_detection
from modules.feature_detection import (
    apply_harris_corners,
    apply_orb_keypoints,
    compute_harris_response,
)
from modules.histogram import build_histogram_figure, compute_histogram
from modules.morphology import (
    apply_closing,
    apply_dilation,
    apply_erosion,
    apply_opening,
)
from modules.object_detection import apply_object_detection


class StageResult(NamedTuple):
    """What a stage handler returns: the image/figure to display plus
    an optional one-line caption shown underneath it."""

    image: object
    extra_info: Optional[str] = None


class Stage(NamedTuple):
    """Metadata + handler for a single pipeline stage."""

    category: str
    description: str
    pipeline: str
    requires_color: bool
    handler: Callable[[object, object, dict], StageResult]
    # True for stages whose result is a matplotlib figure (rendered
    # with st.pyplot) rather than an image array (rendered with
    # st.image). Only the histogram currently needs this.
    is_figure: bool = False
    # False for stages that work off the color image directly and
    # never touch the grayscale conversion, so app.py can skip
    # computing it for them entirely.
    needs_grayscale: bool = True


# ==================================================
# HANDLERS
# ==================================================
# One small function per stage. Each takes (image_np, grayscale,
# params) and returns a StageResult. Kept intentionally simple —
# these just call into the modules/* functions with the right
# arguments and phrase the caption.


def _original(image_np, grayscale, params):
    return StageResult(image_np)


def _grayscale(image_np, grayscale, params):
    return StageResult(grayscale)


def _binary(image_np, grayscale, params):
    return StageResult(apply_threshold(grayscale, params["threshold_value"]))


def _gaussian_blur(image_np, grayscale, params):
    return StageResult(apply_gaussian_blur(grayscale, params["kernel_size"]))


def _canny(image_np, grayscale, params):
    blurred = apply_gaussian_blur(grayscale, params["kernel_size"])

    # Guard clause: the sidebar already warns if lower > upper, but
    # cv2.Canny still needs well-ordered bounds to behave correctly.
    lower = min(params["lower_threshold"], params["upper_threshold"])
    upper = max(params["lower_threshold"], params["upper_threshold"])

    result = apply_canny(blurred, lower, upper)

    return StageResult(
        result,
        "Canny uses the Gaussian Blur settings defined above.",
    )


def _sobel(image_np, grayscale, params):
    return StageResult(apply_sobel(grayscale, params["sobel_kernel_size"]))


def _laplacian(image_np, grayscale, params):
    return StageResult(
        apply_laplacian(grayscale, params["laplacian_kernel_size"])
    )


def _make_morphology_handler(morph_fn):
    """Erosion / Dilation / Opening / Closing all share the same
    "threshold then apply a structuring-element op" shape — build
    their handler from the underlying modules.morphology function
    instead of writing four near-identical handlers."""

    def handler(image_np, grayscale, params):
        binary = apply_threshold(grayscale, params["threshold_value"])

        result = morph_fn(
            binary,
            params["morph_kernel_size"],
            params["morph_iterations"],
        )

        return StageResult(result)

    return handler


def _contours(image_np, grayscale, params):
    binary = apply_threshold(grayscale, params["threshold_value"])

    result, contour_count = apply_contour_detection(
        binary,
        grayscale,
        params["contour_retrieval_mode"],
        params["contour_approx_method"],
        params["min_contour_area"],
    )

    return StageResult(
        result,
        f"Found **{contour_count}** contours after the area filter.",
    )


def _rgb_channels(image_np, grayscale, params):
    return StageResult(
        build_rgb_channel_composite(image_np),
        "Left to right: **Red**, **Green**, **Blue** channel isolated.",
    )


def _hsv_channels(image_np, grayscale, params):
    return StageResult(
        build_hsv_channel_composite(image_np),
        "Left to right: **Hue** (colorized), **Saturation**, **Value**.",
    )


def _color_mask(image_np, grayscale, params):
    mask, _result = apply_color_threshold(
        image_np,
        params["hue_range"],
        params["saturation_range"],
        params["value_range"],
    )

    return StageResult(
        mask,
        "White pixels fall inside the selected HSV range.",
    )


def _color_threshold_result(image_np, grayscale, params):
    _mask, result = apply_color_threshold(
        image_np,
        params["hue_range"],
        params["saturation_range"],
        params["value_range"],
    )

    return StageResult(
        result,
        "Pixels outside the selected HSV range are masked out.",
    )


def _harris_corners(image_np, grayscale, params):
    harris_response = compute_harris_response(
        grayscale,
        params["harris_block_size"],
        params["harris_ksize"],
        params["harris_sensitivity"],
    )

    result = apply_harris_corners(
        grayscale,
        harris_response,
        params["harris_threshold"],
    )

    return StageResult(
        result,
        "Detected corners are marked by strong intensity changes "
        "in multiple directions.",
    )


def _orb_keypoints(image_np, grayscale, params):
    result, keypoint_count = apply_orb_keypoints(
        grayscale,
        params["orb_features"],
    )

    return StageResult(
        result,
        f"Found **{keypoint_count}** ORB keypoints "
        "(circle size ≈ feature scale).",
    )


def _face_detection(image_np, grayscale, params):
    result, face_count = apply_face_detection(
        image_np,
        grayscale,
        params["face_scale_factor"],
        params["face_min_neighbors"],
        params["face_min_size"],
    )

    return StageResult(
        result,
        f"Found **{face_count}** face(s). "
        "Haar Cascade works best on frontal, well-lit faces.",
    )


def _object_detection(image_np, grayscale, params):
    object_type = params["object_type"]

    result, object_count = apply_object_detection(
        image_np,
        grayscale,
        object_type,
        params["object_scale_factor"],
        params["object_min_neighbors"],
        params["object_min_size"],
    )

    return StageResult(
        result,
        f"Found **{object_count}** instance(s) of **{object_type}**. "
        "Not every image will contain this object class.",
    )


def _grayscale_histogram(image_np, grayscale, params):
    histogram = compute_histogram(grayscale)
    return StageResult(build_histogram_figure(histogram))


# ==================================================
# STAGE REGISTRY
# ==================================================

STAGES: dict[str, Stage] = {
    "Original Image": Stage(
        category="Basic Processing",
        description=(
            "Displays the original uploaded image without "
            "any image processing."
        ),
        pipeline="Input image",
        requires_color=False,
        handler=_original,
        needs_grayscale=False,
    ),
    "Grayscale Image": Stage(
        category="Basic Processing",
        description=(
            "Converts the input image from RGB/BGR color "
            "representation to a single grayscale channel."
        ),
        pipeline="Original → Grayscale",
        requires_color=False,
        handler=_grayscale,
    ),
    "Binary Image": Stage(
        category="Basic Processing",
        description=(
            "Converts the grayscale image into a binary "
            "black-and-white image using a threshold value."
        ),
        pipeline="Original → Grayscale → Threshold",
        requires_color=False,
        handler=_binary,
    ),
    "Gaussian Blur": Stage(
        category="Basic Processing",
        description=(
            "Applies Gaussian smoothing to reduce image noise "
            "and high-frequency details."
        ),
        pipeline="Original → Grayscale → Gaussian Blur",
        requires_color=False,
        handler=_gaussian_blur,
    ),
    "Canny Edge Detection": Stage(
        category="Edge Detection",
        description=(
            "Detects edges using the Canny edge detection "
            "algorithm after Gaussian smoothing."
        ),
        pipeline="Original → Grayscale → Gaussian Blur → Canny",
        requires_color=False,
        handler=_canny,
    ),
    "Sobel Edge Detection": Stage(
        category="Edge Detection",
        description=(
            "Computes image intensity gradients using the "
            "Sobel operator."
        ),
        pipeline="Original → Grayscale → Sobel",
        requires_color=False,
        handler=_sobel,
    ),
    "Laplacian Edge Detection": Stage(
        category="Edge Detection",
        description=(
            "Detects regions of rapid intensity change using "
            "the Laplacian operator."
        ),
        pipeline="Original → Grayscale → Laplacian",
        requires_color=False,
        handler=_laplacian,
    ),
    "Erosion": Stage(
        category="Morphology",
        description=(
            "Shrinks foreground regions using a morphological "
            "erosion operation."
        ),
        pipeline="Original → Grayscale → Threshold → Erosion",
        requires_color=False,
        handler=_make_morphology_handler(apply_erosion),
    ),
    "Dilation": Stage(
        category="Morphology",
        description=(
            "Expands foreground regions using a morphological "
            "dilation operation."
        ),
        pipeline="Original → Grayscale → Threshold → Dilation",
        requires_color=False,
        handler=_make_morphology_handler(apply_dilation),
    ),
    "Opening": Stage(
        category="Morphology",
        description=(
            "Applies erosion followed by dilation to remove "
            "small foreground noise."
        ),
        pipeline="Original → Grayscale → Threshold → Opening",
        requires_color=False,
        handler=_make_morphology_handler(apply_opening),
    ),
    "Closing": Stage(
        category="Morphology",
        description=(
            "Applies dilation followed by erosion to close "
            "small holes and gaps."
        ),
        pipeline="Original → Grayscale → Threshold → Closing",
        requires_color=False,
        handler=_make_morphology_handler(apply_closing),
    ),
    "Contour Detection": Stage(
        category="Contours",
        description=(
            "Detects object contours from the thresholded "
            "binary image."
        ),
        pipeline="Original → Grayscale → Threshold → Contours",
        requires_color=False,
        handler=_contours,
    ),
    "RGB Channels": Stage(
        category="Color Analysis",
        description=(
            "Separates the input image into Red, Green, "
            "and Blue channel representations."
        ),
        pipeline="Original → RGB Channel Separation",
        requires_color=True,
        handler=_rgb_channels,
        needs_grayscale=False,
    ),
    "HSV Channels": Stage(
        category="Color Analysis",
        description=(
            "Separates the image into Hue, Saturation, "
            "and Value components."
        ),
        pipeline="Original → HSV Conversion",
        requires_color=True,
        handler=_hsv_channels,
        needs_grayscale=False,
    ),
    "Color Mask": Stage(
        category="Color Analysis",
        description=(
            "Creates a binary mask containing pixels that "
            "fall inside the selected HSV range."
        ),
        pipeline="Original → HSV → Color Mask",
        requires_color=True,
        handler=_color_mask,
        needs_grayscale=False,
    ),
    "Color Threshold Result": Stage(
        category="Color Analysis",
        description=(
            "Keeps pixels inside the selected HSV range "
            "and masks out the remaining regions."
        ),
        pipeline="Original → HSV → Color Thresholding",
        requires_color=True,
        handler=_color_threshold_result,
        needs_grayscale=False,
    ),
    "Harris Corners": Stage(
        category="Feature Detection",
        description=(
            "Detects corners using the Harris corner "
            "detection algorithm."
        ),
        pipeline="Original → Grayscale → Harris Response → Corners",
        requires_color=False,
        handler=_harris_corners,
    ),
    "ORB Keypoints": Stage(
        category="Feature Detection",
        description=(
            "Detects local image features using the "
            "ORB feature detector."
        ),
        pipeline="Original → Grayscale → ORB",
        requires_color=False,
        handler=_orb_keypoints,
    ),
    "Face Detection": Stage(
        category="Object Detection",
        description=(
            "Detects faces using OpenCV's built-in Haar "
            "Cascade classifier and marks them with boxes."
        ),
        pipeline=(
            "Original → Grayscale → Haar Cascade → Bounding Boxes"
        ),
        requires_color=False,
        handler=_face_detection,
    ),
    "Object Detection": Stage(
        category="Object Detection",
        description=(
            "Detects instances of a selected object class "
            "(eyes, smiles, full bodies, cat faces, license "
            "plates) using OpenCV's bundled Haar Cascade "
            "classifiers."
        ),
        pipeline=(
            "Original → Grayscale → Haar Cascade (selected class) "
            "→ Bounding Boxes"
        ),
        requires_color=False,
        handler=_object_detection,
    ),
    "Grayscale Histogram": Stage(
        category="Histogram Analysis",
        description=(
            "Displays the distribution of grayscale intensity "
            "values in the image."
        ),
        pipeline="Original → Grayscale → Histogram",
        requires_color=False,
        handler=_grayscale_histogram,
        is_figure=True,
    ),
}
