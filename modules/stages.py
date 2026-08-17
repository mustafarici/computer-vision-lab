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
from modules.deep_detection import (
    MissingDependencyError,
    apply_mediapipe_landmarks,
    apply_yolo_detection,
    is_mediapipe_available,
    is_yolo_available,
)
from modules.edges import apply_canny, apply_laplacian, apply_sobel
from modules.enhancement import apply_clahe, apply_histogram_equalization
from modules.face_detection import apply_face_detection
from modules.feature_detection import (
    apply_harris_corners,
    apply_orb_keypoints,
    compute_harris_response,
)
from modules.filters import apply_bilateral_filter, apply_median_filter
from modules.histogram import (
    build_color_histogram_figure,
    build_histogram_figure,
    compute_color_histograms,
    compute_histogram,
)
from modules.hough import apply_hough_circles, apply_hough_lines
from modules.morphology import (
    apply_closing,
    apply_dilation,
    apply_erosion,
    apply_opening,
)
from modules.object_detection import apply_object_detection
from modules.thresholding import apply_adaptive_threshold, apply_otsu_threshold


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
    # True for the stage that reports mean/median/min/max under its
    # result. Only the grayscale histogram does.
    shows_statistics: bool = False


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


def _otsu_threshold(image_np, grayscale, params):
    result, chosen = apply_otsu_threshold(grayscale)

    return StageResult(
        result,
        f"Otsu picked **{int(chosen)}** as the threshold, derived from "
        "the image histogram rather than set by hand.",
    )


def _adaptive_threshold(image_np, grayscale, params):
    result = apply_adaptive_threshold(
        grayscale,
        params["adaptive_method"],
        params["adaptive_block_size"],
        params["adaptive_constant"],
    )

    return StageResult(
        result,
        "Each pixel is compared against its own neighbourhood, so "
        "uneven lighting doesn't wash out whole regions.",
    )


def _median_filter(image_np, grayscale, params):
    return StageResult(
        apply_median_filter(image_np, params["median_kernel_size"]),
        "Median filtering removes salt-and-pepper noise outright "
        "instead of smearing it, unlike a Gaussian blur.",
    )


def _bilateral_filter(image_np, grayscale, params):
    return StageResult(
        apply_bilateral_filter(
            image_np,
            params["bilateral_diameter"],
            params["bilateral_sigma_color"],
            params["bilateral_sigma_space"],
        ),
        "Smooths flat regions while keeping edges sharp — compare it "
        "with Gaussian Blur on the same image.",
    )


def _histogram_equalization(image_np, grayscale, params):
    return StageResult(
        apply_histogram_equalization(grayscale),
        "Stretches the intensity histogram across the full 0–255 "
        "range, lifting detail out of a low-contrast image.",
    )


def _clahe(image_np, grayscale, params):
    return StageResult(
        apply_clahe(
            grayscale,
            params["clahe_clip_limit"],
            params["clahe_tile_size"],
        ),
        "Equalizes contrast per tile with a clip limit, so local "
        "detail improves without noise being over-amplified.",
    )


def _hough_lines(image_np, grayscale, params):
    result, count = apply_hough_lines(
        grayscale,
        min(params["lower_threshold"], params["upper_threshold"]),
        max(params["lower_threshold"], params["upper_threshold"]),
        params["hough_line_threshold"],
        params["hough_min_line_length"],
        params["hough_max_line_gap"],
    )

    return StageResult(
        result,
        f"Found **{count}** line segment(s). Uses the Canny thresholds "
        "from the Edge Detection section to find edges first.",
    )


def _hough_circles(image_np, grayscale, params):
    result, count = apply_hough_circles(
        grayscale,
        params["circle_min_distance"],
        params["circle_threshold"],
        params["circle_min_radius"],
        params["circle_max_radius"],
    )

    return StageResult(
        result,
        f"Found **{count}** circle(s). Circle detection is sensitive — "
        "expect to tune the radius range for your image.",
    )


def _grayscale_histogram(image_np, grayscale, params):
    histogram = compute_histogram(grayscale)
    return StageResult(build_histogram_figure(histogram))


def _color_histogram(image_np, grayscale, params):
    histograms = compute_color_histograms(image_np)

    return StageResult(
        build_color_histogram_figure(histograms),
        "Each curve is one colour channel — a strong colour cast "
        "shows up as one channel shifted away from the others.",
    )


# The two deep-learning stages degrade gracefully: if the optional
# package isn't installed, or the model can't be fetched, they return
# no image and explain what's missing instead of raising and taking the
# whole page down.

def _yolo_detection(image_np, grayscale, params):
    if not is_yolo_available():
        return StageResult(
            None,
            "**YOLO isn't installed.** Run `pip install -r "
            "requirements-ml.txt` to enable this stage. The rest of "
            "the app works without it.",
        )

    model_name = params["yolo_model"]

    try:
        result, count, summary = apply_yolo_detection(
            image_np,
            model_name,
            params["yolo_confidence"],
            params["yolo_iou"],
        )

    except (MissingDependencyError, RuntimeError) as error:
        return StageResult(None, str(error))

    if count == 0:
        detail = (
            "Nothing detected above the confidence threshold — "
            "try lowering it in the sidebar."
        )
    else:
        detail = f"Detected: {summary}."

    return StageResult(
        result,
        f"**{count}** object(s) found by {model_name}. {detail}",
    )


def _mediapipe_landmarks(image_np, grayscale, params):
    if not is_mediapipe_available():
        return StageResult(
            None,
            "**MediaPipe isn't installed.** Run `pip install -r "
            "requirements-ml.txt` to enable this stage. The rest of "
            "the app works without it.",
        )

    task_name = params["mediapipe_task"]

    try:
        result, count, summary = apply_mediapipe_landmarks(
            image_np, task_name
        )

    except (MissingDependencyError, RuntimeError) as error:
        return StageResult(None, str(error))

    if count == 0:
        detail = (
            "Nothing detected — MediaPipe needs the subject to be "
            "reasonably large and clearly visible in the frame."
        )
    else:
        detail = f"Found {summary}."

    return StageResult(result, f"**{task_name}.** {detail}")


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
    "Otsu Threshold": Stage(
        category="Thresholding",
        description=(
            "Binarizes the image using Otsu's method, which analyses "
            "the histogram and picks the threshold that best separates "
            "foreground from background — no slider needed."
        ),
        pipeline="Original → Grayscale → Otsu Threshold",
        requires_color=False,
        handler=_otsu_threshold,
    ),
    "Adaptive Threshold": Stage(
        category="Thresholding",
        description=(
            "Computes a separate threshold for every neighbourhood "
            "instead of one for the whole image, which is what makes "
            "unevenly lit photos and scanned documents readable."
        ),
        pipeline="Original → Grayscale → Adaptive Threshold",
        requires_color=False,
        handler=_adaptive_threshold,
    ),
    "Median Filter": Stage(
        category="Noise Filtering",
        description=(
            "Replaces each pixel with the median of its neighbourhood. "
            "The classic remedy for salt-and-pepper noise, which a "
            "Gaussian blur only smears around."
        ),
        pipeline="Original → Median Filter",
        requires_color=False,
        handler=_median_filter,
        needs_grayscale=False,
    ),
    "Bilateral Filter": Stage(
        category="Noise Filtering",
        description=(
            "Edge-preserving smoothing: averages only over neighbours "
            "that are both close by and similar in intensity, so flat "
            "areas are cleaned up while edges stay sharp."
        ),
        pipeline="Original → Bilateral Filter",
        requires_color=False,
        handler=_bilateral_filter,
        needs_grayscale=False,
    ),
    "Histogram Equalization": Stage(
        category="Enhancement",
        description=(
            "Redistributes intensities so the full range is used, "
            "which pulls detail out of flat, low-contrast images."
        ),
        pipeline="Original → Grayscale → Histogram Equalization",
        requires_color=False,
        handler=_histogram_equalization,
    ),
    "CLAHE": Stage(
        category="Enhancement",
        description=(
            "Contrast Limited Adaptive Histogram Equalization — "
            "equalizes each tile separately with a clip limit, giving "
            "local contrast without the noise amplification that "
            "global equalization causes."
        ),
        pipeline="Original → Grayscale → CLAHE",
        requires_color=False,
        handler=_clahe,
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
    "Hough Lines": Stage(
        category="Hough Transform",
        description=(
            "Finds straight line segments by letting every edge pixel "
            "vote for the lines that could pass through it, then "
            "keeping the peaks. The standard way to detect roads, "
            "document edges or building structure."
        ),
        pipeline="Original → Grayscale → Canny → Hough Lines",
        requires_color=False,
        handler=_hough_lines,
    ),
    "Hough Circles": Stage(
        category="Hough Transform",
        description=(
            "The same voting idea applied to circles, using gradient "
            "directions to keep the search tractable. Used for coins, "
            "pupils, dials and other round objects."
        ),
        pipeline=(
            "Original → Grayscale → Median Blur → Hough Circles"
        ),
        requires_color=False,
        handler=_hough_circles,
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
    "YOLO Object Detection": Stage(
        category="Deep Learning",
        description=(
            "Detects objects with a YOLOv8 neural network trained on "
            "the COCO dataset (80 classes: people, vehicles, animals, "
            "everyday items). Unlike the Haar Cascades above, it finds "
            "many classes at once and copes with angle, scale and "
            "lighting variation. Requires the optional ML dependencies."
        ),
        pipeline="Original → YOLOv8 (CNN) → Boxes + Class + Confidence",
        requires_color=False,
        handler=_yolo_detection,
        needs_grayscale=False,
    ),
    "MediaPipe Landmarks": Stage(
        category="Deep Learning",
        description=(
            "Detects fine-grained landmarks with Google's MediaPipe: a "
            "468-point face mesh, 21-point hand skeletons, or a "
            "33-point body pose. Where a bounding box says only where "
            "something is, landmarks describe its shape and posture. "
            "Requires the optional ML dependencies."
        ),
        pipeline="Original → MediaPipe Task → Landmarks + Connections",
        requires_color=False,
        handler=_mediapipe_landmarks,
        needs_grayscale=False,
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
        shows_statistics=True,
    ),
    "Color Histogram": Stage(
        category="Histogram Analysis",
        description=(
            "Plots the intensity distribution of the Red, Green and "
            "Blue channels together, which makes colour casts and "
            "clipped channels obvious at a glance."
        ),
        pipeline="Original → Per-Channel Histograms",
        requires_color=True,
        handler=_color_histogram,
        is_figure=True,
        needs_grayscale=False,
    ),
}
