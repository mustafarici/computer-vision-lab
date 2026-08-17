"""
Declarative definition of the sidebar controls.

Why this exists: the sidebar used to be ~440 lines of hand-written
st.slider / st.selectbox calls, and every parameter had to be spelled
out three times — once as a widget, once when packing the `params`
dict, and once when a stage handler read it back. Adding a parameter
meant three edits in two files, and the widget's key, its dict key and
the handler's lookup key had to be kept identical by hand.

Here a parameter is one Control entry. render_sidebar() walks the
sections, draws the right widget for each control, and returns the
params dict that modules/stages.py handlers read — so the key can only
ever be written once.

Sections also declare which stage categories they're relevant to, which
lets the sidebar open the section belonging to whatever stage is on
screen and collapse the rest.
"""

from typing import Any, Callable, NamedTuple, Optional

import streamlit as st

from modules.deep_detection import MEDIAPIPE_TASKS, YOLO_MODELS
from modules.object_detection import OBJECT_CASCADES


class Control(NamedTuple):
    """One sidebar widget, and the params key its value lands under."""

    key: str
    label: str
    kind: str  # "slider" | "range" | "select"
    default: Any
    help: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[tuple] = None
    # Optional heading rendered above this control, used to group
    # several related controls inside one section.
    subheader: str = ""


class Section(NamedTuple):
    """A collapsible group of controls in the sidebar."""

    title: str
    controls: tuple
    # Stage categories (as used in modules/stages.py) whose stages are
    # actually affected by these controls. Drives which section opens.
    categories: frozenset
    caption: str = ""
    # Optional validation run after the section renders; returns a
    # warning string, or None when the values are fine.
    warning: Optional[Callable[[dict], Optional[str]]] = None


def _canny_threshold_warning(params: dict) -> Optional[str]:
    if params["lower_threshold"] > params["upper_threshold"]:
        return "Lower threshold is higher than upper threshold."
    return None


SECTIONS: tuple = (
    Section(
        title="🔧 Basic Processing",
        # The threshold feeds morphology and contours, and the blur
        # feeds Canny, so this section stays relevant well beyond the
        # Basic Processing stages themselves.
        categories=frozenset(
            {"Basic Processing", "Edge Detection", "Morphology", "Contours"}
        ),
        controls=(
            Control(
                key="threshold_value",
                label="Threshold Value",
                kind="slider",
                default=127,
                min_value=0,
                max_value=255,
                subheader="Threshold",
                help=(
                    "Pixel intensity threshold used to convert "
                    "the grayscale image into a binary image."
                ),
            ),
            Control(
                key="kernel_size",
                label="Kernel Size",
                kind="slider",
                default=5,
                min_value=1,
                max_value=15,
                step=2,
                subheader="Gaussian Blur",
                help=(
                    "Controls the size of the Gaussian smoothing "
                    "kernel. Larger values produce stronger blur."
                ),
            ),
        ),
    ),
    Section(
        title="🎚️ Adaptive Thresholding",
        categories=frozenset({"Thresholding"}),
        caption="Otsu picks its own threshold and needs no settings.",
        controls=(
            Control(
                key="adaptive_method",
                label="Adaptive Method",
                kind="select",
                default="Gaussian",
                options=("Gaussian", "Mean"),
                help=(
                    "How each neighbourhood's threshold is computed: a "
                    "plain mean, or a Gaussian-weighted mean that "
                    "favours nearby pixels."
                ),
            ),
            Control(
                key="adaptive_block_size",
                label="Neighbourhood Size",
                kind="slider",
                default=11,
                min_value=3,
                max_value=51,
                step=2,
                help=(
                    "Side length of the neighbourhood used to compute "
                    "each local threshold. Must be odd."
                ),
            ),
            Control(
                key="adaptive_constant",
                label="Constant Subtracted (C)",
                kind="slider",
                default=2,
                min_value=-20,
                max_value=20,
                help=(
                    "Subtracted from the local mean. Raise it to make "
                    "the result darker and less noisy."
                ),
            ),
        ),
    ),
    Section(
        title="🧹 Noise Filtering",
        categories=frozenset({"Noise Filtering"}),
        controls=(
            Control(
                key="median_kernel_size",
                label="Median Kernel Size",
                kind="slider",
                default=5,
                min_value=1,
                max_value=15,
                step=2,
                subheader="Median",
                help=(
                    "Neighbourhood size the median is taken over. "
                    "Larger values remove bigger speckles."
                ),
            ),
            Control(
                key="bilateral_diameter",
                label="Diameter",
                kind="slider",
                default=9,
                min_value=3,
                max_value=25,
                step=2,
                subheader="Bilateral",
                help=(
                    "Neighbourhood diameter in pixels. Larger is "
                    "smoother and considerably slower."
                ),
            ),
            Control(
                key="bilateral_sigma_color",
                label="Sigma Color",
                kind="slider",
                default=75,
                min_value=10,
                max_value=200,
                step=5,
                help=(
                    "How different two intensities may be and still be "
                    "averaged together. Higher blurs across edges."
                ),
            ),
            Control(
                key="bilateral_sigma_space",
                label="Sigma Space",
                kind="slider",
                default=75,
                min_value=10,
                max_value=200,
                step=5,
                help=(
                    "How far apart two pixels may be and still "
                    "influence each other."
                ),
            ),
        ),
    ),
    Section(
        title="✨ Contrast Enhancement",
        categories=frozenset({"Enhancement"}),
        caption=(
            "Global histogram equalization takes no parameters; these "
            "control CLAHE."
        ),
        controls=(
            Control(
                key="clahe_clip_limit",
                label="Clip Limit",
                kind="slider",
                default=2.0,
                min_value=1.0,
                max_value=10.0,
                step=0.5,
                help=(
                    "Caps how much contrast any one tile may gain. "
                    "Lower values suppress noise amplification."
                ),
            ),
            Control(
                key="clahe_tile_size",
                label="Tile Grid Size",
                kind="slider",
                default=8,
                min_value=2,
                max_value=16,
                step=2,
                help=(
                    "The image is split into this many tiles per side, "
                    "each equalized independently."
                ),
            ),
        ),
    ),
    Section(
        title="📏 Hough Transform",
        categories=frozenset({"Hough Transform"}),
        caption=(
            "Line detection also uses the Canny thresholds from the "
            "Edge Detection section."
        ),
        controls=(
            Control(
                key="hough_line_threshold",
                label="Vote Threshold",
                kind="slider",
                default=100,
                min_value=10,
                max_value=300,
                step=10,
                subheader="Lines",
                help=(
                    "How many votes a line needs to be accepted. "
                    "Higher means fewer, more confident lines."
                ),
            ),
            Control(
                key="hough_min_line_length",
                label="Minimum Line Length",
                kind="slider",
                default=50,
                min_value=0,
                max_value=300,
                step=10,
                help="Shorter segments than this are discarded.",
            ),
            Control(
                key="hough_max_line_gap",
                label="Maximum Line Gap",
                kind="slider",
                default=10,
                min_value=0,
                max_value=100,
                step=5,
                help=(
                    "Largest gap between two segments that still "
                    "counts as one continuous line."
                ),
            ),
            Control(
                key="circle_min_distance",
                label="Minimum Center Distance",
                kind="slider",
                default=50,
                min_value=10,
                max_value=300,
                step=10,
                subheader="Circles",
                help=(
                    "How far apart two circle centers must be. Too "
                    "small and one circle is detected many times."
                ),
            ),
            Control(
                key="circle_threshold",
                label="Accumulator Threshold",
                kind="slider",
                default=30,
                min_value=10,
                max_value=150,
                step=5,
                help=(
                    "Lower values detect more circles, including false "
                    "ones; higher values only obvious circles."
                ),
            ),
            Control(
                key="circle_min_radius",
                label="Minimum Radius",
                kind="slider",
                default=0,
                min_value=0,
                max_value=200,
                step=5,
                help="Smallest circle radius to look for, in pixels.",
            ),
            Control(
                key="circle_max_radius",
                label="Maximum Radius",
                kind="slider",
                default=100,
                min_value=0,
                max_value=500,
                step=10,
                help=(
                    "Largest circle radius to look for. 0 means no "
                    "upper limit."
                ),
            ),
        ),
    ),
    Section(
        title="📐 Edge Detection",
        categories=frozenset({"Edge Detection", "Hough Transform"}),
        warning=_canny_threshold_warning,
        controls=(
            Control(
                key="lower_threshold",
                label="Lower Threshold",
                kind="slider",
                default=50,
                min_value=0,
                max_value=255,
                subheader="Canny",
                help=(
                    "Lower boundary used by the Canny edge detector "
                    "to identify weak edges."
                ),
            ),
            Control(
                key="upper_threshold",
                label="Upper Threshold",
                kind="slider",
                default=150,
                min_value=0,
                max_value=255,
                help=(
                    "Upper boundary used by the Canny edge detector "
                    "to identify strong edges."
                ),
            ),
            Control(
                key="sobel_kernel_size",
                label="Sobel Kernel Size",
                kind="slider",
                default=3,
                min_value=1,
                max_value=7,
                step=2,
                subheader="Sobel",
                help=(
                    "Kernel size used by the Sobel operator. "
                    "Larger values detect broader gradients."
                ),
            ),
            Control(
                key="laplacian_kernel_size",
                label="Laplacian Kernel Size",
                kind="slider",
                default=3,
                min_value=1,
                max_value=7,
                step=2,
                subheader="Laplacian",
                help=(
                    "Aperture size used by the Laplacian operator. "
                    "Larger values can increase sensitivity to noise."
                ),
            ),
        ),
    ),
    Section(
        title="🧩 Morphological Operations",
        categories=frozenset({"Morphology"}),
        caption=(
            "Morphological operations are applied to "
            "the thresholded binary image."
        ),
        controls=(
            Control(
                key="morph_kernel_size",
                label="Structuring Element Size",
                kind="slider",
                default=5,
                min_value=1,
                max_value=15,
                step=2,
                help=(
                    "Size of the structuring element used for "
                    "erosion, dilation, opening, and closing."
                ),
            ),
            Control(
                key="morph_iterations",
                label="Iterations",
                kind="slider",
                default=1,
                min_value=1,
                max_value=5,
                help=(
                    "Number of times the selected morphological "
                    "operation is applied."
                ),
            ),
        ),
    ),
    Section(
        title="🔲 Contour Detection",
        categories=frozenset({"Contours"}),
        controls=(
            Control(
                key="contour_retrieval_mode",
                label="Retrieval Mode",
                kind="select",
                default="External",
                options=("External", "List", "Tree", "CComp"),
                help=(
                    "Controls how contour hierarchy is retrieved. "
                    "External keeps only outer contours."
                ),
            ),
            Control(
                key="contour_approx_method",
                label="Approximation Method",
                kind="select",
                default="Simple",
                options=("Simple", "None"),
                help=(
                    "Simple compresses contour segments, while "
                    "None preserves all contour points."
                ),
            ),
            Control(
                key="min_contour_area",
                label="Minimum Contour Area",
                kind="slider",
                default=50,
                min_value=0,
                max_value=5000,
                step=10,
                help=(
                    "Contours smaller than this area are filtered "
                    "out as potential noise."
                ),
            ),
        ),
    ),
    Section(
        title="🎨 Color Analysis",
        categories=frozenset({"Color Analysis"}),
        caption="These controls require a color image.",
        controls=(
            Control(
                key="hue_range",
                label="Hue Range",
                kind="range",
                default=(0, 179),
                min_value=0,
                max_value=179,
                help="OpenCV represents Hue in the range 0–179.",
            ),
            Control(
                key="saturation_range",
                label="Saturation Range",
                kind="range",
                default=(0, 255),
                min_value=0,
                max_value=255,
                help=(
                    "Controls the minimum and maximum "
                    "color saturation."
                ),
            ),
            Control(
                key="value_range",
                label="Value (Brightness) Range",
                kind="range",
                default=(0, 255),
                min_value=0,
                max_value=255,
                help=(
                    "Controls the minimum and maximum "
                    "brightness values."
                ),
            ),
        ),
    ),
    Section(
        title="🔍 Feature Detection",
        categories=frozenset({"Feature Detection"}),
        controls=(
            Control(
                key="harris_block_size",
                label="Harris Block Size",
                kind="slider",
                default=2,
                min_value=2,
                max_value=10,
                help=(
                    "Neighborhood size considered when "
                    "computing the Harris response."
                ),
            ),
            Control(
                key="harris_ksize",
                label="Harris Sobel Aperture",
                kind="slider",
                default=3,
                min_value=1,
                max_value=31,
                step=2,
                help=(
                    "Aperture size used for the Sobel derivatives "
                    "inside the Harris detector."
                ),
            ),
            Control(
                key="harris_sensitivity",
                label="Harris Sensitivity (k)",
                kind="slider",
                default=0.04,
                min_value=0.01,
                max_value=0.20,
                step=0.01,
                help=(
                    "Harris detector sensitivity parameter. "
                    "Higher values make the detector more selective."
                ),
            ),
            Control(
                key="harris_threshold",
                label="Harris Response Threshold",
                kind="slider",
                default=0.01,
                min_value=0.01,
                max_value=0.50,
                step=0.01,
                help=(
                    "Fraction of the strongest Harris response "
                    "required to mark a corner."
                ),
            ),
            Control(
                key="orb_features",
                label="ORB Max Keypoints",
                kind="slider",
                default=300,
                min_value=50,
                max_value=1000,
                step=50,
                help="Maximum number of ORB keypoints to detect.",
            ),
        ),
    ),
    Section(
        title="😀 Face Detection",
        categories=frozenset({"Object Detection"}),
        controls=(
            Control(
                key="face_scale_factor",
                label="Scale Factor",
                kind="slider",
                default=1.10,
                min_value=1.05,
                max_value=1.50,
                step=0.05,
                help=(
                    "How much the image size is reduced at each "
                    "scale. Smaller values are slower but more "
                    "thorough."
                ),
            ),
            Control(
                key="face_min_neighbors",
                label="Minimum Neighbors",
                kind="slider",
                default=5,
                min_value=1,
                max_value=10,
                help=(
                    "How many overlapping detections are required "
                    "to confirm a face. Higher values reduce false "
                    "positives."
                ),
            ),
            Control(
                key="face_min_size",
                label="Minimum Face Size (px)",
                kind="slider",
                default=30,
                min_value=10,
                max_value=200,
                step=10,
                help=(
                    "Smallest face size (in pixels) the detector "
                    "will consider."
                ),
            ),
        ),
    ),
    Section(
        title="🔎 Object Detection",
        categories=frozenset({"Object Detection"}),
        controls=(
            Control(
                key="object_type",
                label="Object Class",
                kind="select",
                default=next(iter(OBJECT_CASCADES)),
                options=tuple(OBJECT_CASCADES.keys()),
                help=(
                    "Which bundled Haar Cascade classifier to run. "
                    "Not every image will contain the selected class."
                ),
            ),
            Control(
                key="object_scale_factor",
                label="Scale Factor",
                kind="slider",
                default=1.10,
                min_value=1.05,
                max_value=1.50,
                step=0.05,
                help=(
                    "How much the image size is reduced at each "
                    "scale. Smaller values are slower but more "
                    "thorough."
                ),
            ),
            Control(
                key="object_min_neighbors",
                label="Minimum Neighbors",
                kind="slider",
                default=5,
                min_value=1,
                max_value=10,
                help=(
                    "How many overlapping detections are required "
                    "to confirm an object. Higher values reduce "
                    "false positives."
                ),
            ),
            Control(
                key="object_min_size",
                label="Minimum Object Size (px)",
                kind="slider",
                default=20,
                min_value=10,
                max_value=200,
                step=10,
                help=(
                    "Smallest object size (in pixels) the detector "
                    "will consider."
                ),
            ),
        ),
    ),
    Section(
        title="🤖 Deep Learning",
        categories=frozenset({"Deep Learning"}),
        caption=(
            "These stages need the optional ML dependencies "
            "(`pip install -r requirements-ml.txt`)."
        ),
        controls=(
            Control(
                key="yolo_model",
                label="YOLO Model",
                kind="select",
                default=next(iter(YOLO_MODELS)),
                options=tuple(YOLO_MODELS.keys()),
                subheader="YOLO",
                help=(
                    "Larger models are more accurate but slower. "
                    "Weights download automatically on first use."
                ),
            ),
            Control(
                key="yolo_confidence",
                label="Confidence Threshold",
                kind="slider",
                default=0.25,
                min_value=0.05,
                max_value=0.95,
                step=0.05,
                help=(
                    "Minimum score a detection needs to be shown. "
                    "Raise it to cut false positives, lower it to "
                    "catch harder objects."
                ),
            ),
            Control(
                key="yolo_iou",
                label="NMS IoU Threshold",
                kind="slider",
                default=0.45,
                min_value=0.10,
                max_value=0.90,
                step=0.05,
                help=(
                    "How much two boxes may overlap before "
                    "non-maximum suppression treats them as the same "
                    "object and keeps only the stronger one."
                ),
            ),
            Control(
                key="mediapipe_task",
                label="MediaPipe Task",
                kind="select",
                default=next(iter(MEDIAPIPE_TASKS)),
                options=tuple(MEDIAPIPE_TASKS.keys()),
                subheader="MediaPipe",
                help=(
                    "Which landmark model to run: a dense face mesh, "
                    "hand skeletons, or a full-body pose."
                ),
            ),
        ),
    ),
)


def default_params() -> dict:
    """
    The params dict as it would be with every control left at its
    default. Used by the tests so they don't have to restate all of
    the parameter names by hand, and so a control added here is
    automatically exercised.
    """

    return {
        control.key: control.default
        for section in SECTIONS
        for control in section.controls
    }


def _render_control(control: Control):
    """Draw a single control and return the value the user picked."""

    if control.subheader:
        st.subheader(control.subheader)

    common = dict(
        label=control.label,
        help=control.help,
        key=control.key,
    )

    if control.kind == "select":
        options = list(control.options)

        # Select the declared default explicitly rather than relying on
        # it happening to be the first option, so default_params() and
        # what the widget actually shows can't drift apart.
        return st.selectbox(
            options=options,
            index=options.index(control.default),
            **common,
        )

    # Both plain and range sliders are st.slider; passing a tuple as
    # the default value is what makes Streamlit render a range.
    slider_kwargs = dict(
        min_value=control.min_value,
        max_value=control.max_value,
        value=control.default,
        **common,
    )

    if control.step is not None:
        slider_kwargs["step"] = control.step

    return st.slider(**slider_kwargs)


def render_sidebar(active_category: str = "") -> dict:
    """
    Render every control section into the sidebar and return the
    collected values as the params dict that stage handlers read.

    The section(s) relevant to `active_category` are expanded and the
    rest are collapsed, so the controls that affect what's currently on
    screen are the ones in view.
    """

    params: dict = {}

    for section in SECTIONS:
        is_relevant = active_category in section.categories

        with st.expander(section.title, expanded=is_relevant):

            if section.caption:
                st.caption(section.caption)

            for control in section.controls:
                params[control.key] = _render_control(control)

            if section.warning is not None:
                message = section.warning(params)

                if message:
                    st.warning(message)

    return params
