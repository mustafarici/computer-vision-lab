import numpy as np
import streamlit as st

from modules.basic_ops import (
    apply_gaussian_blur,
    apply_threshold,
    convert_to_grayscale,
)
from modules.color_analysis import (
    build_hsv_channel_composite,
    build_rgb_channel_composite,
)
from modules.color_threshold import apply_color_threshold
from modules.contours import apply_contour_detection
from modules.edges import apply_canny, apply_laplacian, apply_sobel
from modules.feature_detection import (
    apply_harris_corners,
    apply_orb_keypoints,
    compute_harris_response,
)
from modules.histogram import build_histogram_figure, compute_histogram
from modules.io_utils import load_image
from modules.morphology import (
    apply_closing,
    apply_dilation,
    apply_erosion,
    apply_opening,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Computer Vision Lab",
    page_icon="🧪",
    layout="wide",
)


# ==================================================
# APPLICATION HEADER
# ==================================================

st.title("🧪 Computer Vision Lab")
st.caption("Interactive image processing laboratory")


# ==================================================
# IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
    help="Upload a PNG, JPG, or JPEG image to start processing.",
)


if uploaded_file is not None:

    # ==================================================
    # LOAD IMAGE
    # ==================================================

    try:
        image_np = load_image(uploaded_file.getvalue())

    except Exception:
        st.error(
            "This file couldn't be read as an image. "
            "Please upload a valid JPG or PNG file."
        )
        st.stop()


    # ==================================================
    # IMAGE INFORMATION
    # ==================================================

    image_height, image_width = image_np.shape[:2]

    channels = (
        image_np.shape[2]
        if image_np.ndim == 3
        else 1
    )

    is_color_image = (
        image_np.ndim == 3
        and image_np.shape[2] >= 3
    )


    # ==================================================
    # STAGE DEFINITIONS
    # ==================================================

    STAGES = {
        "Original Image": {
            "category": "Basic Processing",
            "description": (
                "Displays the original uploaded image without "
                "any image processing."
            ),
            "pipeline": "Input image",
            "requires_color": False,
        },

        "Grayscale Image": {
            "category": "Basic Processing",
            "description": (
                "Converts the input image from RGB/BGR color "
                "representation to a single grayscale channel."
            ),
            "pipeline": "Original → Grayscale",
            "requires_color": False,
        },

        "Binary Image": {
            "category": "Basic Processing",
            "description": (
                "Converts the grayscale image into a binary "
                "black-and-white image using a threshold value."
            ),
            "pipeline": "Original → Grayscale → Threshold",
            "requires_color": False,
        },

        "Gaussian Blur": {
            "category": "Basic Processing",
            "description": (
                "Applies Gaussian smoothing to reduce image noise "
                "and high-frequency details."
            ),
            "pipeline": "Original → Grayscale → Gaussian Blur",
            "requires_color": False,
        },

        "Canny Edge Detection": {
            "category": "Edge Detection",
            "description": (
                "Detects edges using the Canny edge detection "
                "algorithm after Gaussian smoothing."
            ),
            "pipeline": (
                "Original → Grayscale → Gaussian Blur → Canny"
            ),
            "requires_color": False,
        },

        "Sobel Edge Detection": {
            "category": "Edge Detection",
            "description": (
                "Computes image intensity gradients using the "
                "Sobel operator."
            ),
            "pipeline": "Original → Grayscale → Sobel",
            "requires_color": False,
        },

        "Laplacian Edge Detection": {
            "category": "Edge Detection",
            "description": (
                "Detects regions of rapid intensity change using "
                "the Laplacian operator."
            ),
            "pipeline": "Original → Grayscale → Laplacian",
            "requires_color": False,
        },

        "Erosion": {
            "category": "Morphology",
            "description": (
                "Shrinks foreground regions using a morphological "
                "erosion operation."
            ),
            "pipeline": (
                "Original → Grayscale → Threshold → Erosion"
            ),
            "requires_color": False,
        },

        "Dilation": {
            "category": "Morphology",
            "description": (
                "Expands foreground regions using a morphological "
                "dilation operation."
            ),
            "pipeline": (
                "Original → Grayscale → Threshold → Dilation"
            ),
            "requires_color": False,
        },

        "Opening": {
            "category": "Morphology",
            "description": (
                "Applies erosion followed by dilation to remove "
                "small foreground noise."
            ),
            "pipeline": (
                "Original → Grayscale → Threshold → Opening"
            ),
            "requires_color": False,
        },

        "Closing": {
            "category": "Morphology",
            "description": (
                "Applies dilation followed by erosion to close "
                "small holes and gaps."
            ),
            "pipeline": (
                "Original → Grayscale → Threshold → Closing"
            ),
            "requires_color": False,
        },

        "Contour Detection": {
            "category": "Contours",
            "description": (
                "Detects object contours from the thresholded "
                "binary image."
            ),
            "pipeline": (
                "Original → Grayscale → Threshold → Contours"
            ),
            "requires_color": False,
        },

        "RGB Channels": {
            "category": "Color Analysis",
            "description": (
                "Separates the input image into Red, Green, "
                "and Blue channel representations."
            ),
            "pipeline": "Original → RGB Channel Separation",
            "requires_color": True,
        },

        "HSV Channels": {
            "category": "Color Analysis",
            "description": (
                "Separates the image into Hue, Saturation, "
                "and Value components."
            ),
            "pipeline": "Original → HSV Conversion",
            "requires_color": True,
        },

        "Color Mask": {
            "category": "Color Analysis",
            "description": (
                "Creates a binary mask containing pixels that "
                "fall inside the selected HSV range."
            ),
            "pipeline": "Original → HSV → Color Mask",
            "requires_color": True,
        },

        "Color Threshold Result": {
            "category": "Color Analysis",
            "description": (
                "Keeps pixels inside the selected HSV range "
                "and masks out the remaining regions."
            ),
            "pipeline": (
                "Original → HSV → Color Thresholding"
            ),
            "requires_color": True,
        },

        "Harris Corners": {
            "category": "Feature Detection",
            "description": (
                "Detects corners using the Harris corner "
                "detection algorithm."
            ),
            "pipeline": (
                "Original → Grayscale → Harris Response → Corners"
            ),
            "requires_color": False,
        },

        "ORB Keypoints": {
            "category": "Feature Detection",
            "description": (
                "Detects local image features using the "
                "ORB feature detector."
            ),
            "pipeline": "Original → Grayscale → ORB",
            "requires_color": False,
        },

        "Grayscale Histogram": {
            "category": "Histogram Analysis",
            "description": (
                "Displays the distribution of grayscale intensity "
                "values in the image."
            ),
            "pipeline": "Original → Grayscale → Histogram",
            "requires_color": False,
        },
    }


    # ==================================================
    # SESSION STATE
    # ==================================================

    if (
        "current_stage" not in st.session_state
        or st.session_state.current_stage not in STAGES
    ):
        st.session_state.current_stage = "Original Image"


    stage = st.session_state.current_stage


    # ==================================================
    # SIDEBAR CONTROLS
    # ==================================================

    with st.sidebar:

        st.header("⚙️ Image Processing")
        st.divider()


        # ==================================================
        # BASIC PROCESSING
        # ==================================================

        with st.expander(
            "🔧 Basic Processing",
            expanded=True,
        ):

            st.subheader("Threshold")

            threshold_value = st.slider(
                "Threshold Value",
                min_value=0,
                max_value=255,
                value=127,
                help=(
                    "Pixel intensity threshold used to convert "
                    "the grayscale image into a binary image."
                ),
            )


            st.subheader("Gaussian Blur")

            kernel_size = st.slider(
                "Kernel Size",
                min_value=1,
                max_value=15,
                value=5,
                step=2,
                help=(
                    "Controls the size of the Gaussian smoothing "
                    "kernel. Larger values produce stronger blur."
                ),
            )


        # ==================================================
        # EDGE DETECTION
        # ==================================================

        with st.expander("📐 Edge Detection"):

            st.subheader("Canny")

            lower_threshold = st.slider(
                "Lower Threshold",
                min_value=0,
                max_value=255,
                value=50,
                help=(
                    "Lower boundary used by the Canny edge detector "
                    "to identify weak edges."
                ),
            )

            upper_threshold = st.slider(
                "Upper Threshold",
                min_value=0,
                max_value=255,
                value=150,
                help=(
                    "Upper boundary used by the Canny edge detector "
                    "to identify strong edges."
                ),
            )

            if lower_threshold > upper_threshold:
                st.warning(
                    "Lower threshold is higher than upper threshold."
                )


            st.subheader("Sobel")

            sobel_kernel_size = st.slider(
                "Sobel Kernel Size",
                min_value=1,
                max_value=7,
                value=3,
                step=2,
                help=(
                    "Kernel size used by the Sobel operator. "
                    "Larger values detect broader gradients."
                ),
            )


            st.subheader("Laplacian")

            laplacian_kernel_size = st.slider(
                "Laplacian Kernel Size",
                min_value=1,
                max_value=7,
                value=3,
                step=2,
                help=(
                    "Aperture size used by the Laplacian operator. "
                    "Larger values can increase sensitivity to noise."
                ),
            )


        # ==================================================
        # MORPHOLOGY
        # ==================================================

        with st.expander("🧩 Morphological Operations"):

            morph_kernel_size = st.slider(
                "Structuring Element Size",
                min_value=1,
                max_value=15,
                value=5,
                step=2,
                help=(
                    "Size of the structuring element used for "
                    "erosion, dilation, opening, and closing."
                ),
            )

            morph_iterations = st.slider(
                "Iterations",
                min_value=1,
                max_value=5,
                value=1,
                help=(
                    "Number of times the selected morphological "
                    "operation is applied."
                ),
            )

            st.caption(
                "Morphological operations are applied to "
                "the thresholded binary image."
            )


        # ==================================================
        # CONTOURS
        # ==================================================

        with st.expander("🔲 Contour Detection"):

            contour_retrieval_mode = st.selectbox(
                "Retrieval Mode",
                options=[
                    "External",
                    "List",
                    "Tree",
                    "CComp",
                ],
                help=(
                    "Controls how contour hierarchy is retrieved. "
                    "External keeps only outer contours."
                ),
            )

            contour_approx_method = st.selectbox(
                "Approximation Method",
                options=[
                    "Simple",
                    "None",
                ],
                help=(
                    "Simple compresses contour segments, while "
                    "None preserves all contour points."
                ),
            )

            min_contour_area = st.slider(
                "Minimum Contour Area",
                min_value=0,
                max_value=5000,
                value=50,
                step=10,
                help=(
                    "Contours smaller than this area are filtered "
                    "out as potential noise."
                ),
            )


        # ==================================================
        # COLOR ANALYSIS
        # ==================================================

        with st.expander("🎨 Color Analysis"):

            st.caption(
                "These controls require a color image."
            )

            hue_range = st.slider(
                "Hue Range",
                min_value=0,
                max_value=179,
                value=(0, 179),
                help=(
                    "OpenCV represents Hue in the range 0–179."
                ),
            )

            saturation_range = st.slider(
                "Saturation Range",
                min_value=0,
                max_value=255,
                value=(0, 255),
                help=(
                    "Controls the minimum and maximum "
                    "color saturation."
                ),
            )

            value_range = st.slider(
                "Value (Brightness) Range",
                min_value=0,
                max_value=255,
                value=(0, 255),
                help=(
                    "Controls the minimum and maximum "
                    "brightness values."
                ),
            )


        # ==================================================
        # FEATURE DETECTION
        # ==================================================

        with st.expander("🔍 Feature Detection"):

            harris_block_size = st.slider(
                "Harris Block Size",
                min_value=2,
                max_value=10,
                value=2,
                help=(
                    "Neighborhood size considered when "
                    "computing the Harris response."
                ),
            )

            harris_ksize = st.slider(
                "Harris Sobel Aperture",
                min_value=1,
                max_value=31,
                value=3,
                step=2,
                help=(
                    "Aperture size used for the Sobel derivatives "
                    "inside the Harris detector."
                ),
            )

            harris_sensitivity = st.slider(
                "Harris Sensitivity (k)",
                min_value=0.01,
                max_value=0.20,
                value=0.04,
                step=0.01,
                help=(
                    "Harris detector sensitivity parameter. "
                    "Higher values make the detector more selective."
                ),
            )

            harris_threshold = st.slider(
                "Harris Response Threshold",
                min_value=0.01,
                max_value=0.50,
                value=0.01,
                step=0.01,
                help=(
                    "Fraction of the strongest Harris response "
                    "required to mark a corner."
                ),
            )

            orb_features = st.slider(
                "ORB Max Keypoints",
                min_value=50,
                max_value=1000,
                value=300,
                step=50,
                help=(
                    "Maximum number of ORB keypoints to detect."
                ),
            )


        # ==================================================
        # IMAGE INFORMATION
        # ==================================================

        with st.expander("ℹ️ Image Information", expanded=True):

            st.write(
                f"**Resolution:** "
                f"{image_width} × {image_height}"
            )

            st.write(
                f"**Channels:** {channels}"
            )

            if is_color_image:
                st.write("**Image Type:** Color")

            else:
                st.write("**Image Type:** Grayscale")


    # ==================================================
    # CURRENT STAGE INFORMATION
    # ==================================================

    stage_info = STAGES[stage]

    stage_category = stage_info["category"]
    stage_description = stage_info["description"]
    stage_pipeline = stage_info["pipeline"]
    stage_requires_color = stage_info["requires_color"]


    # ==================================================
    # COLOR VALIDATION
    # ==================================================

    if stage_requires_color and not is_color_image:

        st.divider()

        st.subheader(stage)

        st.warning(
            "⚠️ This stage requires a color image."
        )

        st.info(
            "Please upload an RGB/color image to use "
            "RGB, HSV, and color thresholding operations."
        )

        st.write("")

        # Navigation is still available even when
        # the selected stage cannot be processed.

    else:

        # ==================================================
        # LAZY PROCESSING PIPELINE
        # ==================================================

        current_image = None
        extra_info = None


        # ==================================================
        # ORIGINAL IMAGE
        # ==================================================

        if stage == "Original Image":

            current_image = image_np


        # ==================================================
        # GRAYSCALE
        # ==================================================

        elif stage == "Grayscale Image":

            grayscale = convert_to_grayscale(image_np)

            current_image = grayscale


        # ==================================================
        # BINARY
        # ==================================================

        elif stage == "Binary Image":

            grayscale = convert_to_grayscale(image_np)

            current_image = apply_threshold(
                grayscale,
                threshold_value,
            )


        # ==================================================
        # GAUSSIAN BLUR
        # ==================================================

        elif stage == "Gaussian Blur":

            grayscale = convert_to_grayscale(image_np)

            current_image = apply_gaussian_blur(
                grayscale,
                kernel_size,
            )


        # ==================================================
        # CANNY
        # ==================================================

        elif stage == "Canny Edge Detection":

            grayscale = convert_to_grayscale(image_np)

            blurred = apply_gaussian_blur(
                grayscale,
                kernel_size,
            )

            current_image = apply_canny(
                blurred,
                lower_threshold,
                upper_threshold,
            )

            extra_info = (
                "Canny uses the Gaussian Blur settings "
                "defined above."
            )


        # ==================================================
        # SOBEL
        # ==================================================

        elif stage == "Sobel Edge Detection":

            grayscale = convert_to_grayscale(image_np)

            current_image = apply_sobel(
                grayscale,
                sobel_kernel_size,
            )


        # ==================================================
        # LAPLACIAN
        # ==================================================

        elif stage == "Laplacian Edge Detection":

            grayscale = convert_to_grayscale(image_np)

            current_image = apply_laplacian(
                grayscale,
                laplacian_kernel_size,
            )


        # ==================================================
        # MORPHOLOGY
        # ==================================================

        elif stage in [
            "Erosion",
            "Dilation",
            "Opening",
            "Closing",
        ]:

            grayscale = convert_to_grayscale(image_np)

            binary = apply_threshold(
                grayscale,
                threshold_value,
            )

            if stage == "Erosion":

                current_image = apply_erosion(
                    binary,
                    morph_kernel_size,
                    morph_iterations,
                )

            elif stage == "Dilation":

                current_image = apply_dilation(
                    binary,
                    morph_kernel_size,
                    morph_iterations,
                )

            elif stage == "Opening":

                current_image = apply_opening(
                    binary,
                    morph_kernel_size,
                    morph_iterations,
                )

            elif stage == "Closing":

                current_image = apply_closing(
                    binary,
                    morph_kernel_size,
                    morph_iterations,
                )


        # ==================================================
        # CONTOUR DETECTION
        # ==================================================

        elif stage == "Contour Detection":

            grayscale = convert_to_grayscale(image_np)

            binary = apply_threshold(
                grayscale,
                threshold_value,
            )

            current_image, contour_count = (
                apply_contour_detection(
                    binary,
                    grayscale,
                    contour_retrieval_mode,
                    contour_approx_method,
                    min_contour_area,
                )
            )

            extra_info = (
                f"Found **{contour_count}** contours "
                "after the area filter."
            )


        # ==================================================
        # RGB CHANNELS
        # ==================================================

        elif stage == "RGB Channels":

            current_image = build_rgb_channel_composite(
                image_np
            )

            extra_info = (
                "Left to right: **Red**, **Green**, "
                "**Blue** channel isolated."
            )


        # ==================================================
        # HSV CHANNELS
        # ==================================================

        elif stage == "HSV Channels":

            current_image = build_hsv_channel_composite(
                image_np
            )

            extra_info = (
                "Left to right: **Hue** (colorized), "
                "**Saturation**, **Value**."
            )


        # ==================================================
        # COLOR THRESHOLDING
        # ==================================================

        elif stage in [
            "Color Mask",
            "Color Threshold Result",
        ]:

            color_mask, color_result = (
                apply_color_threshold(
                    image_np,
                    hue_range,
                    saturation_range,
                    value_range,
                )
            )

            if stage == "Color Mask":

                current_image = color_mask

                extra_info = (
                    "White pixels fall inside the "
                    "selected HSV range."
                )

            else:

                current_image = color_result

                extra_info = (
                    "Pixels outside the selected HSV "
                    "range are masked out."
                )


        # ==================================================
        # HARRIS CORNERS
        # ==================================================

        elif stage == "Harris Corners":

            grayscale = convert_to_grayscale(image_np)

            harris_response = compute_harris_response(
                grayscale,
                harris_block_size,
                harris_ksize,
                harris_sensitivity,
            )

            current_image = apply_harris_corners(
                grayscale,
                harris_response,
                harris_threshold,
            )

            extra_info = (
                "Detected corners are marked by "
                "strong intensity changes in multiple directions."
            )


        # ==================================================
        # ORB
        # ==================================================

        elif stage == "ORB Keypoints":

            grayscale = convert_to_grayscale(image_np)

            current_image, orb_keypoint_count = (
                apply_orb_keypoints(
                    grayscale,
                    orb_features,
                )
            )

            extra_info = (
                f"Found **{orb_keypoint_count}** ORB keypoints "
                "(circle size ≈ feature scale)."
            )


        # ==================================================
        # HISTOGRAM
        # ==================================================

        elif stage == "Grayscale Histogram":

            grayscale = convert_to_grayscale(image_np)

            histogram = compute_histogram(
                grayscale
            )

            current_image = build_histogram_figure(
                histogram
            )


        # ==================================================
        # MAIN DISPLAY
        # ==================================================

        st.divider()

        st.subheader(stage)

        st.caption(
            f"**Category:** {stage_category}"
        )

        st.write(stage_description)

        with st.expander(
            "🔗 Processing Pipeline",
            expanded=False,
        ):
            st.code(
                stage_pipeline,
                language="text",
            )


        # ==================================================
        # DISPLAY CURRENT RESULT
        # ==================================================

        if stage == "Grayscale Histogram":

            st.pyplot(
                current_image,
                width="content",
            )

        elif current_image is None:

            st.info(
                "No result is available for this stage."
            )

        else:

            st.image(
                current_image,
                width="stretch",
            )

            if extra_info:

                st.caption(
                    extra_info
                )


        # ==================================================
        # HISTOGRAM STATISTICS
        # ==================================================

        if stage == "Grayscale Histogram":

            grayscale = convert_to_grayscale(
                image_np
            )

            histogram_values = compute_histogram(
                grayscale
            )

            st.subheader(
                "📊 Image Statistics"
            )

            stat_col1, stat_col2, stat_col3, stat_col4 = (
                st.columns(4)
            )

            with stat_col1:

                st.metric(
                    "Mean",
                    f"{grayscale.mean():.2f}",
                )

            with stat_col2:

                st.metric(
                    "Median",
                    f"{float(np.median(grayscale)):.2f}",
                )

            with stat_col3:

                st.metric(
                    "Minimum",
                    f"{grayscale.min()}",
                )

            with stat_col4:

                st.metric(
                    "Maximum",
                    f"{grayscale.max()}",
                )


    # ==================================================
    # NAVIGATION
    # ==================================================

    st.write("")

    stage_titles = list(STAGES.keys())

    current_index = stage_titles.index(stage)


    previous_col, counter_col, next_col = st.columns(
        [1, 2, 1]
    )


    with previous_col:

        if st.button(
            "← Previous",
            key="previous_button",
            width="stretch",
        ):

            st.session_state.current_stage = (
                stage_titles[
                    (current_index - 1)
                    % len(stage_titles)
                ]
            )

            st.rerun()


    with counter_col:

        st.markdown(
            f"""
            <div style="
                text-align: center;
                padding-top: 8px;
                font-size: 14px;
            ">
                <b>
                    {current_index + 1} / {len(stage_titles)}
                </b>
            </div>
            """,
            unsafe_allow_html=True,
        )


    with next_col:

        if st.button(
            "Next →",
            key="next_button",
            width="stretch",
        ):

            st.session_state.current_stage = (
                stage_titles[
                    (current_index + 1)
                    % len(stage_titles)
                ]
            )

            st.rerun()


    st.write("")

    st.selectbox(
        "📍 Jump to a specific stage",
        options=stage_titles,
        key="current_stage",
    )


else:

    st.info(
        "👆 Upload a JPG or PNG image to get started."
    )