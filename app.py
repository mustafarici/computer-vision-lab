import numpy as np
import streamlit as st

from modules.basic_ops import convert_to_grayscale
from modules.histogram import get_figure_download_bytes
from modules.io_utils import (
    get_image_download_bytes,
    load_image,
    save_result_locally,
)
from modules.object_detection import OBJECT_CASCADES
from modules.stages import STAGES


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
        image_np, resize_info = load_image(uploaded_file.getvalue())

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
    # SESSION STATE
    # ==================================================
    #
    # STAGES itself now lives in modules/stages.py — see that file
    # for the full stage registry (metadata + processing handler for
    # every pipeline stage). Adding a new stage means adding one
    # entry there instead of editing this file in three places.

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
        # FACE DETECTION
        # ==================================================

        with st.expander("😀 Face Detection"):

            face_scale_factor = st.slider(
                "Scale Factor",
                min_value=1.05,
                max_value=1.50,
                value=1.10,
                step=0.05,
                key="face_scale_factor",
                help=(
                    "How much the image size is reduced at each "
                    "scale. Smaller values are slower but more "
                    "thorough."
                ),
            )

            face_min_neighbors = st.slider(
                "Minimum Neighbors",
                min_value=1,
                max_value=10,
                value=5,
                key="face_min_neighbors",
                help=(
                    "How many overlapping detections are required "
                    "to confirm a face. Higher values reduce false "
                    "positives."
                ),
            )

            face_min_size = st.slider(
                "Minimum Face Size (px)",
                min_value=10,
                max_value=200,
                value=30,
                step=10,
                key="face_min_size",
                help=(
                    "Smallest face size (in pixels) the detector "
                    "will consider."
                ),
            )


        # ==================================================
        # OBJECT DETECTION
        # ==================================================

        with st.expander("🔎 Object Detection"):

            object_type = st.selectbox(
                "Object Class",
                options=list(OBJECT_CASCADES.keys()),
                help=(
                    "Which bundled Haar Cascade classifier to run. "
                    "Not every image will contain the selected class."
                ),
            )

            object_scale_factor = st.slider(
                "Scale Factor",
                min_value=1.05,
                max_value=1.50,
                value=1.10,
                step=0.05,
                key="object_scale_factor",
                help=(
                    "How much the image size is reduced at each "
                    "scale. Smaller values are slower but more "
                    "thorough."
                ),
            )

            object_min_neighbors = st.slider(
                "Minimum Neighbors",
                min_value=1,
                max_value=10,
                value=5,
                key="object_min_neighbors",
                help=(
                    "How many overlapping detections are required "
                    "to confirm an object. Higher values reduce "
                    "false positives."
                ),
            )

            object_min_size = st.slider(
                "Minimum Object Size (px)",
                min_value=10,
                max_value=200,
                value=20,
                step=10,
                key="object_min_size",
                help=(
                    "Smallest object size (in pixels) the detector "
                    "will consider."
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

            if resize_info is not None:
                st.caption(
                    f"Original: {resize_info['original_width']}×"
                    f"{resize_info['original_height']} "
                    f"(resized to {resize_info['new_width']}×"
                    f"{resize_info['new_height']} for performance)"
                )


    # ==================================================
    # SIDEBAR PARAMETERS
    # ==================================================
    #
    # Every stage handler in modules/stages.py reads its inputs from
    # this single dict instead of app.py passing each slider value
    # around by hand — one place to look when a new stage needs a
    # new parameter.

    params = dict(
        threshold_value=threshold_value,
        kernel_size=kernel_size,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        sobel_kernel_size=sobel_kernel_size,
        laplacian_kernel_size=laplacian_kernel_size,
        morph_kernel_size=morph_kernel_size,
        morph_iterations=morph_iterations,
        contour_retrieval_mode=contour_retrieval_mode,
        contour_approx_method=contour_approx_method,
        min_contour_area=min_contour_area,
        hue_range=hue_range,
        saturation_range=saturation_range,
        value_range=value_range,
        harris_block_size=harris_block_size,
        harris_ksize=harris_ksize,
        harris_sensitivity=harris_sensitivity,
        harris_threshold=harris_threshold,
        orb_features=orb_features,
        face_scale_factor=face_scale_factor,
        face_min_neighbors=face_min_neighbors,
        face_min_size=face_min_size,
        object_type=object_type,
        object_scale_factor=object_scale_factor,
        object_min_neighbors=object_min_neighbors,
        object_min_size=object_min_size,
    )


    # ==================================================
    # CURRENT STAGE INFORMATION
    # ==================================================

    stage_info = STAGES[stage]


    # ==================================================
    # COLOR VALIDATION
    # ==================================================

    if stage_info.requires_color and not is_color_image:

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
        #
        # Only the selected stage runs, and it only gets a grayscale
        # conversion if it actually uses one — the color stages
        # (Original, RGB/HSV Channels, Color Mask/Threshold) work off
        # image_np directly and receive None here.

        grayscale = (
            convert_to_grayscale(image_np)
            if stage_info.needs_grayscale
            else None
        )

        result = stage_info.handler(image_np, grayscale, params)

        current_image = result.image
        extra_info = result.extra_info


        # ==================================================
        # MAIN DISPLAY
        # ==================================================

        st.divider()

        st.subheader(stage)

        st.caption(
            f"**Category:** {stage_info.category}"
        )

        st.write(stage_info.description)

        with st.expander(
            "🔗 Processing Pipeline",
            expanded=False,
        ):
            st.code(
                stage_info.pipeline,
                language="text",
            )


        # ==================================================
        # DISPLAY CURRENT RESULT
        # ==================================================

        download_filename = (
            stage.lower().replace(" ", "_").replace("/", "_")
            + ".png"
        )

        if stage_info.is_figure:

            st.pyplot(
                current_image,
                width="content",
            )

            result_bytes = get_figure_download_bytes(current_image)

            download_col, save_col = st.columns(2)

            with download_col:
                st.download_button(
                    "⬇️ Download Histogram (PNG)",
                    data=result_bytes,
                    file_name=download_filename,
                    mime="image/png",
                    width="stretch",
                )

            with save_col:
                if st.button(
                    "💾 Save a copy to results/",
                    key="save_result_button",
                    width="stretch",
                ):
                    saved_path = save_result_locally(
                        result_bytes, download_filename
                    )
                    st.success(f"Saved to {saved_path}")

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

            result_bytes = get_image_download_bytes(current_image)

            download_col, save_col = st.columns(2)

            with download_col:
                st.download_button(
                    "⬇️ Download Result (PNG)",
                    data=result_bytes,
                    file_name=download_filename,
                    mime="image/png",
                    width="stretch",
                )

            with save_col:
                if st.button(
                    "💾 Save a copy to results/",
                    key="save_result_button",
                    width="stretch",
                ):
                    saved_path = save_result_locally(
                        result_bytes, download_filename
                    )
                    st.success(f"Saved to {saved_path}")


        # ==================================================
        # HISTOGRAM STATISTICS
        # ==================================================

        if stage_info.is_figure:

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
