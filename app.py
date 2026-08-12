import streamlit as st

from modules.io_utils import load_image
from modules.basic_ops import (
    convert_to_grayscale,
    apply_threshold,
    apply_gaussian_blur,
)
from modules.edges import apply_canny, apply_sobel, apply_laplacian
from modules.morphology import (
    apply_erosion,
    apply_dilation,
    apply_opening,
    apply_closing,
)
from modules.contours import apply_contour_detection
from modules.color_analysis import build_rgb_channel_composite, build_hsv_channel_composite
from modules.histogram import compute_histogram, build_histogram_figure


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Computer Vision Lab",
    layout="wide"
)

st.title("🧪 Computer Vision Lab")
st.caption("Interactive image processing laboratory")


# ==================================================
# IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
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


    # ==================================================
    # SIDEBAR
    # ==================================================

    with st.sidebar:

        st.header("⚙️ Image Processing")

        st.divider()


        # --------------------------------------------------
        # THRESHOLD
        # --------------------------------------------------

        st.subheader("Threshold")

        threshold_value = st.slider(
            "Threshold Value",
            min_value=0,
            max_value=255,
            value=127,
            help=(
                "The threshold value used to separate "
                "an image into black and white regions."
            )
        )


        st.divider()


        # --------------------------------------------------
        # GAUSSIAN BLUR
        # --------------------------------------------------

        st.subheader("Gaussian Blur")

        kernel_size = st.slider(
            "Kernel Size",
            min_value=1,
            max_value=15,
            value=5,
            step=2,
            help=(
                "Controls the size of the blur filter. "
                "Larger values apply stronger blur."
            )
        )


        st.divider()


        # --------------------------------------------------
        # CANNY EDGE DETECTION
        # --------------------------------------------------

        st.subheader("Canny Edge Detection")

        lower_threshold = st.slider(
            "Lower Threshold",
            min_value=0,
            max_value=255,
            value=50,
            help="The lower boundary for weak edges."
        )

        upper_threshold = st.slider(
            "Upper Threshold",
            min_value=0,
            max_value=255,
            value=150,
            help="The threshold for strong edges."
        )

        if lower_threshold > upper_threshold:
            st.warning(
                "Lower threshold is higher than the upper threshold. "
                "Canny will treat every edge as weak — "
                "consider lowering it below the upper value."
            )


        st.divider()


        # --------------------------------------------------
        # SOBEL EDGE DETECTION
        # --------------------------------------------------

        st.subheader("Sobel Edge Detection")

        sobel_kernel_size = st.slider(
            "Sobel Kernel Size",
            min_value=1,
            max_value=7,
            value=3,
            step=2,
            help=(
                "Size of the Sobel operator. Larger values "
                "detect broader, less noisy gradients."
            )
        )


        st.divider()


        # --------------------------------------------------
        # LAPLACIAN EDGE DETECTION
        # --------------------------------------------------

        st.subheader("Laplacian Edge Detection")

        laplacian_kernel_size = st.slider(
            "Laplacian Kernel Size",
            min_value=1,
            max_value=7,
            value=3,
            step=2,
            help=(
                "Aperture size for the Laplacian operator. "
                "Larger values are more sensitive to noise."
            )
        )


        st.divider()


        # --------------------------------------------------
        # MORPHOLOGICAL OPERATIONS
        # --------------------------------------------------

        st.subheader("Morphological Operations")

        st.caption("Applied on top of the binary (thresholded) image.")

        morph_kernel_size = st.slider(
            "Structuring Element Size",
            min_value=1,
            max_value=15,
            value=5,
            step=2,
            help=(
                "Size of the structuring element (kernel) "
                "used for erosion, dilation, opening and closing."
            )
        )

        morph_iterations = st.slider(
            "Iterations",
            min_value=1,
            max_value=5,
            value=1,
            help="How many times the operation is applied in a row."
        )


        st.divider()


        # --------------------------------------------------
        # CONTOUR DETECTION
        # --------------------------------------------------

        st.subheader("Contour Detection")

        st.caption("Applied on top of the binary (thresholded) image.")

        contour_retrieval_mode = st.selectbox(
            "Retrieval Mode",
            options=["External", "List", "Tree", "CComp"],
            index=0,
            help=(
                "External: only outermost contours. "
                "List: all contours, no hierarchy. "
                "Tree/CComp: full or two-level nesting hierarchy."
            )
        )

        contour_approx_method = st.selectbox(
            "Approximation Method",
            options=["Simple", "None"],
            index=0,
            help=(
                "Simple compresses straight segments to their "
                "endpoints. None keeps every contour point."
            )
        )

        min_contour_area = st.slider(
            "Minimum Contour Area",
            min_value=0,
            max_value=5000,
            value=50,
            step=10,
            help="Filters out tiny contours likely caused by noise."
        )


        st.divider()


        # --------------------------------------------------
        # COLOR SPACE ANALYSIS
        # --------------------------------------------------

        st.subheader("Color Space Analysis")

        st.caption(
            "Splits the original image into its RGB and HSV "
            "channels, shown side by side. Requires a color "
            "(non-grayscale) input image."
        )


        st.divider()


        # --------------------------------------------------
        # IMAGE INFORMATION
        # --------------------------------------------------

        st.subheader("Image Information")

        st.write(f"**Resolution:** {image_width} × {image_height}")
        st.write(f"**Channels:** {channels}")


    # ==================================================
    # PROCESSING PIPELINE
    #
    # Each step is cached inside its own module, so moving
    # one slider only recomputes the steps that actually
    # depend on it, instead of rerunning the full pipeline
    # on every interaction.
    # ==================================================

    grayscale = convert_to_grayscale(image_np)
    binary = apply_threshold(grayscale, threshold_value)
    blurred = apply_gaussian_blur(grayscale, kernel_size)
    edges = apply_canny(blurred, lower_threshold, upper_threshold)
    sobel = apply_sobel(grayscale, sobel_kernel_size)
    laplacian = apply_laplacian(grayscale, laplacian_kernel_size)
    erosion = apply_erosion(binary, morph_kernel_size, morph_iterations)
    dilation = apply_dilation(binary, morph_kernel_size, morph_iterations)
    opening = apply_opening(binary, morph_kernel_size, morph_iterations)
    closing = apply_closing(binary, morph_kernel_size, morph_iterations)
    contour_image, contour_count = apply_contour_detection(
        binary,
        grayscale,
        contour_retrieval_mode,
        contour_approx_method,
        min_contour_area
    )
    histogram = compute_histogram(grayscale)
    histogram_figure = build_histogram_figure(histogram)
    rgb_composite = build_rgb_channel_composite(image_np)
    hsv_composite = build_hsv_channel_composite(image_np)


    # ==================================================
    # IMAGE GALLERY
    # ==================================================

    images = [
        ("Original Image", image_np),
        ("Grayscale Image", grayscale),
        ("Binary Image", binary),
        ("Gaussian Blur", blurred),
        ("Canny Edge Detection", edges),
        ("Sobel Edge Detection", sobel),
        ("Laplacian Edge Detection", laplacian),
        ("Erosion", erosion),
        ("Dilation", dilation),
        ("Opening", opening),
        ("Closing", closing),
        ("Contour Detection", contour_image),
        ("RGB Channels", rgb_composite),
        ("HSV Channels", hsv_composite),
        ("Grayscale Histogram", histogram_figure)
    ]


    # ==================================================
    # NAVIGATION STATE
    #
    # Stored by title (not index) so that the Previous/Next
    # buttons and the stage selectbox below can share the
    # exact same session_state key without fighting each
    # other for control.
    # ==================================================

    titles = [title for title, _ in images]

    if (
        "current_stage" not in st.session_state
        or st.session_state.current_stage not in titles
    ):
        st.session_state.current_stage = titles[0]

    current_index = titles.index(st.session_state.current_stage)
    current_title, current_image = images[current_index]


    # ==================================================
    # MAIN DISPLAY
    # ==================================================

    st.divider()
    st.subheader(current_title)


    # ==================================================
    # DISPLAY CURRENT RESULT
    # ==================================================

    if current_title == "Grayscale Histogram":
        st.pyplot(current_image, use_container_width=False)
    elif current_image is None:
        st.info(
            "This stage needs a color image, but the uploaded "
            "image only has a single (grayscale) channel."
        )
    else:
        st.image(current_image, use_container_width=True)

        if current_title == "Contour Detection":
            st.caption(f"Found **{contour_count}** contours after the area filter.")
        elif current_title == "RGB Channels":
            st.caption("Left to right: **Red**, **Green**, **Blue** channel isolated.")
        elif current_title == "HSV Channels":
            st.caption(
                "Left to right: **Hue** (colorized), **Saturation**, "
                "**Value** — each as a separate intensity map."
            )


    # ==================================================
    # PREVIOUS / NEXT / JUMP TO STAGE
    # ==================================================

    st.write("")

    previous_col, counter_col, next_col = st.columns([1, 2, 1])

    with previous_col:
        if st.button("← Previous", key="previous_button", use_container_width=True):
            st.session_state.current_stage = titles[(current_index - 1) % len(titles)]
            st.rerun()

    with counter_col:
        st.markdown(
            f"""
            <div style="text-align: center; padding-top: 8px; font-size: 14px;">
                <b>{current_index + 1} / {len(titles)}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    with next_col:
        if st.button("Next →", key="next_button", use_container_width=True):
            st.session_state.current_stage = titles[(current_index + 1) % len(titles)]
            st.rerun()

    st.write("")

    st.selectbox(
        "📍 Jump to a specific stage",
        options=titles,
        key="current_stage"
    )

else:
    st.info("👆 Upload a JPG or PNG image to get started.")
    