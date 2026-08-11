import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    image = Image.open(uploaded_file)

    image_np = np.array(image)


    # ==================================================
    # IMAGE INFORMATION
    # ==================================================

    image_height, image_width = image_np.shape[:2]

    channels = (
        image_np.shape[2]
        if len(image_np.shape) == 3
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
            help=(
                "The lower boundary for weak edges."
            )
        )

        upper_threshold = st.slider(
            "Upper Threshold",
            min_value=0,
            max_value=255,
            value=150,
            help=(
                "The threshold for strong edges."
            )
        )


        st.divider()


        # --------------------------------------------------
        # IMAGE INFORMATION
        # --------------------------------------------------

        st.subheader("Image Information")

        st.write(
            f"**Resolution:** "
            f"{image_width} × {image_height}"
        )

        st.write(
            f"**Channels:** {channels}"
        )


    # ==================================================
    # GRAYSCALE
    # ==================================================

    if len(image_np.shape) == 3:

        grayscale = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY
        )

    else:

        grayscale = image_np


    # ==================================================
    # BINARY THRESHOLD
    # ==================================================

    _, binary = cv2.threshold(
        grayscale,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )


    # ==================================================
    # GAUSSIAN BLUR
    # ==================================================

    blurred = cv2.GaussianBlur(
        grayscale,
        (kernel_size, kernel_size),
        0
    )


    # ==================================================
    # CANNY EDGE DETECTION
    # ==================================================

    edges = cv2.Canny(
        blurred,
        lower_threshold,
        upper_threshold
    )


    # ==================================================
    # GRAYSCALE HISTOGRAM
    # ==================================================

    histogram = cv2.calcHist(
        [grayscale],
        [0],
        None,
        [256],
        [0, 256]
    )


    # ==================================================
    # HISTOGRAM FIGURE
    # ==================================================

    fig, ax = plt.subplots(
        figsize=(5, 3)
    )

    ax.plot(histogram)

    ax.set_title(
        "Grayscale Histogram"
    )

    ax.set_xlabel(
        "Pixel Intensity"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_xlim(
        [0, 256]
    )

    fig.tight_layout()


    # ==================================================
    # IMAGE GALLERY
    # ==================================================

    images = [
        ("Original Image", image_np),
        ("Grayscale Image", grayscale),
        ("Binary Image", binary),
        ("Gaussian Blur", blurred),
        ("Canny Edge Detection", edges),
        ("Grayscale Histogram", fig)
    ]


    # ==================================================
    # NAVIGATION STATE
    # ==================================================

    if "image_index" not in st.session_state:

        st.session_state.image_index = 0


    current_index = st.session_state.image_index


    if (
        current_index < 0
        or current_index >= len(images)
    ):

        current_index = 0
        st.session_state.image_index = 0


    current_title, current_image = images[
        current_index
    ]


    # ==================================================
    # MAIN DISPLAY
    # ==================================================

    st.divider()

    st.subheader(
        current_title
    )


    # ==================================================
    # DISPLAY CURRENT RESULT
    # ==================================================

    if current_title == "Grayscale Histogram":

        st.pyplot(
            current_image,
            use_container_width=False
        )

    else:

        st.image(
            current_image,
            width=850
        )


    # ==================================================
    # PREVIOUS / NEXT
    # ==================================================

    st.write("")

    previous_col, counter_col, next_col = st.columns(
        [1, 2, 1]
    )


    with previous_col:

        if st.button(
            "← Previous",
            key="previous_button",
            use_container_width=True
        ):

            st.session_state.image_index = (
                current_index - 1
            ) % len(images)

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
                    {current_index + 1} / {len(images)}
                </b>
            </div>
            """,
            unsafe_allow_html=True
        )


    with next_col:

        if st.button(
            "Next →",
            key="next_button",
            use_container_width=True
        ):

            st.session_state.image_index = (
                current_index + 1
            ) % len(images)

            st.rerun()


    # ==================================================
    # CAROUSEL INDICATORS
    # ==================================================

    st.write("")


    # --------------------------------------------------
    # Carousel-specific CSS
    # --------------------------------------------------

    st.markdown(
        """
        <style>

        /* ---------------------------------------------
           Only the carousel row
           --------------------------------------------- */

        .carousel-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;

            gap: 5px;

            height: 25px;
        }


        .carousel-dot {
            font-size: 14px;

            cursor: pointer;

            text-decoration: none;

            line-height: 1;
        }


        .carousel-dot:hover {
            transform: scale(1.2);
        }

        </style>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # Indicator Buttons
    # --------------------------------------------------

    indicator_columns = st.columns(
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        gap=None
    )


    dot_names = [
        "Original Image",
        "Grayscale Image",
        "Binary Image",
        "Gaussian Blur",
        "Canny Edge Detection",
        "Grayscale Histogram"
    ]


    for i in range(len(images)):

        with indicator_columns[i]:

            if i == current_index:

                symbol = "●"

            else:

                symbol = "○"


            if st.button(
                symbol,
                key=f"carousel_dot_{i}",
                help=dot_names[i]
            ):

                st.session_state.image_index = i

                st.rerun()