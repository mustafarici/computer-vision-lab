import io

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
# CACHED PROCESSING FUNCTIONS
#
# Each step is cached independently so that moving
# one slider only recomputes the steps that actually
# depend on it, instead of rerunning the full pipeline
# on every interaction.
# ==================================================

@st.cache_data(show_spinner=False)
def load_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into a normalized RGB/RGBA/L numpy array."""

    image = Image.open(io.BytesIO(file_bytes))

    # Normalize unusual modes (palette, CMYK, etc.) to RGB so the
    # rest of the pipeline only ever has to deal with L / RGB / RGBA.
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")

    return np.array(image)


@st.cache_data(show_spinner=False)
def convert_to_grayscale(image_np: np.ndarray) -> np.ndarray:
    """Convert an L / RGB / RGBA image to single-channel grayscale."""

    if image_np.ndim == 2:
        return image_np

    channels = image_np.shape[2]

    if channels == 4:
        return cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)

    if channels == 3:
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Fallback for unexpected channel counts.
    return image_np[:, :, 0]


@st.cache_data(show_spinner=False)
def apply_threshold(grayscale: np.ndarray, threshold_value: int) -> np.ndarray:
    _, binary = cv2.threshold(
        grayscale,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )
    return binary


@st.cache_data(show_spinner=False)
def apply_gaussian_blur(grayscale: np.ndarray, kernel_size: int) -> np.ndarray:
    return cv2.GaussianBlur(
        grayscale,
        (kernel_size, kernel_size),
        0
    )


@st.cache_data(show_spinner=False)
def apply_canny(blurred: np.ndarray, lower: int, upper: int) -> np.ndarray:
    return cv2.Canny(blurred, lower, upper)


@st.cache_data(show_spinner=False)
def apply_sobel(grayscale: np.ndarray, ksize: int) -> np.ndarray:
    """Combined Sobel gradient magnitude (x and y), normalized to uint8."""

    sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = cv2.magnitude(sobel_x, sobel_y)

    return cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)


@st.cache_data(show_spinner=False)
def apply_laplacian(grayscale: np.ndarray, ksize: int) -> np.ndarray:
    """Laplacian edge response, absolute value and normalized to uint8."""

    laplacian = cv2.Laplacian(grayscale, cv2.CV_64F, ksize=ksize)

    return cv2.convertScaleAbs(laplacian)


@st.cache_data(show_spinner=False)
def compute_histogram(grayscale: np.ndarray) -> np.ndarray:
    return cv2.calcHist([grayscale], [0], None, [256], [0, 256])


def build_histogram_figure(histogram: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(histogram)
    ax.set_title("Grayscale Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim([0, 256])

    fig.tight_layout()

    return fig


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
        # IMAGE INFORMATION
        # --------------------------------------------------

        st.subheader("Image Information")

        st.write(f"**Resolution:** {image_width} × {image_height}")
        st.write(f"**Channels:** {channels}")


    # ==================================================
    # PROCESSING PIPELINE
    # ==================================================

    grayscale = convert_to_grayscale(image_np)
    binary = apply_threshold(grayscale, threshold_value)
    blurred = apply_gaussian_blur(grayscale, kernel_size)
    edges = apply_canny(blurred, lower_threshold, upper_threshold)
    sobel = apply_sobel(grayscale, sobel_kernel_size)
    laplacian = apply_laplacian(grayscale, laplacian_kernel_size)
    histogram = compute_histogram(grayscale)
    histogram_figure = build_histogram_figure(histogram)


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
        ("Grayscale Histogram", histogram_figure)
    ]


    # ==================================================
    # NAVIGATION STATE
    # ==================================================

    if "image_index" not in st.session_state:
        st.session_state.image_index = 0

    current_index = st.session_state.image_index

    if current_index < 0 or current_index >= len(images):
        current_index = 0
        st.session_state.image_index = 0

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
    else:
        st.image(current_image, use_container_width=True)


    # ==================================================
    # PREVIOUS / NEXT
    # ==================================================

    st.write("")

    previous_col, counter_col, next_col = st.columns([1, 2, 1])

    with previous_col:
        if st.button("← Previous", key="previous_button", use_container_width=True):
            st.session_state.image_index = (current_index - 1) % len(images)
            st.rerun()

    with counter_col:
        st.markdown(
            f"""
            <div style="text-align: center; padding-top: 8px; font-size: 14px;">
                <b>{current_index + 1} / {len(images)}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    with next_col:
        if st.button("Next →", key="next_button", use_container_width=True):
            st.session_state.image_index = (current_index + 1) % len(images)
            st.rerun()


    # ==================================================
    # CAROUSEL INDICATORS
    # ==================================================

    st.write("")

    st.markdown(
        """
        <style>
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

    indicator_columns = st.columns(
        [0.01] * len(images),
        gap=None
    )

    dot_names = [title for title, _ in images]

    for i in range(len(images)):
        with indicator_columns[i]:
            symbol = "●" if i == current_index else "○"

            if st.button(symbol, key=f"carousel_dot_{i}", help=dot_names[i]):
                st.session_state.image_index = i
                st.rerun()

else:
    st.info("👆 Upload a JPG or PNG image to get started.")