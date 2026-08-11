import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Computer Vision Lab",
    layout="wide"
)

st.title("🧪 Computer Vision Lab")
st.caption("Interactive image processing laboratory")


# --------------------------------------------------
# Image Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    # --------------------------------------------------
    # Load Original Image
    # --------------------------------------------------

    image = Image.open(uploaded_file)

    # Original image as NumPy array
    image_np = np.array(image)


    # --------------------------------------------------
    # Image Information
    # --------------------------------------------------

    image_height, image_width = image_np.shape[:2]


    # --------------------------------------------------
    # Sidebar - Image Processing Controls
    # --------------------------------------------------

    with st.sidebar:

        st.header("⚙️ Image Processing")

        st.divider()

        st.subheader("Threshold")

        threshold_value = st.slider(
            "Threshold Value",
            min_value=0,
            max_value=255,
            value=127
        )

        st.divider()

        st.subheader("Gaussian Blur")

        kernel_size = st.slider(
            "Kernel Size",
            min_value=1,
            max_value=15,
            value=5,
            step=2
        )

        st.divider()

        st.subheader("Image Information")

        st.write(
            f"**Resolution:** {image_width} × {image_height}"
        )

        st.write(
            f"**Channels:** "
            f"{image_np.shape[2] if len(image_np.shape) == 3 else 1}"
        )


    # --------------------------------------------------
    # Grayscale
    # --------------------------------------------------

    if len(image_np.shape) == 3:

        grayscale = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY
        )

    else:

        grayscale = image_np


    # --------------------------------------------------
    # Binary Threshold
    # --------------------------------------------------

    _, binary = cv2.threshold(
        grayscale,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )


    # --------------------------------------------------
    # Gaussian Blur
    # --------------------------------------------------

    blurred = cv2.GaussianBlur(
        image_np,
        (kernel_size, kernel_size),
        0
    )


    # --------------------------------------------------
    # Histogram
    # --------------------------------------------------

    histogram = cv2.calcHist(
        [grayscale],
        [0],
        None,
        [256],
        [0, 256]
    )


    # Create Histogram Figure

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


    # --------------------------------------------------
    # Gallery
    # --------------------------------------------------

    images = [
        (
            "Original Image",
            image_np
        ),
        (
            "Grayscale Image",
            grayscale
        ),
        (
            "Binary Image",
            binary
        ),
        (
            "Gaussian Blur",
            blurred
        ),
        (
            "Grayscale Histogram",
            fig
        )
    ]


    # --------------------------------------------------
    # Gallery Navigation State
    # --------------------------------------------------

    if "image_index" not in st.session_state:

        st.session_state.image_index = 0


    current_index = (
        st.session_state.image_index
    )

    current_title, current_image = (
        images[current_index]
    )


    # --------------------------------------------------
    # Main Display
    # --------------------------------------------------

    st.divider()

    st.subheader(
        current_title
    )


    # --------------------------------------------------
    # Display Image
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Navigation Buttons
    # --------------------------------------------------

    st.write("")

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )


    with col1:

        if st.button(
            "← Previous",
            use_container_width=True
        ):

            st.session_state.image_index = (
                current_index - 1
            ) % len(images)

            st.rerun()


    with col2:

        st.markdown(
            f"""
            <div style="text-align: center;
                        padding-top: 8px;">
                <b>
                    {current_index + 1} / {len(images)}
                </b>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        if st.button(
            "Next →",
            use_container_width=True
        ):

            st.session_state.image_index = (
                current_index + 1
            ) % len(images)

            st.rerun()


    # --------------------------------------------------
    # Image Indicators
    # --------------------------------------------------

    indicators = ""

    for i in range(len(images)):

        if i == current_index:

            indicators += "● "

        else:

            indicators += "○ "


    st.markdown(
        f"""
        <div style="
            text-align: center;
            margin-top: 10px;
        ">
            {indicators}
        </div>
        """,
        unsafe_allow_html=True
    )