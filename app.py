"""
Computer Vision Lab — Streamlit entry point.

This file does four things and nothing else: configure the page, ask
where the image is coming from, draw the sidebar, and hand off to a
view. The processing lives in modules/ and the rendering lives in
views/, so neither has to be read to understand the other.

Run it with:

    streamlit run app.py
"""

import streamlit as st

from modules.controls import active_categories_for, render_sidebar
from modules.stages import STAGES
from views import image_view, navigation, video_view

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Computer Vision Lab",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Computer Vision Lab")
st.caption("Interactive image processing laboratory")


# ==================================================
# INPUT SOURCE
# ==================================================

INPUT_SOURCES = {
    "🖼️ Image": "image",
    "📷 Camera": "camera",
    "🎬 Video": "video",
}

source_label = st.radio(
    "Input source",
    options=list(INPUT_SOURCES),
    horizontal=True,
    key="input_source",
    help=(
        "Process an uploaded picture, a snapshot from your webcam, or "
        "run one stage over every frame of a video clip."
    ),
)

input_mode = INPUT_SOURCES[source_label]


# The current stage has to be resolved before the sidebar renders, so
# the sidebar knows which sections to open.
stage = navigation.current_stage()
stage_info = STAGES[stage]


# ==================================================
# SIDEBAR
# ==================================================
#
# Controls are declared as data in modules/controls.py. The sections
# relevant to what's on screen are expanded and the rest collapsed —
# and for a custom pipeline, "relevant" means the sections its chosen
# steps take their settings from, which is why the previous run's
# selection (held in session state) decides what opens.

with st.sidebar:

    st.header("⚙️ Image Processing")
    st.divider()

    params = render_sidebar(
        active_categories_for(stage_info.category, st.session_state)
    )

    image_information = st.container()


# ==================================================
# DISPATCH
# ==================================================

if input_mode == "video":
    video_view.render(params)

else:
    uploaded_file = image_view.read_input(input_mode)

    if uploaded_file is None:
        st.info(
            "📷 Allow camera access and take a photo to get started."
            if input_mode == "camera"
            else "👆 Upload a JPG or PNG image to get started."
        )
        st.stop()

    image_np, resize_info = image_view.decode(uploaded_file)

    is_color_image = image_np.ndim == 3 and image_np.shape[2] >= 3

    with image_information:
        image_view.render_image_information(
            image_np, resize_info, is_color_image
        )

    image_view.render(image_np, stage, stage_info, params, is_color_image)

    navigation.render(stage)
