import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from modules.basic_ops import convert_to_grayscale
from modules.compare import (
    compute_difference,
    is_comparable,
    summarize_difference,
)
from modules.controls import render_sidebar
from modules.histogram import get_figure_download_bytes
from modules.io_utils import (
    get_image_download_bytes,
    load_image,
    save_result_locally,
)
from modules.stages import STAGES
from modules.video import (
    DEFAULT_LIMITS,
    SUPPORTED_SUFFIXES,
    VideoLimits,
    describe_plan,
    probe_video,
    process_video,
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
# SHARED HELPERS
# ==================================================


def run_stage(stage_info, image_np, params):
    """
    Run one stage over one image.

    The single-image view and the video view both go through here, so
    a frame of video is processed by exactly the same code as a still —
    there is no second, subtly different implementation to keep in sync.

    Returns (grayscale, result); the grayscale is handed back because
    the histogram stage reports statistics over it and would otherwise
    recompute it.
    """

    grayscale = (
        convert_to_grayscale(image_np) if stage_info.needs_grayscale else None
    )

    return grayscale, stage_info.handler(image_np, grayscale, params)


def stage_selector(label: str, key: str, eligible: dict):
    """A stage dropdown that shows the category alongside the name."""

    return st.selectbox(
        label,
        options=list(eligible),
        key=key,
        format_func=lambda name: f"{name}  ·  {eligible[name].category}",
    )


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
# the sidebar knows which section to expand.
if (
    "current_stage" not in st.session_state
    or st.session_state.current_stage not in STAGES
):
    st.session_state.current_stage = "Original Image"

stage = st.session_state.current_stage
stage_info = STAGES[stage]


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("⚙️ Image Processing")
    st.divider()

    # Controls that affect the stage currently on screen are
    # expanded; everything else starts collapsed.
    params = render_sidebar(active_category=stage_info.category)

    IMAGE_INFO_SLOT = st.container()


# ==================================================
# VIDEO MODE
# ==================================================

if input_mode == "video":

    # A histogram figure can't be a video frame, so those stages aren't
    # offered here rather than being offered and then failing.
    VIDEO_STAGES = {
        name: info for name, info in STAGES.items() if not info.is_figure
    }

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=list(SUPPORTED_SUFFIXES),
        help=(
            "Every frame runs through the stage you pick below. Long "
            "clips are capped — the exact plan is shown before you start."
        ),
    )

    if uploaded_video is None:
        st.info("👆 Upload a short clip to process it frame by frame.")
        st.stop()

    source_path = (
        Path(tempfile.gettempdir()) / f"cvlab_input_{uploaded_video.name}"
    )
    source_path.write_bytes(uploaded_video.getvalue())

    try:
        info = probe_video(source_path)

    except ValueError as error:
        st.error(str(error))
        st.stop()

    detail_col, cap_col = st.columns(2)

    with detail_col:
        st.caption(
            f"**{info.width}×{info.height}**, {info.fps:.0f} fps, "
            f"{info.frame_count} frames "
            f"({info.duration_seconds:.1f}s)"
        )

    with cap_col:
        video_stage = stage_selector(
            "Stage to apply", "video_stage", VIDEO_STAGES
        )

    limit_col1, limit_col2, limit_col3 = st.columns(3)

    with limit_col1:
        max_frames = st.slider(
            "Maximum frames", 30, 600, DEFAULT_LIMITS.max_frames, step=30
        )

    with limit_col2:
        stride = st.slider(
            "Take every Nth frame",
            1,
            5,
            DEFAULT_LIMITS.stride,
            help=(
                "Higher values process less of the clip while still "
                "covering its whole length, and play back at the same "
                "speed."
            ),
        )

    with limit_col3:
        max_dimension = st.select_slider(
            "Frame size (long side)",
            options=[360, 480, 720, 1080],
            value=DEFAULT_LIMITS.max_dimension,
        )

    limits = VideoLimits(
        max_dimension=max_dimension, max_frames=max_frames, stride=stride
    )

    st.caption(describe_plan(info, limits))

    if st.button("▶️ Process video", type="primary"):

        video_stage_info = VIDEO_STAGES[video_stage]

        progress = st.progress(0.0, text="Processing frames…")

        def process_frame(frame):
            _grayscale, result = run_stage(video_stage_info, frame, params)

            # A stage that can't produce a result for this frame (an
            # optional model missing, say) leaves the frame untouched
            # rather than aborting the whole clip.
            return result.image if result.image is not None else frame

        try:
            with st.spinner("Processing…"):
                result = process_video(
                    source_path,
                    process_frame,
                    limits=limits,
                    progress_callback=lambda fraction: progress.progress(
                        fraction, text="Processing frames…"
                    ),
                )

        except (ValueError, RuntimeError) as error:
            progress.empty()
            st.error(str(error))
            st.stop()

        progress.empty()

        st.session_state["video_result"] = {
            "path": str(result.path),
            "frames": result.frames_written,
            "fps": result.fps,
            "playable": result.browser_playable,
            "stage": video_stage,
        }

    stored = st.session_state.get("video_result")

    if stored is not None and Path(stored["path"]).exists():

        st.divider()

        st.subheader(f"{stored['stage']} — processed clip")

        st.caption(
            f"{stored['frames']} frames at {stored['fps']:.0f} fps."
        )

        video_bytes = Path(stored["path"]).read_bytes()

        if stored["playable"]:
            st.video(video_bytes)
        else:
            st.info(
                "ffmpeg isn't installed here, so the clip couldn't be "
                "converted to a format browsers play inline. The "
                "download below is a valid video file — open it in any "
                "media player."
            )

        st.download_button(
            "⬇️ Download processed video",
            data=video_bytes,
            file_name=f"{stored['stage'].lower().replace(' ', '_')}.mp4",
            mime="video/mp4",
        )

    st.stop()


# ==================================================
# IMAGE / CAMERA INPUT
# ==================================================

if input_mode == "camera":
    uploaded_file = st.camera_input(
        "Take a photo",
        help="The snapshot is processed in memory and never uploaded anywhere.",
    )

else:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        help="Upload a PNG, JPG, or JPEG image to start processing.",
    )


if uploaded_file is None:
    st.info(
        "📷 Allow camera access and take a photo to get started."
        if input_mode == "camera"
        else "👆 Upload a JPG or PNG image to get started."
    )
    st.stop()


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

channels = image_np.shape[2] if image_np.ndim == 3 else 1

is_color_image = image_np.ndim == 3 and image_np.shape[2] >= 3


with IMAGE_INFO_SLOT, st.expander("ℹ️ Image Information", expanded=True):

    st.write(f"**Resolution:** {image_width} × {image_height}")
    st.write(f"**Channels:** {channels}")
    st.write(
        f"**Image Type:** {'Color' if is_color_image else 'Grayscale'}"
    )

    if resize_info is not None:
        st.caption(
            f"Original: {resize_info['original_width']}×"
            f"{resize_info['original_height']} "
            f"(resized to {resize_info['new_width']}×"
            f"{resize_info['new_height']} for performance)"
        )


# ==================================================
# COLOR VALIDATION
# ==================================================
#
# The stage registry (metadata + processing handler for every stage)
# lives in modules/stages.py, and the sidebar controls are declared in
# modules/controls.py. Adding a stage or a parameter means one new
# entry in one of those files, rather than edits scattered through
# this one.

if stage_info.requires_color and not is_color_image:

    st.divider()

    st.subheader(stage)

    st.warning("⚠️ This stage requires a color image.")

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

    grayscale, result = run_stage(stage_info, image_np, params)

    current_image = result.image
    extra_info = result.extra_info


    # ==================================================
    # MAIN DISPLAY
    # ==================================================

    st.divider()

    st.subheader(stage)

    st.caption(f"**Category:** {stage_info.category}")

    st.write(stage_info.description)

    with st.expander("🔗 Processing Pipeline", expanded=False):
        st.code(stage_info.pipeline, language="text")


    # ==================================================
    # VIEW MODE
    # ==================================================
    #
    # Offered only when there is something to compare against: a
    # figure isn't an image, and the channel-composite stages return a
    # canvas three frames wide that can't be laid over the original.

    can_compare = (
        not stage_info.is_figure
        and current_image is not None
        and is_comparable(image_np, current_image)
    )

    view_mode = "Result"

    if can_compare:
        view_mode = st.radio(
            "View mode",
            options=("Result", "Side by side", "Difference"),
            horizontal=True,
            key="view_mode",
            label_visibility="collapsed",
            help=(
                "Side by side shows the original next to the result. "
                "Difference shows only what this stage changed."
            ),
        )


    # ==================================================
    # DISPLAY CURRENT RESULT
    # ==================================================

    download_filename = (
        stage.lower().replace(" ", "_").replace("/", "_") + ".png"
    )

    result_bytes = None

    if stage_info.is_figure:

        st.pyplot(current_image, width="content")

        result_bytes = get_figure_download_bytes(current_image)

    elif current_image is None:

        # Stages that can't produce a result (an optional
        # dependency missing, a model that couldn't be fetched)
        # explain why in extra_info rather than raising.
        st.info(extra_info or "No result is available for this stage.")

    elif view_mode == "Side by side":

        before_col, after_col = st.columns(2)

        with before_col:
            st.caption("**Before** — original")
            st.image(image_np, width="stretch")

        with after_col:
            st.caption(f"**After** — {stage}")
            st.image(current_image, width="stretch")

        if extra_info:
            st.caption(extra_info)

        result_bytes = get_image_download_bytes(current_image)

    elif view_mode == "Difference":

        difference = compute_difference(image_np, current_image)

        st.image(difference, width="stretch")

        st.caption(summarize_difference(difference))

        download_filename = download_filename.replace(".png", "_diff.png")

        result_bytes = get_image_download_bytes(difference)

    else:

        st.image(current_image, width="stretch")

        if extra_info:
            st.caption(extra_info)

        result_bytes = get_image_download_bytes(current_image)


    if result_bytes is not None:

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

    if stage_info.shows_statistics:

        st.subheader("📊 Image Statistics")

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            st.metric("Mean", f"{grayscale.mean():.2f}")

        with stat_col2:
            st.metric("Median", f"{float(np.median(grayscale)):.2f}")

        with stat_col3:
            st.metric("Minimum", f"{grayscale.min()}")

        with stat_col4:
            st.metric("Maximum", f"{grayscale.max()}")


# ==================================================
# NAVIGATION
# ==================================================

st.write("")

stage_titles = list(STAGES.keys())

current_index = stage_titles.index(stage)


previous_col, counter_col, next_col = st.columns([1, 2, 1])


with previous_col:

    if st.button("← Previous", key="previous_button", width="stretch"):

        st.session_state.current_stage = stage_titles[
            (current_index - 1) % len(stage_titles)
        ]

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

    if st.button("Next →", key="next_button", width="stretch"):

        st.session_state.current_stage = stage_titles[
            (current_index + 1) % len(stage_titles)
        ]

        st.rerun()


st.write("")

st.selectbox(
    "📍 Jump to a specific stage",
    options=stage_titles,
    key="current_stage",
)
