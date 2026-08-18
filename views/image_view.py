"""The single-image workspace: upload or camera in, one stage out."""

import numpy as np
import streamlit as st

from modules.compare import (
    compute_difference,
    is_comparable,
    summarize_difference,
)
from modules.histogram import get_figure_download_bytes
from modules.io_utils import (
    get_image_download_bytes,
    is_local_run,
    load_image,
    save_result_locally,
)
from modules.stages import run_stage

VIEW_MODES = ("Result", "Side by side", "Difference")


def read_input(input_mode: str):
    """
    Draw the file uploader or the camera widget and return whatever the
    user supplied, or None.
    """

    if input_mode == "camera":
        return st.camera_input(
            "Take a photo",
            # Not "never uploaded anywhere": the frame is sent to
            # whatever machine is running this app, which is this
            # laptop locally and someone else's server when deployed.
            # Saying otherwise would be a privacy claim the app can't
            # keep.
            help=(
                "The snapshot is sent to the machine running this app "
                "and held in memory for processing — it isn't saved or "
                "sent anywhere else."
            ),
        )

    return st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        help="Upload a PNG, JPG, or JPEG image to start processing.",
    )


def render_image_information(image_np, resize_info, is_color_image):
    """The sidebar panel describing what was loaded."""

    height, width = image_np.shape[:2]
    channels = image_np.shape[2] if image_np.ndim == 3 else 1

    with st.expander("ℹ️ Image Information", expanded=True):

        st.write(f"**Resolution:** {width} × {height}")
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


def decode(uploaded_file):
    """Decode the upload, or explain why it couldn't be read and stop."""

    try:
        return load_image(uploaded_file.getvalue())

    except Exception:
        st.error(
            "This file couldn't be read as an image. "
            "Please upload a valid JPG or PNG file."
        )
        st.stop()


def _choose_view_mode(stage: str, image_np, current_image, stage_info) -> str:
    """
    Offer the comparison modes, but only where there's something to
    compare: a matplotlib figure isn't an image, and the channel
    composites are three frames wide.

    The widget key is per stage, so "show me the difference" applies to
    the stage it was asked about instead of silently following the user
    to the next one.
    """

    comparable = (
        not stage_info.is_figure
        and current_image is not None
        and is_comparable(image_np, current_image)
    )

    if not comparable:
        return "Result"

    return st.radio(
        "View mode",
        options=VIEW_MODES,
        horizontal=True,
        key=f"view_mode::{stage}",
        label_visibility="collapsed",
        help=(
            "Side by side shows the original next to the result. "
            "Difference shows only what this stage changed."
        ),
    )


def _render_result(stage, stage_info, image_np, current_image, extra_info,
                   view_mode, filename):
    """Draw the result and return (bytes to offer, filename)."""

    if stage_info.is_figure:
        st.pyplot(current_image, width="content")

        return get_figure_download_bytes(current_image), filename

    if current_image is None:
        # A stage that can't produce a result (an optional dependency
        # missing, a model that couldn't be fetched) explains itself in
        # extra_info rather than raising.
        st.info(extra_info or "No result is available for this stage.")

        return None, filename

    if view_mode == "Side by side":
        before_col, after_col = st.columns(2)

        with before_col:
            st.caption("**Before** — original")
            st.image(image_np, width="stretch")

        with after_col:
            st.caption(f"**After** — {stage}")
            st.image(current_image, width="stretch")

        if extra_info:
            st.caption(extra_info)

        return get_image_download_bytes(current_image), filename

    if view_mode == "Difference":
        difference = compute_difference(image_np, current_image)

        st.image(difference, width="stretch")
        st.caption(summarize_difference(difference))

        return (
            get_image_download_bytes(difference),
            filename.replace(".png", "_diff.png"),
        )

    st.image(current_image, width="stretch")

    if extra_info:
        st.caption(extra_info)

    return get_image_download_bytes(current_image), filename


def _render_export(result_bytes, filename):
    if result_bytes is None:
        return

    # "Save to results/" only means anything on the machine the user is
    # sitting at; see modules.io_utils.is_local_run.
    if not is_local_run():
        st.download_button(
            "⬇️ Download Result (PNG)",
            data=result_bytes,
            file_name=filename,
            mime="image/png",
            width="stretch",
        )
        return

    download_col, save_col = st.columns(2)

    with download_col:
        st.download_button(
            "⬇️ Download Result (PNG)",
            data=result_bytes,
            file_name=filename,
            mime="image/png",
            width="stretch",
        )

    with save_col:
        if st.button(
            "💾 Save a copy to results/",
            key="save_result_button",
            width="stretch",
        ):
            st.success(f"Saved to {save_result_locally(result_bytes, filename)}")


def _render_statistics(grayscale):
    st.subheader("📊 Image Statistics")

    columns = st.columns(4)

    values = (
        ("Mean", f"{grayscale.mean():.2f}"),
        ("Median", f"{float(np.median(grayscale)):.2f}"),
        ("Minimum", f"{grayscale.min()}"),
        ("Maximum", f"{grayscale.max()}"),
    )

    for column, (label, value) in zip(columns, values, strict=True):
        with column:
            st.metric(label, value)


def render(image_np, stage: str, stage_info, params: dict, is_color_image: bool):
    """Run the selected stage over the loaded image and present it."""

    if stage_info.requires_color and not is_color_image:
        st.divider()
        st.subheader(stage)

        st.warning("⚠️ This stage requires a color image.")
        st.info(
            "Please upload an RGB/color image to use "
            "RGB, HSV, and color thresholding operations."
        )
        st.write("")

        # Navigation stays available even when this stage can't run.
        return

    grayscale, result = run_stage(stage_info, image_np, params)

    st.divider()

    st.subheader(stage)
    st.caption(f"**Category:** {stage_info.category}")
    st.write(stage_info.description)

    with st.expander("🔗 Processing Pipeline", expanded=False):
        st.code(stage_info.pipeline, language="text")

    view_mode = _choose_view_mode(stage, image_np, result.image, stage_info)

    filename = stage.lower().replace(" ", "_").replace("/", "_") + ".png"

    result_bytes, filename = _render_result(
        stage,
        stage_info,
        image_np,
        result.image,
        result.extra_info,
        view_mode,
        filename,
    )

    _render_export(result_bytes, filename)

    if stage_info.shows_statistics:
        _render_statistics(grayscale)
