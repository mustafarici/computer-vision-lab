"""The video workspace: one stage applied to every frame of a clip."""

from pathlib import Path

import streamlit as st

from modules.caching import caching_disabled
from modules.stages import STAGES, run_stage
from modules.video import (
    DEFAULT_LIMITS,
    SUPPORTED_SUFFIXES,
    VideoLimits,
    describe_plan,
    probe_video,
    process_video,
)
from modules.workspace import clear_workspace, session_workspace, store_upload

# A histogram figure can't be a video frame, so those stages aren't
# offered here rather than being offered and then failing.
VIDEO_STAGES = {
    name: info for name, info in STAGES.items() if not info.is_figure
}

RESULT_KEY = "video_result"


def _limit_controls() -> VideoLimits:
    first, second, third = st.columns(3)

    with first:
        max_frames = st.slider(
            "Maximum frames", 30, 600, DEFAULT_LIMITS.max_frames, step=30
        )

    with second:
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

    with third:
        max_dimension = st.select_slider(
            "Frame size (long side)",
            options=[360, 480, 720, 1080],
            value=DEFAULT_LIMITS.max_dimension,
        )

    return VideoLimits(
        max_dimension=max_dimension, max_frames=max_frames, stride=stride
    )


def _run(source_path, stage_name, limits, params, output_dir):
    """Process the clip and remember where the result landed."""

    stage_info = VIDEO_STAGES[stage_name]

    progress = st.progress(0.0, text="Processing frames…")

    def process_frame(frame):
        _grayscale, result = run_stage(stage_info, frame, params)

        # A stage that can't produce a result for this frame (an
        # optional model missing, say) leaves the frame untouched
        # rather than aborting the whole clip.
        return result.image if result.image is not None else frame

    try:
        # Every frame is a different array, so @st.cache_data can only
        # ever miss here — and still pays to hash the frame and copy the
        # result back out. Measured at ~31% of the total runtime.
        with st.spinner("Processing…"), caching_disabled():
            result = process_video(
                source_path,
                process_frame,
                limits=limits,
                output_dir=Path(output_dir),
                progress_callback=lambda fraction: progress.progress(
                    fraction, text="Processing frames…"
                ),
            )

    except (ValueError, RuntimeError) as error:
        progress.empty()
        st.error(str(error))
        return

    progress.empty()

    st.session_state[RESULT_KEY] = {
        "path": str(result.path),
        "frames": result.frames_written,
        "fps": result.fps,
        "playable": result.browser_playable,
        "stage": stage_name,
    }


def _render_result():
    stored = st.session_state.get(RESULT_KEY)

    if stored is None or not Path(stored["path"]).exists():
        return

    st.divider()

    st.subheader(f"{stored['stage']} — processed clip")
    st.caption(f"{stored['frames']} frames at {stored['fps']:.0f} fps.")

    video_bytes = Path(stored["path"]).read_bytes()

    if stored["playable"]:
        st.video(video_bytes)
    else:
        st.info(
            "ffmpeg isn't installed here, so the clip couldn't be "
            "converted to a format browsers play inline. The download "
            "below is a valid video file — open it in any media player."
        )

    st.download_button(
        "⬇️ Download processed video",
        data=video_bytes,
        file_name=f"{stored['stage'].lower().replace(' ', '_')}.mp4",
        mime="video/mp4",
    )


def render(params: dict):
    """Draw the whole video workspace."""

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
        return

    workspace = session_workspace()

    # store_upload sanitizes the client-supplied filename; see
    # modules/workspace.py for why that isn't optional.
    source_path = store_upload(
        uploaded_video.getvalue(), uploaded_video.name, workspace
    )

    try:
        info = probe_video(source_path)

    except ValueError as error:
        st.error(str(error))
        return

    detail_col, stage_col = st.columns(2)

    with detail_col:
        st.caption(
            f"**{info.width}×{info.height}**, {info.fps:.0f} fps, "
            f"{info.frame_count} frames ({info.duration_seconds:.1f}s)"
        )

    with stage_col:
        stage_name = st.selectbox(
            "Stage to apply",
            options=list(VIDEO_STAGES),
            key="video_stage",
            format_func=lambda name: (
                f"{name}  ·  {VIDEO_STAGES[name].category}"
            ),
        )

    limits = _limit_controls()

    st.caption(describe_plan(info, limits))

    if st.button("▶️ Process video", type="primary"):
        # The previous run's frames and encodes are worth exactly
        # nothing now, and this is a long-running server: clear before
        # writing rather than accumulating a clip per click.
        output = clear_workspace(Path(workspace) / "output")

        st.session_state.pop(RESULT_KEY, None)

        _run(source_path, stage_name, limits, params, output)

    _render_result()
