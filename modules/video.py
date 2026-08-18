"""
Frame-by-frame video processing.

A still image tells you what an operation does; a video tells you
whether it holds up. Detection that flickers on and off between frames,
a threshold that was only ever tuned for one lighting condition, edges
that boil with noise — none of that is visible in a single frame, and
all of it is obvious in a few seconds of footage.

The work here is deliberately bounded. Every frame runs the same stage
handler the still-image path runs, so there is no second implementation
of anything, but a long clip at full resolution would take minutes and
the app would look hung. `VideoLimits` caps the resolution, the frame
count and the sampling stride up front, and `describe_plan` reports
what those caps will actually do to the clip that was uploaded before
any work starts.

Output encoding: OpenCV's pip wheels ship an MPEG-4 encoder but not
H.264, and browsers will not play the former. So the frames are written
with what OpenCV has, and then transcoded with ffmpeg if it is
available — `packages.txt` installs it on Streamlit Community Cloud.
When it isn't available the download button still hands over a
perfectly good file; only the inline preview is lost, and the caller is
told so rather than being shown a silently blank player.
"""

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from modules.image_utils import to_rgb

SUPPORTED_SUFFIXES = ("mp4", "mov", "avi", "mkv")


class VideoLimits(NamedTuple):
    """Caps applied before processing, so a long clip can't hang the app."""

    # Long side of each frame, matching the still-image path's cap in
    # spirit but much lower: this runs once per frame, not once.
    max_dimension: int = 720
    # Hard ceiling on frames actually processed.
    max_frames: int = 240
    # Take every Nth frame. 2 halves the work and, paired with a halved
    # output frame rate, plays back at the original speed.
    stride: int = 1


# A single shared instance, so the caps the app's controls start from
# and the caps process_video() falls back to are literally the same
# values rather than two copies that can drift.
DEFAULT_LIMITS = VideoLimits()


class VideoInfo(NamedTuple):
    """What a clip contains, read from its container metadata."""

    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0

        return self.frame_count / self.fps


def probe_video(path) -> VideoInfo:
    """Read frame count, frame rate and size without decoding the clip."""

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        capture.release()

        raise ValueError(
            "This file couldn't be opened as a video. Supported "
            f"formats: {', '.join(SUPPORTED_SUFFIXES)}."
        )

    try:
        return VideoInfo(
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    finally:
        capture.release()


def frames_to_process(info: VideoInfo, limits: VideoLimits) -> int:
    """How many frames will actually be processed under these limits."""

    if info.frame_count <= 0:
        return 0

    sampled = (info.frame_count + limits.stride - 1) // limits.stride

    return min(sampled, limits.max_frames)


def _ordinal(number: int) -> str:
    """2 -> '2nd', 3 -> '3rd'. Only ever sees small stride values."""

    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")

    return f"{number}{suffix}"


def describe_plan(info: VideoInfo, limits: VideoLimits) -> str:
    """
    A sentence stating what the limits will do to this particular clip,
    shown before processing starts so a truncated result is never a
    surprise.
    """

    planned = frames_to_process(info, limits)
    covered = min(planned * limits.stride, max(info.frame_count, 0))

    parts = [
        f"Processing **{planned}** of {info.frame_count} frames"
    ]

    if limits.stride > 1:
        parts.append(f"taking every {_ordinal(limits.stride)} frame")

    if covered < info.frame_count:
        seconds = covered / info.fps if info.fps > 0 else 0
        parts.append(f"covering the first {seconds:.1f}s")

    if max(info.width, info.height) > limits.max_dimension:
        parts.append(f"downscaled to {limits.max_dimension}px on the long side")

    return ", ".join(parts) + "."


def _resize_to_limit(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    long_side = max(height, width)

    if long_side <= max_dimension:
        return frame

    scale = max_dimension / long_side

    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _transcode_for_browser(source: Path) -> Path | None:
    """
    Re-encode to H.264 so the result plays inline.

    Returns None when ffmpeg isn't installed or the transcode fails —
    the caller falls back to offering the original file for download.
    """

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        return None

    destination = source.with_name(source.stem + "_h264.mp4")

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel", "error",
                "-i", str(source),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                # Browsers refuse odd frame dimensions with yuv420p.
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-movflags", "+faststart",
                str(destination),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )

    except (subprocess.SubprocessError, OSError):
        destination.unlink(missing_ok=True)

        return None

    return destination


class VideoResult(NamedTuple):
    """Where the processed clip landed, and what happened on the way."""

    path: Path
    frames_written: int
    fps: float
    # True when the file is H.264 and can be played inline; False when
    # it is the raw OpenCV output, which browsers won't render.
    browser_playable: bool


def process_video(
    source_path,
    process_frame: Callable[[np.ndarray], np.ndarray],
    limits: VideoLimits = DEFAULT_LIMITS,
    output_dir: Path | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> VideoResult:
    """
    Run `process_frame` over the clip and write the results to a new one.

    `process_frame` takes an RGB frame and returns an RGB or grayscale
    frame of the same height and width — in practice a closure around a
    stage handler, which is what keeps the video path and the still path
    from drifting apart.
    """

    source_path = Path(source_path)

    capture = cv2.VideoCapture(str(source_path))

    if not capture.isOpened():
        capture.release()

        raise ValueError("This file couldn't be opened as a video.")

    output_dir = Path(output_dir or tempfile.mkdtemp(prefix="cvlab_video_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_output = output_dir / "processed.mp4"

    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    # Sampling every Nth frame and dividing the frame rate by N keeps
    # the output the same duration as the input.
    output_fps = max(source_fps / limits.stride, 1.0)

    info = VideoInfo(
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=source_fps,
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    planned = frames_to_process(info, limits) or limits.max_frames

    writer = None
    written = 0
    read_index = 0

    try:
        while written < limits.max_frames:
            ok, frame_bgr = capture.read()

            if not ok:
                break

            if read_index % limits.stride != 0:
                read_index += 1
                continue

            read_index += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_to_limit(frame_rgb, limits.max_dimension)

            processed = to_rgb(process_frame(frame_rgb))

            if writer is None:
                height, width = processed.shape[:2]

                writer = cv2.VideoWriter(
                    str(raw_output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "Couldn't open a video writer. The installed "
                        "OpenCV build may lack an MPEG-4 encoder."
                    )

            writer.write(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            written += 1

            if progress_callback is not None and planned > 0:
                progress_callback(min(written / planned, 1.0))

    finally:
        capture.release()

        if writer is not None:
            writer.release()

    if written == 0:
        raise ValueError("No frames could be read from this video.")

    playable = _transcode_for_browser(raw_output)

    return VideoResult(
        path=playable or raw_output,
        frames_written=written,
        fps=output_fps,
        browser_playable=playable is not None,
    )
