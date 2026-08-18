"""
Tests for frame-by-frame video processing.

The clips are generated here rather than committed, so the suite stays
self-contained and doesn't carry a binary fixture around.
"""

import cv2
import numpy as np
import pytest

from modules.video import (
    VideoInfo,
    VideoLimits,
    _ordinal,
    describe_plan,
    frames_to_process,
    probe_video,
    process_video,
)


@pytest.fixture
def sample_video(tmp_path):
    """A 30-frame clip with a square that moves across the frame."""

    path = tmp_path / "sample.mp4"

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
    )

    if not writer.isOpened():
        pytest.skip("this OpenCV build has no MPEG-4 encoder")

    for index in range(30):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        x = 2 + index
        cv2.rectangle(frame, (x, 20), (x + 10, 30), (255, 255, 255), -1)
        writer.write(frame)

    writer.release()

    return path


# ==================================================
# PROBING
# ==================================================


def test_probe_reads_the_clip_metadata(sample_video):
    info = probe_video(sample_video)

    assert info.frame_count == 30
    assert info.width == 64
    assert info.height == 64
    assert info.fps == pytest.approx(10.0, abs=0.5)
    assert info.duration_seconds == pytest.approx(3.0, abs=0.2)


def test_probe_rejects_a_file_that_is_not_a_video(tmp_path):
    path = tmp_path / "not_a_video.mp4"
    path.write_bytes(b"this is not a video")

    with pytest.raises(ValueError, match="couldn't be opened"):
        probe_video(path)


def test_duration_is_zero_when_the_frame_rate_is_unknown():
    info = VideoInfo(frame_count=100, fps=0.0, width=64, height=64)

    assert info.duration_seconds == 0.0


# ==================================================
# PLANNING
# ==================================================


def test_frame_count_respects_the_cap():
    info = VideoInfo(frame_count=1000, fps=25, width=64, height=64)

    planned = frames_to_process(info, VideoLimits(max_frames=100, stride=1))

    assert planned == 100


def test_stride_reduces_the_work():
    info = VideoInfo(frame_count=100, fps=25, width=64, height=64)

    planned = frames_to_process(info, VideoLimits(max_frames=500, stride=4))

    assert planned == 25


def test_the_plan_mentions_truncation_before_it_happens():
    info = VideoInfo(frame_count=1000, fps=25, width=1920, height=1080)

    plan = describe_plan(info, VideoLimits(max_frames=100, max_dimension=720))

    assert "100" in plan
    assert "1000" in plan
    assert "covering the first" in plan
    assert "720px" in plan


def test_the_plan_says_nothing_about_truncation_when_there_is_none():
    info = VideoInfo(frame_count=50, fps=25, width=320, height=240)

    plan = describe_plan(info, VideoLimits(max_frames=100, max_dimension=720))

    assert "covering the first" not in plan
    assert "downscaled" not in plan


@pytest.mark.parametrize(
    "number,expected", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th")]
)
def test_ordinals_read_correctly(number, expected):
    assert _ordinal(number) == expected


# ==================================================
# PROCESSING
# ==================================================


def test_every_frame_goes_through_the_callback(sample_video, tmp_path):
    seen = []

    def record(frame):
        seen.append(frame.shape)
        return frame

    result = process_video(
        sample_video,
        record,
        limits=VideoLimits(max_frames=10, stride=1),
        output_dir=tmp_path / "out",
    )

    assert len(seen) == 10
    assert result.frames_written == 10
    assert result.path.exists()


def test_a_grayscale_result_is_written_as_a_video(sample_video, tmp_path):
    """
    Most stages return a single-channel image. A video file can't hold
    one, so the writer has to lift it back to three channels rather
    than failing or writing a corrupt file.
    """

    def to_gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    result = process_video(
        sample_video,
        to_gray,
        limits=VideoLimits(max_frames=5),
        output_dir=tmp_path / "out",
    )

    assert result.frames_written == 5

    written = probe_video(result.path)

    assert written.frame_count >= 5


def test_stride_skips_frames_and_slows_the_output_rate(
    sample_video, tmp_path
):
    result = process_video(
        sample_video,
        lambda frame: frame,
        limits=VideoLimits(max_frames=100, stride=3),
        output_dir=tmp_path / "out",
    )

    # 30 source frames, every third one.
    assert result.frames_written == 10
    # A third of the frames at a third of the rate is the same duration.
    assert result.fps == pytest.approx(10.0 / 3, abs=0.1)


def test_frames_are_downscaled_to_the_limit(sample_video, tmp_path):
    sizes = []

    def record(frame):
        sizes.append(frame.shape[:2])
        return frame

    process_video(
        sample_video,
        record,
        limits=VideoLimits(max_frames=3, max_dimension=32),
        output_dir=tmp_path / "out",
    )

    assert sizes and all(max(size) <= 32 for size in sizes)


def test_progress_is_reported_and_never_exceeds_one(sample_video, tmp_path):
    reported = []

    process_video(
        sample_video,
        lambda frame: frame,
        limits=VideoLimits(max_frames=6),
        output_dir=tmp_path / "out",
        progress_callback=reported.append,
    )

    assert reported
    assert reported == sorted(reported)
    assert max(reported) <= 1.0
    assert reported[-1] == pytest.approx(1.0)


def test_processing_a_non_video_raises(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"nope")

    with pytest.raises(ValueError):
        process_video(
            path, lambda frame: frame, output_dir=tmp_path / "out"
        )
