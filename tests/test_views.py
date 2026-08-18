"""
Tests for the UI layer.

Everything under views/ was, until these existed, the only part of the
project with no coverage at all — which is also the part that breaks
in ways a user notices immediately: a widget that raises takes the
whole page down with a traceback.

Streamlit's own `AppTest` runs the real script in-process, so these
exercise the actual app.py and the actual view functions rather than a
reimplementation of them. What they can't do is drive a file uploader,
so the image view is reached through a small harness that hands it an
array directly — the same trick the sidebar tests use.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from modules.stages import STAGES

APP = str(Path(__file__).resolve().parent.parent / "app.py")

# Renders the image workspace against a synthetic image, standing in
# for an upload that AppTest has no way to perform.
IMAGE_HARNESS = """
import cv2
import numpy as np
import streamlit as st

from modules.controls import render_sidebar, active_categories_for
from modules.stages import STAGES
from views import image_view, navigation

image = np.zeros((64, 64, 3), dtype=np.uint8)
cv2.rectangle(image, (10, 10), (30, 30), (255, 255, 255), -1)
cv2.rectangle(image, (40, 40), (55, 55), (255, 0, 0), -1)

stage = st.session_state.get("stage", "Original Image")
stage_info = STAGES[stage]

with st.sidebar:
    params = render_sidebar(
        active_categories_for(stage_info.category, st.session_state)
    )
    image_view.render_image_information(image, None, True)

image_view.render(image, stage, stage_info, params, True)
navigation.render(stage)
"""


# ==================================================
# THE APP ITSELF
# ==================================================


def test_the_app_starts_and_asks_for_input():
    app = AppTest.from_file(APP, default_timeout=30).run()

    assert not app.exception
    assert any("Upload a JPG or PNG" in info.value for info in app.info)


@pytest.mark.parametrize("source", ["🖼️ Image", "📷 Camera", "🎬 Video"])
def test_every_input_mode_renders(source):
    app = AppTest.from_file(APP, default_timeout=30)
    app.session_state["input_source"] = source
    app.run()

    assert not app.exception


def test_the_app_survives_a_stage_left_over_from_a_previous_version():
    """
    Session state outlives a deploy. A stage name that no longer exists
    must reset rather than raise a KeyError on the first line.
    """

    app = AppTest.from_file(APP, default_timeout=30)
    app.session_state["current_stage"] = "Stage That Was Deleted"
    app.run()

    assert not app.exception
    assert app.session_state["current_stage"] in STAGES


# ==================================================
# THE IMAGE WORKSPACE
# ==================================================


@pytest.mark.parametrize(
    "stage_name",
    # Every stage that doesn't need an optional package or a colour
    # image the harness can't provide.
    [
        name
        for name, stage in STAGES.items()
        if stage.category != "Deep Learning"
    ],
)
def test_the_image_view_renders_every_stage(stage_name):
    app = AppTest.from_string(IMAGE_HARNESS, default_timeout=30)
    app.session_state["stage"] = stage_name
    app.run()

    assert not app.exception
    assert any(stage_name in header.value for header in app.subheader)


def test_the_result_can_be_downloaded():
    app = AppTest.from_string(IMAGE_HARNESS, default_timeout=30).run()

    assert not app.exception
    # download_button is its own element type in AppTest, not a button.
    assert any(
        "Download" in element.label
        for element in app.get("download_button")
    )


def test_comparison_modes_are_offered_for_a_comparable_stage():
    app = AppTest.from_string(IMAGE_HARNESS, default_timeout=30)
    app.session_state["stage"] = "Grayscale Image"
    app.run()

    modes = [radio for radio in app.radio if radio.label == "View mode"]

    assert modes
    assert set(modes[0].options) == {"Result", "Side by side", "Difference"}


def test_comparison_modes_are_not_offered_for_a_figure():
    """A histogram is a chart; there's no "before" to lay beside it."""

    app = AppTest.from_string(IMAGE_HARNESS, default_timeout=30)
    app.session_state["stage"] = "Grayscale Histogram"
    app.run()

    assert not [radio for radio in app.radio if radio.label == "View mode"]


def test_comparison_modes_are_not_offered_for_a_wider_composite():
    app = AppTest.from_string(IMAGE_HARNESS, default_timeout=30)
    app.session_state["stage"] = "RGB Channels"
    app.run()

    assert not [radio for radio in app.radio if radio.label == "View mode"]


@pytest.mark.parametrize("mode", ["Side by side", "Difference"])
def test_each_comparison_mode_renders(mode):
    app = AppTest.from_string(IMAGE_HARNESS, default_timeout=30)
    app.session_state["stage"] = "Canny Edge Detection"
    app.run()

    app.radio(key="view_mode::Canny Edge Detection").set_value(mode).run()

    assert not app.exception


def test_the_view_mode_belongs_to_the_stage_that_set_it():
    """
    "Show me the difference" is a statement about the stage in front of
    you. Carrying it to the next stage silently reinterprets it — which
    is exactly what a single shared widget key would do.
    """

    app = AppTest.from_string(IMAGE_HARNESS, default_timeout=30)
    app.session_state["stage"] = "Canny Edge Detection"
    app.run()

    app.radio(key="view_mode::Canny Edge Detection").set_value(
        "Difference"
    ).run()

    app.session_state["stage"] = "Grayscale Image"
    app.run()

    assert app.radio(key="view_mode::Grayscale Image").value == "Result"


def test_statistics_are_shown_for_the_histogram_stage():
    app = AppTest.from_string(IMAGE_HARNESS, default_timeout=30)
    app.session_state["stage"] = "Grayscale Histogram"
    app.run()

    labels = {metric.label for metric in app.metric}

    assert {"Mean", "Median", "Minimum", "Maximum"} <= labels


# ==================================================
# NAVIGATION
# ==================================================


def test_next_moves_to_the_following_stage():
    app = AppTest.from_string(IMAGE_HARNESS, default_timeout=30).run()

    app.button(key="next_button").click().run()

    titles = list(STAGES)

    assert app.session_state["current_stage"] == titles[1]


def test_previous_wraps_around_from_the_first_stage():
    app = AppTest.from_string(IMAGE_HARNESS, default_timeout=30).run()

    app.button(key="previous_button").click().run()

    assert app.session_state["current_stage"] == list(STAGES)[-1]


# ==================================================
# THE VIDEO WORKSPACE
# ==================================================
#
# AppTest can't drive a file uploader, so render() returns early and
# the parts worth checking are reached directly. modules/video.py has
# its own tests for the processing itself; these cover the widgets.

VIDEO_LIMITS_HARNESS = """
import streamlit as st
from views import video_view

st.session_state["limits"] = video_view._limit_controls()
"""

VIDEO_RESULT_HARNESS = """
import streamlit as st
from views import video_view

video_view._render_result()
"""


def test_the_limit_controls_default_to_the_declared_limits():
    from modules.video import DEFAULT_LIMITS

    app = AppTest.from_string(VIDEO_LIMITS_HARNESS, default_timeout=30).run()

    assert not app.exception
    assert app.session_state["limits"] == DEFAULT_LIMITS


def test_a_processed_clip_is_offered_for_download(tmp_path):
    clip = tmp_path / "processed.mp4"
    clip.write_bytes(b"not really a video, but bytes on disk")

    app = AppTest.from_string(VIDEO_RESULT_HARNESS, default_timeout=30)
    app.session_state["video_result"] = {
        "path": str(clip),
        "frames": 42,
        "fps": 15.0,
        "playable": True,
        "stage": "Canny Edge Detection",
    }
    app.run()

    assert not app.exception
    assert any(
        "Download" in element.label
        for element in app.get("download_button")
    )


def test_an_unplayable_clip_says_so_instead_of_showing_a_blank_player():
    """
    Without ffmpeg the output is a valid file browsers won't render.
    Showing an empty player and no explanation reads as a bug.
    """

    app = AppTest.from_string(VIDEO_RESULT_HARNESS, default_timeout=30)
    app.session_state["video_result"] = {
        "path": __file__,  # any file that exists
        "frames": 1,
        "fps": 1.0,
        "playable": False,
        "stage": "Grayscale Image",
    }
    app.run()

    assert not app.exception
    assert any("ffmpeg" in info.value for info in app.info)


def test_a_result_whose_file_has_been_swept_is_not_rendered():
    """
    Workspaces are cleaned up; session state outlives them. A stale
    path must be ignored rather than raising FileNotFoundError.
    """

    app = AppTest.from_string(VIDEO_RESULT_HARNESS, default_timeout=30)
    app.session_state["video_result"] = {
        "path": "/tmp/definitely_not_here/processed.mp4",
        "frames": 1,
        "fps": 1.0,
        "playable": True,
        "stage": "Grayscale Image",
    }
    app.run()

    assert not app.exception
    assert not app.get("download_button")
