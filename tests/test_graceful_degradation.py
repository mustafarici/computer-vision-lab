"""
What the app does when the machine it's running on is not the machine
it was developed on.

Both production breakages in this project's history were of this kind
and neither was catchable by a normal unit test, because in a working
environment there is nothing to catch: OpenCV 5 removed
`cv2.CascadeClassifier` (import still succeeds), and MediaPipe's native
library needs an OpenGL ES runtime that `pip install` does not provide
(the install still succeeds).

So these tests ask a different question: given that some part of the
environment is missing, does the app report it, or does it fall over?
A stage that can't run should cost you that stage. It should never cost
you the page.

The `ml-without-system-libraries` CI job runs the whole suite with the
ML packages installed and the OpenGL libraries deliberately left out,
which is the environment these describe.
"""

import pytest

from modules import stages
from modules.controls import default_params
from modules.deep_detection import _native_library_error
from modules.stages import STAGES, StageResult, run_stage

DEFAULT_PARAMS = default_params()

OPTIONAL_STAGES = ["YOLO Object Detection", "MediaPipe Landmarks"]


@pytest.mark.parametrize("stage_name", OPTIONAL_STAGES)
def test_an_optional_stage_never_raises(stage_name, color_image):
    """
    Whatever the state of the machine — package missing, native library
    missing, model undownloadable — asking for one of these stages
    returns a StageResult. Streamlit renders an exception as a
    full-page traceback, so "raises" and "the app is down" are the same
    event here.
    """

    _grayscale, result = run_stage(
        STAGES[stage_name], color_image, DEFAULT_PARAMS
    )

    assert isinstance(result, StageResult)


@pytest.mark.parametrize("stage_name", OPTIONAL_STAGES)
def test_a_broken_install_leaves_the_other_stages_working(
    monkeypatch, stage_name, color_image, grayscale_image
):
    """
    One stage failing must not be able to take the rest with it — the
    32 classical stages have nothing to do with PyTorch or OpenGL.
    """

    def explode(*args, **kwargs):
        raise _native_library_error("mediapipe", OSError("libGLESv2.so.2"))

    monkeypatch.setattr(stages, "is_yolo_available", lambda: True)
    monkeypatch.setattr(stages, "is_mediapipe_available", lambda: True)
    monkeypatch.setattr(stages, "apply_yolo_detection", explode)
    monkeypatch.setattr(stages, "apply_mediapipe_landmarks", explode)

    broken = STAGES[stage_name].handler(color_image, None, DEFAULT_PARAMS)

    assert broken.image is None

    for name, stage in STAGES.items():
        if name in OPTIONAL_STAGES or stage.requires_color:
            continue

        result = stage.handler(color_image, grayscale_image, DEFAULT_PARAMS)

        assert isinstance(result, StageResult), name


def test_an_unavailable_stage_says_what_to_do_about_it(color_image):
    """
    A blank result is only marginally better than a crash. Whatever the
    reason a deep-learning stage can't run, the caption has to name a
    concrete next step.
    """

    for stage_name in OPTIONAL_STAGES:
        _grayscale, result = run_stage(
            STAGES[stage_name], color_image, DEFAULT_PARAMS
        )

        if result.image is not None:
            continue  # It ran; nothing to explain.

        assert result.extra_info
        assert any(
            hint in result.extra_info
            for hint in ("requirements-ml.txt", "libgles2", "download")
        ), result.extra_info
