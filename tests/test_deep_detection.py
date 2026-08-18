"""
Tests for the optional deep-learning stages.

These are written so the suite passes whether or not ultralytics and
mediapipe are installed: the pure-Python helpers and the "dependency
missing" behaviour are always tested, and the parts that need a real
model are skipped when it isn't available.
"""

from pathlib import Path

import numpy as np
import pytest

from modules import stages
from modules.deep_detection import (
    MEDIAPIPE_TASKS,
    YOLO_MODELS,
    _as_rgb,
    _class_color,
    apply_mediapipe_landmarks,
    apply_yolo_detection,
    download_model,
    is_mediapipe_available,
    is_yolo_available,
)

SAMPLE_IMAGE = (
    Path(__file__).resolve().parent.parent
    / "images"
    / "NASA_Astronaut_Group_15.jpg"
)


# ==================================================
# HELPERS
# ==================================================


def test_as_rgb_expands_grayscale():
    grayscale = np.zeros((10, 10), dtype=np.uint8)

    assert _as_rgb(grayscale).shape == (10, 10, 3)


def test_as_rgb_drops_alpha():
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)

    assert _as_rgb(rgba).shape == (10, 10, 3)


def test_class_color_is_deterministic_and_in_range():
    first = _class_color(3)

    assert first == _class_color(3)
    assert len(first) == 3
    assert all(0 <= channel <= 255 for channel in first)


def test_class_color_separates_neighbouring_classes():
    assert _class_color(0) != _class_color(1)


# ==================================================
# SPEC TABLES
# ==================================================


def test_yolo_model_table_is_well_formed():
    assert YOLO_MODELS

    for label, weights in YOLO_MODELS.items():
        assert label.strip()
        assert weights.endswith(".pt")


def test_mediapipe_task_table_is_well_formed():
    required = {
        "task",
        "filename",
        "url",
        "result_field",
        "connections",
        "point_radius",
    }

    for name, spec in MEDIAPIPE_TASKS.items():
        assert required <= set(spec), name
        assert spec["url"].startswith("https://"), name
        assert spec["filename"].endswith(".task"), name


# ==================================================
# INPUT VALIDATION
# ==================================================


def test_unknown_yolo_model_raises(color_image):
    with pytest.raises(ValueError, match="Unknown model"):
        apply_yolo_detection(color_image, "YOLOv99 Ultra", 0.25, 0.45)


def test_unknown_mediapipe_task_raises(color_image):
    with pytest.raises(ValueError, match="Unknown MediaPipe task"):
        apply_mediapipe_landmarks(color_image, "Elbow Mesh")


# ==================================================
# MODEL DOWNLOAD
# ==================================================


def test_download_model_reuses_an_existing_file(tmp_path):
    """An already-downloaded model must not be fetched again."""

    destination = tmp_path / "already_here.task"
    destination.write_bytes(b"pretend model")

    result = download_model("https://invalid.invalid/model.task", destination)

    assert result == destination
    assert destination.read_bytes() == b"pretend model"


def test_download_model_reports_a_useful_error(tmp_path):
    destination = tmp_path / "missing.task"

    with pytest.raises(RuntimeError, match="download the MediaPipe model"):
        download_model("https://invalid.invalid/model.task", destination)

    # A failed attempt must not leave a partial file behind that would
    # look like a valid model on the next run.
    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []


# ==================================================
# GRACEFUL DEGRADATION
# ==================================================


def test_yolo_stage_explains_how_to_install_when_missing(monkeypatch, color_image):
    # stages.py imported the checker by name, so that's the reference
    # the handler actually calls.
    monkeypatch.setattr(stages, "is_yolo_available", lambda: False)

    result = STAGE_YOLO.handler(color_image, None, {})

    assert result.image is None
    assert "requirements-ml.txt" in result.extra_info


def test_mediapipe_stage_explains_how_to_install_when_missing(
    monkeypatch, color_image
):
    monkeypatch.setattr(stages, "is_mediapipe_available", lambda: False)

    result = STAGE_MEDIAPIPE.handler(color_image, None, {})

    assert result.image is None
    assert "requirements-ml.txt" in result.extra_info


STAGE_YOLO = stages.STAGES["YOLO Object Detection"]
STAGE_MEDIAPIPE = stages.STAGES["MediaPipe Landmarks"]


# ==================================================
# REAL INFERENCE
# ==================================================


@pytest.mark.skipif(not is_yolo_available(), reason="ultralytics not installed")
@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="sample image missing")
def test_yolo_detects_people_in_the_sample_photo():
    """
    End-to-end check against a real photo. This is what would catch the
    RGB/BGR mix-up: ultralytics treats numpy input as BGR, so feeding
    it our RGB array unconverted quietly degrades every detection.
    """

    from PIL import Image

    image = np.array(Image.open(SAMPLE_IMAGE).convert("RGB"))

    canvas, count, summary = apply_yolo_detection(
        image, next(iter(YOLO_MODELS)), 0.25, 0.45
    )

    assert count > 5, "expected a group photo to yield several people"
    assert "person" in summary
    assert canvas.shape == image.shape
    assert canvas.dtype == np.uint8
    # Boxes are drawn on a copy, never on the caller's array.
    assert not np.array_equal(canvas, image)


@pytest.mark.skipif(not is_yolo_available(), reason="ultralytics not installed")
def test_yolo_handles_a_grayscale_image(grayscale_only_image):
    canvas, count, _ = apply_yolo_detection(
        grayscale_only_image, next(iter(YOLO_MODELS)), 0.25, 0.45
    )

    assert canvas.shape == (*grayscale_only_image.shape, 3)
    assert count >= 0


@pytest.mark.skipif(
    not is_mediapipe_available(), reason="mediapipe not installed"
)
@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="sample image missing")
@pytest.mark.parametrize("task_name", list(MEDIAPIPE_TASKS))
def test_mediapipe_runs_every_task_end_to_end(task_name):
    """
    Load the real model, run it on a real photo, draw the result.

    The assertions are deliberately about structure rather than about
    how many faces MediaPipe ought to find in this particular
    photograph — the failure this guards against is the whole Tasks API
    path breaking (MediaPipe removed the old `mp.solutions` API in
    0.10.30, which is exactly the kind of change that silently invalidates
    this module), not a drop in detection quality.
    """

    from PIL import Image

    image = np.array(Image.open(SAMPLE_IMAGE).convert("RGB"))

    try:
        canvas, count, summary = apply_mediapipe_landmarks(image, task_name)

    except RuntimeError as error:
        # The .task bundles are downloaded on first use. No network,
        # no test — but say so rather than reporting a pass.
        pytest.skip(f"MediaPipe model unavailable: {error}")

    assert canvas.shape == image.shape
    assert canvas.dtype == np.uint8
    assert count >= 0
    assert summary

    if count > 0:
        # Landmarks were drawn onto a copy, never the caller's array.
        assert not np.array_equal(canvas, image)
        assert np.array_equal(
            image, np.array(Image.open(SAMPLE_IMAGE).convert("RGB"))
        )
