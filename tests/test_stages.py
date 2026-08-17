"""
Integration tests for the modules/stages.py registry — the piece
app.py's refactor introduced to replace the old if/elif chain.

These exercise every stage handler end-to-end with a representative
set of parameters, to make sure the refactor didn't silently change
behavior or drop a stage.
"""

import pytest

from modules.controls import default_params
from modules.stages import STAGES, StageResult

EXPECTED_STAGE_COUNT = 32

# Built from the sidebar schema rather than restated here, so a
# parameter added to modules/controls.py is automatically exercised
# against every handler — and a handler reading a key the sidebar
# never defines fails these tests with a KeyError.
DEFAULT_PARAMS = default_params()


def test_registry_has_all_expected_stages():
    assert len(STAGES) == EXPECTED_STAGE_COUNT

    for stage in STAGES.values():
        assert stage.category
        assert stage.description
        assert stage.pipeline
        assert callable(stage.handler)


@pytest.mark.parametrize("stage_name", list(STAGES.keys()))
def test_every_stage_handler_runs_without_error(
    stage_name, color_image, grayscale_image
):
    stage = STAGES[stage_name]

    result = stage.handler(color_image, grayscale_image, DEFAULT_PARAMS)

    assert isinstance(result, StageResult)


def test_color_only_stages_are_flagged_correctly():
    color_only = {
        "RGB Channels",
        "HSV Channels",
        "Color Mask",
        "Color Threshold Result",
        "Color Histogram",
    }

    for name, stage in STAGES.items():
        assert stage.requires_color == (name in color_only)


def test_figure_stages_are_the_histograms():
    figure_stages = {
        name for name, stage in STAGES.items() if stage.is_figure
    }

    assert figure_stages == {"Grayscale Histogram", "Color Histogram"}


def test_only_the_grayscale_histogram_shows_statistics():
    with_stats = [
        name for name, stage in STAGES.items() if stage.shows_statistics
    ]

    assert with_stats == ["Grayscale Histogram"]


@pytest.mark.parametrize(
    "stage_name",
    [name for name, stage in STAGES.items() if not stage.needs_grayscale],
)
def test_stages_not_needing_grayscale_run_with_none(stage_name, color_image):
    """
    app.py skips the grayscale conversion for these stages and passes
    None instead, so their handlers must never touch it. Passing None
    here is what proves the flag is honest — if a handler did use the
    grayscale array, this would raise.
    """

    stage = STAGES[stage_name]

    result = stage.handler(color_image, None, DEFAULT_PARAMS)

    assert isinstance(result, StageResult)


def test_grayscale_free_stages_are_the_color_ones():
    grayscale_free = {
        name for name, stage in STAGES.items() if not stage.needs_grayscale
    }

    assert grayscale_free == {
        "Original Image",
        "RGB Channels",
        "HSV Channels",
        "Color Mask",
        "Color Threshold Result",
        "YOLO Object Detection",
        "MediaPipe Landmarks",
        "Median Filter",
        "Bilateral Filter",
        "Color Histogram",
    }
