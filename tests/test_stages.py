"""
Integration tests for the modules/stages.py registry — the piece
app.py's refactor introduced to replace the old if/elif chain.

These exercise every stage handler end-to-end with a representative
set of parameters, to make sure the refactor didn't silently change
behavior or drop a stage.
"""

import pytest

from modules.stages import STAGES, StageResult

EXPECTED_STAGE_COUNT = 21

DEFAULT_PARAMS = dict(
    threshold_value=127,
    kernel_size=5,
    lower_threshold=50,
    upper_threshold=150,
    sobel_kernel_size=3,
    laplacian_kernel_size=3,
    morph_kernel_size=5,
    morph_iterations=1,
    contour_retrieval_mode="External",
    contour_approx_method="Simple",
    min_contour_area=50,
    hue_range=(0, 179),
    saturation_range=(0, 255),
    value_range=(0, 255),
    harris_block_size=2,
    harris_ksize=3,
    harris_sensitivity=0.04,
    harris_threshold=0.01,
    orb_features=300,
    face_scale_factor=1.10,
    face_min_neighbors=5,
    face_min_size=30,
    object_type="Eyes",
    object_scale_factor=1.10,
    object_min_neighbors=5,
    object_min_size=20,
)


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
    }

    for name, stage in STAGES.items():
        assert stage.requires_color == (name in color_only)


def test_histogram_is_the_only_figure_stage():
    figure_stages = [
        name for name, stage in STAGES.items() if stage.is_figure
    ]

    assert figure_stages == ["Grayscale Histogram"]


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
    }
