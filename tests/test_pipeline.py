"""
Tests for the user-composed pipeline.

The interesting property here isn't that any single operation works —
each already has its own tests in the module it came from — it's that
*any order the user can click* produces a result rather than an
OpenCV assertion failure. The multiselect offers no invalid
combinations, so the runner has to make every combination valid.
"""

import itertools

import numpy as np
import pytest

from modules.controls import default_params
from modules.pipeline import OPERATIONS, run_pipeline

DEFAULT_PARAMS = default_params()


# ==================================================
# THE OPERATION TABLE
# ==================================================


def test_every_operation_is_well_formed():
    assert OPERATIONS

    for name, operation in OPERATIONS.items():
        assert name.strip()
        assert callable(operation.apply)
        assert operation.summary.strip(), name

        # Anything defined on a mask is by definition single-channel;
        # declaring otherwise would skip the grayscale conversion and
        # hand cv2 a 3-channel array it rejects.
        if operation.binary_input:
            assert operation.single_channel, name


# ==================================================
# RUNNING CHAINS
# ==================================================


def test_an_empty_chain_returns_the_input_untouched(color_image):
    result, trace = run_pipeline(color_image, [], DEFAULT_PARAMS)

    assert trace == []
    assert np.array_equal(result, color_image)


@pytest.mark.parametrize("step_name", list(OPERATIONS))
def test_every_operation_runs_on_a_color_image(step_name, color_image):
    result, trace = run_pipeline(color_image, [step_name], DEFAULT_PARAMS)

    assert isinstance(result, np.ndarray)
    assert result.size > 0
    assert trace[-1] == step_name


@pytest.mark.parametrize("step_name", list(OPERATIONS))
def test_every_operation_runs_on_a_grayscale_image(
    step_name, grayscale_only_image
):
    result, _trace = run_pipeline(
        grayscale_only_image, [step_name], DEFAULT_PARAMS
    )

    assert isinstance(result, np.ndarray)
    assert result.size > 0


@pytest.mark.parametrize(
    "first,second",
    # Every ordered pair of operations. This is the test that would
    # have caught "Canny after Contour Detection hands cv2 an RGB
    # array" — a combination nobody would think to write by hand.
    list(itertools.permutations(OPERATIONS, 2)),
)
def test_any_pair_of_operations_can_be_chained(first, second, color_image):
    result, trace = run_pipeline(
        color_image, [first, second], DEFAULT_PARAMS
    )

    assert isinstance(result, np.ndarray)
    assert result.size > 0
    assert trace.count(first) == 1
    assert trace.count(second) == 1


def test_a_long_chain_runs_end_to_end(color_image):
    steps = [
        "Gaussian Blur",
        "CLAHE",
        "Otsu Threshold",
        "Opening",
        "Contour Detection",
    ]

    result, trace = run_pipeline(color_image, steps, DEFAULT_PARAMS)

    assert result.shape[:2] == color_image.shape[:2]
    # Every requested step ran, in the order it was requested.
    assert [name for name in trace if name in steps] == steps


# ==================================================
# AUTOMATIC INSERTIONS
# ==================================================


def test_grayscale_is_inserted_before_a_single_channel_step(color_image):
    _result, trace = run_pipeline(
        color_image, ["Canny Edges"], DEFAULT_PARAMS
    )

    assert trace[0].startswith("Grayscale")
    assert "inserted automatically" in trace[0]


def test_grayscale_is_not_inserted_when_it_is_already_grayscale(
    grayscale_only_image
):
    _result, trace = run_pipeline(
        grayscale_only_image, ["Canny Edges"], DEFAULT_PARAMS
    )

    assert trace == ["Canny Edges"]


def test_threshold_is_inserted_before_a_binary_step(grayscale_image):
    """
    grayscale_image holds mid-tones (the red patch converts to ~76), so
    it is genuinely not a mask and morphology needs one made first.
    """

    _result, trace = run_pipeline(
        grayscale_image, ["Erosion"], DEFAULT_PARAMS
    )

    assert "Fixed Threshold *(inserted automatically)*" in trace


def test_threshold_is_not_inserted_for_an_already_binary_image(
    grayscale_only_image
):
    """The fixture is pure 0/255, so it is already a valid mask."""

    _result, trace = run_pipeline(
        grayscale_only_image, ["Erosion"], DEFAULT_PARAMS
    )

    assert trace == ["Erosion"]


def test_threshold_is_not_inserted_after_an_earlier_one(grayscale_only_image):
    """
    Thresholding an already-binary mask is a no-op at best and, with a
    threshold above 0, erases it entirely — so a chain that already
    produced a mask must not get another one bolted in front of the
    morphology step.
    """

    _result, trace = run_pipeline(
        grayscale_only_image,
        ["Otsu Threshold", "Dilation"],
        DEFAULT_PARAMS,
    )

    assert trace == ["Otsu Threshold", "Dilation"]


def test_chained_steps_actually_feed_each_other(grayscale_only_image):
    """
    Dilation after a threshold has to grow the mask the threshold
    produced. If each step secretly restarted from the original image,
    the two results below would be identical.
    """

    thresholded, _ = run_pipeline(
        grayscale_only_image, ["Otsu Threshold"], DEFAULT_PARAMS
    )

    dilated, _ = run_pipeline(
        grayscale_only_image,
        ["Otsu Threshold", "Dilation"],
        DEFAULT_PARAMS,
    )

    assert int((dilated == 255).sum()) > int((thresholded == 255).sum())


# ==================================================
# VALIDATION
# ==================================================


def test_an_unknown_step_raises(color_image):
    with pytest.raises(ValueError, match="Unknown pipeline step"):
        run_pipeline(color_image, ["Deblur With Magic"], DEFAULT_PARAMS)


def test_the_sidebar_only_offers_real_operations():
    """
    The multiselect's options come from OPERATIONS, and its default has
    to survive being run — a default naming a step that no longer
    exists would break the stage on first load, before the user touches
    anything.
    """

    from modules.controls import SECTIONS

    steps_control = next(
        control
        for section in SECTIONS
        for control in section.controls
        if control.key == "pipeline_steps"
    )

    assert set(steps_control.options) == set(OPERATIONS)
    assert set(steps_control.default) <= set(OPERATIONS)
