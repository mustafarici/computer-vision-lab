"""
Tests for the declarative sidebar schema.

These are mostly consistency checks that a hand-written sidebar
couldn't have: that no two controls fight over the same key, that
every default is actually reachable on its own widget, and that the
sections point at stage categories that exist.
"""

import pytest
from streamlit.testing.v1 import AppTest

from modules.controls import (
    SECTIONS,
    active_categories_for,
    default_params,
)
from modules.stages import STAGES

ALL_CONTROLS = [control for section in SECTIONS for control in section.controls]

# A minimal app that does nothing but draw the sidebar, so the widgets
# are exercised by a real Streamlit run rather than only inspected as
# data. This is what catches an invalid step, an out-of-range default
# or a duplicate widget key — none of which show up until Streamlit
# actually tries to render the control.
SIDEBAR_HARNESS = """
import streamlit as st
from modules.controls import render_sidebar

with st.sidebar:
    params = render_sidebar(st.session_state.get("category", ""))

st.session_state["rendered_params"] = params
"""


def test_control_keys_are_unique():
    keys = [control.key for control in ALL_CONTROLS]

    assert len(keys) == len(set(keys))


def test_default_params_covers_every_control():
    params = default_params()

    assert set(params) == {control.key for control in ALL_CONTROLS}


def test_every_control_has_a_known_kind():
    for control in ALL_CONTROLS:
        assert control.kind in {"slider", "range", "select", "multiselect"}


def test_slider_defaults_are_within_bounds():
    for control in ALL_CONTROLS:
        if control.kind != "slider":
            continue

        assert control.min_value is not None, control.key
        assert control.max_value is not None, control.key
        assert control.min_value <= control.default <= control.max_value, (
            control.key
        )


def test_range_defaults_are_ordered_and_within_bounds():
    for control in ALL_CONTROLS:
        if control.kind != "range":
            continue

        low, high = control.default

        assert low <= high, control.key
        assert control.min_value <= low, control.key
        assert high <= control.max_value, control.key


def test_select_defaults_are_valid_options():
    for control in ALL_CONTROLS:
        if control.kind != "select":
            continue

        assert control.options, control.key
        assert control.default in control.options, control.key


def test_multiselect_defaults_are_valid_options():
    for control in ALL_CONTROLS:
        if control.kind != "multiselect":
            continue

        assert control.options, control.key

        for choice in control.default:
            assert choice in control.options, (control.key, choice)


def test_every_control_has_help_text():
    for control in ALL_CONTROLS:
        assert control.help.strip(), control.key


def test_sections_reference_real_stage_categories():
    known_categories = {stage.category for stage in STAGES.values()}

    for section in SECTIONS:
        assert section.categories, section.title
        assert section.categories <= known_categories, section.title


def test_every_stage_category_with_parameters_has_a_section():
    """
    Every category whose stages actually read a parameter should have
    somewhere in the sidebar to set it. Histogram Analysis is the one
    category with no tunable parameters, so it's expected to have none.
    """

    covered = set().union(*(section.categories for section in SECTIONS))
    all_categories = {stage.category for stage in STAGES.values()}

    assert all_categories - covered == {"Histogram Analysis"}


@pytest.mark.parametrize(
    "category", sorted({stage.category for stage in STAGES.values()})
)
def test_sidebar_renders_for_every_stage_category(category):
    """Rendering must succeed whichever stage the user is looking at."""

    app = AppTest.from_string(SIDEBAR_HARNESS)
    app.session_state["category"] = category
    app.run()

    assert not app.exception
    assert set(app.session_state["rendered_params"]) == {
        control.key for control in ALL_CONTROLS
    }


def test_rendered_defaults_match_the_declared_defaults():
    """
    What the widgets hand back on a fresh run has to be exactly what
    default_params() promises, or the tests that build parameters from
    the schema would be testing something the app never produces.
    """

    app = AppTest.from_string(SIDEBAR_HARNESS)
    app.run()

    assert not app.exception
    assert app.session_state["rendered_params"] == default_params()


def test_relevant_section_is_expanded_and_others_are_not():
    app = AppTest.from_string(SIDEBAR_HARNESS)
    app.session_state["category"] = "Feature Detection"
    app.run()

    expanded = {
        expander.label: expander.proto.expanded
        for expander in app.sidebar.expander
    }

    assert expanded["🔍 Feature Detection"] is True
    assert expanded["🧩 Morphological Operations"] is False


def test_canny_warning_fires_only_when_thresholds_are_inverted():
    section = next(s for s in SECTIONS if s.warning is not None)
    params = default_params()

    assert section.warning(params) is None

    params["lower_threshold"] = 200
    params["upper_threshold"] = 100

    assert section.warning(params)


# ==================================================
# WHICH SECTIONS OPEN
# ==================================================


def test_a_normal_stage_only_opens_its_own_section():
    assert active_categories_for("Morphology", default_params()) == frozenset(
        {"Morphology"}
    )


def test_a_pipeline_opens_the_sections_its_steps_are_tuned_from():
    """
    A chain has one control of its own and borrows the rest. Being told
    "the blur lives somewhere else" and having to go find it is the
    failure this prevents.
    """

    params = default_params()
    params["pipeline_steps"] = ["Gaussian Blur", "CLAHE", "Erosion"]

    categories = active_categories_for("Pipeline", params)

    assert "Pipeline" in categories
    assert "Basic Processing" in categories   # Gaussian Blur
    assert "Enhancement" in categories        # CLAHE
    assert "Morphology" in categories         # Erosion
    # Nothing it doesn't use.
    assert "Hough Transform" not in categories


def test_a_pipeline_of_parameterless_steps_opens_nothing_extra():
    params = default_params()
    params["pipeline_steps"] = ["Grayscale", "Otsu Threshold", "Invert"]

    assert active_categories_for("Pipeline", params) == frozenset({"Pipeline"})


def test_an_unrendered_pipeline_falls_back_to_the_declared_default():
    """
    This is called before the sidebar renders, so on first load session
    state has no value for the multiselect yet — the sections that open
    still have to match the chain the user is about to be shown.
    """

    from_empty = active_categories_for("Pipeline", {})
    from_default = active_categories_for("Pipeline", default_params())

    assert from_empty == from_default
    assert len(from_empty) > 1


def test_every_pipeline_step_points_at_a_real_section():
    from modules.pipeline import OPERATIONS

    known = set().union(*(section.categories for section in SECTIONS))

    for name, operation in OPERATIONS.items():
        if operation.settings_category:
            assert operation.settings_category in known, name
