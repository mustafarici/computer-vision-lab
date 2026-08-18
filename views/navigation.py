"""Stage navigation: previous / next, the counter, and the jump list."""

import streamlit as st

from modules.stages import STAGES

DEFAULT_STAGE = "Original Image"


def current_stage() -> str:
    """
    The stage on screen, defaulting on first load and self-healing if
    session state somehow holds a stage that no longer exists.
    """

    if (
        "current_stage" not in st.session_state
        or st.session_state.current_stage not in STAGES
    ):
        st.session_state.current_stage = DEFAULT_STAGE

    return st.session_state.current_stage


def _step_to(offset: int, stage: str):
    titles = list(STAGES)
    index = titles.index(stage)

    st.session_state.current_stage = titles[(index + offset) % len(titles)]

    st.rerun()


def render(stage: str):
    """Draw the navigation controls under the result."""

    titles = list(STAGES)
    index = titles.index(stage)

    st.write("")

    previous_col, counter_col, next_col = st.columns([1, 2, 1])

    with previous_col:
        if st.button("← Previous", key="previous_button", width="stretch"):
            _step_to(-1, stage)

    with counter_col:
        # unsafe_allow_html is the only way to centre text in Streamlit,
        # and the only thing interpolated here is two integers computed
        # from the registry — no user input reaches this string.
        st.markdown(
            f"""
            <div style="
                text-align: center;
                padding-top: 8px;
                font-size: 14px;
            ">
                <b>
                    {index + 1} / {len(titles)}
                </b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with next_col:
        if st.button("Next →", key="next_button", width="stretch"):
            _step_to(1, stage)

    st.write("")

    st.selectbox(
        "📍 Jump to a specific stage",
        options=titles,
        key="current_stage",
    )
