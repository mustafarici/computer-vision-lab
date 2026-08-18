"""
The UI layer.

`modules/` computes things and knows nothing about Streamlit widgets;
`views/` draws things and contains no image processing. app.py picks
which view to render and hands it what it needs.

The split exists because app.py had already been through this once:
before the stage registry it was a 1400-line file where metadata, a
giant if/elif chain and the sidebar all had to be kept in step by hand.
Adding camera and video input started it growing back — three input
modes in one script, with `st.stop()` used as control flow. One file
per view keeps each one short enough to read in a sitting.
"""
