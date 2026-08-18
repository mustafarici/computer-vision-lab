# assets/

Static files the project references but the app doesn't compute:
currently the screenshots embedded in the top-level `README.md`.

They are real captures of the running app rather than mockups —
produced by driving a local `streamlit run` with a headless browser
against `images/NASA_Astronaut_Group_15.jpg`, so what the README shows
is what the code actually renders. Regenerating them after a UI change
means re-capturing, not editing.

This folder is also the right home for anything else the UI itself
needs later — a favicon or logo passed to
`st.set_page_config(page_icon=...)`, for example.
