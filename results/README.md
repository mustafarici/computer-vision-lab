# results/

Local output folder. When you click **"💾 Save a copy to results/"**
next to any stage's download button, the app writes a timestamped PNG
here (e.g. `20260817_184530_canny_edge_detection.png`).

This only works for local runs (`streamlit run app.py`) — it's a
convenience alongside the in-browser download button, not a
replacement for it. Deployed environments (e.g. Streamlit Community
Cloud) typically have an ephemeral filesystem, so files saved there
won't persist.

Saved images are git-ignored (see `.gitignore`) so your local results
don't end up committed to the repo.
