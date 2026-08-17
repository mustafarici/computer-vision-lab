"""Grayscale histogram computation, plotting, and export helpers."""

import io

import cv2
import numpy as np
from matplotlib.figure import Figure


def compute_histogram(grayscale: np.ndarray) -> np.ndarray:
    """Not cached: ~0.9 ms per run, cheaper than a cache lookup."""

    return cv2.calcHist([grayscale], [0], None, [256], [0, 256])


def build_histogram_figure(histogram: np.ndarray) -> Figure:
    """
    Build the histogram plot.

    Uses matplotlib's object-oriented Figure directly instead of
    plt.subplots(). pyplot keeps every figure it creates alive in a
    global registry, so calling this on each Streamlit rerun leaked one
    figure per rerun (25 reruns => 25 figures still in memory, plus
    matplotlib's "More than 20 figures have been opened" warning).
    A bare Figure isn't registered anywhere, so it's freed normally
    once Streamlit is done rendering it. It's also the thread-safe
    option, which matters because Streamlit runs scripts off the main
    thread.
    """

    fig = Figure(figsize=(5, 3))
    ax = fig.subplots()

    ax.plot(histogram)
    ax.set_title("Grayscale Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim([0, 256])

    fig.tight_layout()

    return fig


def get_figure_download_bytes(fig: Figure) -> bytes:
    """Encode a matplotlib figure as PNG bytes for st.download_button."""

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    return buffer.getvalue()
