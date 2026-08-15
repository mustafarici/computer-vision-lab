"""Grayscale histogram computation, plotting, and export helpers."""

import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def compute_histogram(grayscale: np.ndarray) -> np.ndarray:
    return cv2.calcHist([grayscale], [0], None, [256], [0, 256])


def build_histogram_figure(histogram: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(histogram)
    ax.set_title("Grayscale Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim([0, 256])

    fig.tight_layout()

    return fig


def get_figure_download_bytes(fig) -> bytes:
    """Encode a matplotlib figure as PNG bytes for st.download_button."""

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    return buffer.getvalue()
