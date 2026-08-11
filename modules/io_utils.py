"""Image loading and normalization."""

import io

import numpy as np
import streamlit as st
from PIL import Image


@st.cache_data(show_spinner=False)
def load_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into a normalized RGB/RGBA/L numpy array."""

    image = Image.open(io.BytesIO(file_bytes))

    # Normalize unusual modes (palette, CMYK, etc.) to RGB so the
    # rest of the pipeline only ever has to deal with L / RGB / RGBA.
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")

    return np.array(image)
