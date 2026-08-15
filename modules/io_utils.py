"""Image loading, normalization, downsampling, and export helpers."""

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Images with a long side above this many pixels are downscaled
# before processing, so Haar Cascade / Harris / Canny etc. don't
# choke on 12-20MP+ photos.
MAX_DIMENSION = 1920


@st.cache_data(show_spinner=False)
def load_image(file_bytes: bytes):
    """
    Decode uploaded bytes into a normalized RGB/RGBA/L numpy array,
    downscaling it first if its long side exceeds MAX_DIMENSION.

    Returns (image_np, resize_info), where resize_info is None if
    no resizing was needed, or a dict with the original and new
    dimensions otherwise.
    """

    image = Image.open(io.BytesIO(file_bytes))

    # Normalize unusual modes (palette, CMYK, etc.) to RGB so the
    # rest of the pipeline only ever has to deal with L / RGB / RGBA.
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")

    image_np = np.array(image)

    original_height, original_width = image_np.shape[:2]
    long_side = max(original_height, original_width)

    resize_info = None

    if long_side > MAX_DIMENSION:
        scale = MAX_DIMENSION / long_side
        new_width = max(1, round(original_width * scale))
        new_height = max(1, round(original_height * scale))

        image_np = cv2.resize(
            image_np,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        resize_info = {
            "original_width": original_width,
            "original_height": original_height,
            "new_width": new_width,
            "new_height": new_height,
        }

    return image_np, resize_info


@st.cache_data(show_spinner=False)
def get_image_download_bytes(image_np: np.ndarray) -> bytes:
    """
    Encode an RGB/RGBA/grayscale numpy image array as PNG bytes,
    suitable for st.download_button.
    """

    if image_np.ndim == 2:
        encode_input = image_np
    else:
        rgb = image_np[:, :, :3]
        encode_input = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".png", encode_input)

    if not success:
        raise RuntimeError("Failed to encode image to PNG.")

    return buffer.tobytes()
