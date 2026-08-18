"""Image loading, normalization, downsampling, and export helpers."""

import io
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from modules.caching import cache_data

# Images with a long side above this many pixels are downscaled
# before processing, so Haar Cascade / Harris / Canny etc. don't
# choke on 12-20MP+ photos.
MAX_DIMENSION = 1920

# Where "save a copy to disk" writes results when running locally
# (`streamlit run app.py`). Resolved relative to this file so it
# works regardless of the process's current working directory.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Bounded caches: decoding and PNG-encoding are expensive enough to be
# worth caching (~46 ms to encode a 1920x1080 PNG), but each entry
# holds a full-size image, so old entries are evicted rather than kept
# forever.
MAX_CACHE_ENTRIES = 8


@cache_data(show_spinner=False, max_entries=3)
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


@cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
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


def is_local_run() -> bool:
    """
    True when the app is running on the machine the user is sitting at.

    "Save a copy to results/" only makes sense there. On a hosted
    deployment the same button writes into the server's application
    directory, where the user can't reach the file, the filesystem is
    ephemeral anyway, and anyone passing by can fill the disk one click
    at a time — so the button is hidden instead.

    Detection is deliberately conservative and overridable: Streamlit
    Community Cloud checks out the repository under /mount/src, and
    CVLAB_HOSTED=1 forces the hosted behaviour anywhere else.
    """

    if os.environ.get("CVLAB_HOSTED", "").strip():
        return False

    return not str(Path(__file__).resolve()).startswith("/mount/src")


def save_result_locally(
    image_bytes: bytes,
    filename: str,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """
    Write already-encoded image bytes (PNG) to the local results/
    folder, timestamping the filename so repeated saves of the same
    stage don't overwrite each other.

    This is a convenience for local runs — not cached, since it's a
    side effect rather than a pure computation. Deployed/cloud
    environments (e.g. Streamlit Community Cloud) typically have an
    ephemeral filesystem, so this complements the in-browser download
    button rather than replacing it.

    Returns the path the file was written to.
    """

    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".png"

    output_path = results_dir / f"{timestamp}_{stem}{suffix}"
    output_path.write_bytes(image_bytes)

    return output_path
