import io

import numpy as np
from PIL import Image

from modules.io_utils import (
    MAX_DIMENSION,
    get_image_download_bytes,
    load_image,
    save_result_locally,
)


def _png_bytes(width, height, mode="RGB"):
    color = (100, 150, 200) if mode == "RGB" else 100
    image = Image.new(mode, (width, height), color=color)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_no_resize_when_small():
    image_np, resize_info = load_image(_png_bytes(50, 40))

    assert image_np.shape[:2] == (40, 50)
    assert resize_info is None


def test_load_image_downscales_large_images():
    long_side = MAX_DIMENSION + 500

    image_np, resize_info = load_image(_png_bytes(long_side, 100))

    assert resize_info is not None
    assert resize_info["original_width"] == long_side
    assert max(image_np.shape[:2]) == MAX_DIMENSION


def test_get_image_download_bytes_roundtrip_color():
    image_np = np.zeros((10, 10, 3), dtype=np.uint8)
    image_np[:, :, 0] = 255  # pure red, RGB order

    png_bytes = get_image_download_bytes(image_np)
    decoded = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))

    assert np.array_equal(decoded, image_np)


def test_get_image_download_bytes_roundtrip_grayscale():
    grayscale = np.full((10, 10), 128, dtype=np.uint8)

    png_bytes = get_image_download_bytes(grayscale)
    decoded = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))

    assert np.array_equal(decoded, grayscale)


def test_save_result_locally_creates_file(tmp_path):
    results_dir = tmp_path / "results"
    png_bytes = get_image_download_bytes(np.zeros((5, 5, 3), dtype=np.uint8))

    saved_path = save_result_locally(
        png_bytes, "binary_image.png", results_dir=results_dir
    )

    assert saved_path.exists()
    assert saved_path.read_bytes() == png_bytes
    assert saved_path.parent == results_dir
    assert saved_path.name.endswith("_binary_image.png")


def test_save_result_locally_creates_missing_directory(tmp_path):
    results_dir = tmp_path / "nested" / "results"
    png_bytes = get_image_download_bytes(np.zeros((5, 5, 3), dtype=np.uint8))

    saved_path = save_result_locally(
        png_bytes, "canny_edge_detection.png", results_dir=results_dir
    )

    assert results_dir.exists()
    assert saved_path.exists()
