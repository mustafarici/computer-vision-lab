"""Harris corner detection and ORB keypoint detection."""

import cv2
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def compute_harris_response(
    grayscale: np.ndarray,
    block_size: int,
    ksize: int,
    k: float
) -> np.ndarray:
    """
    Compute the raw Harris corner response.

    Kept separate from thresholding/drawing so that moving only the
    threshold slider doesn't force cornerHarris (the expensive part)
    to recompute — it only depends on block_size, ksize and k.
    """

    gray_float = np.float32(grayscale)
    return cv2.cornerHarris(gray_float, block_size, ksize, k)


@st.cache_data(show_spinner=False)
def apply_harris_corners(
    grayscale: np.ndarray,
    harris_response: np.ndarray,
    threshold_ratio: float
) -> np.ndarray:
    """
    Threshold the Harris response and mark detected corners on a
    color version of the grayscale image.
    """

    # 1. Eşik değerini belirle ve maskeyi oluştur
    corner_mask = harris_response > (threshold_ratio * harris_response.max())

    # 2. Görselleştirme için maskeyi hafifçe genişlet (dilasyon)
    # Eşiklemeden SONRA dilasyon yapmak köşe tespit mantığını bozmaz
    corner_mask_uint8 = np.uint8(corner_mask) * 255
    corner_mask_dilated = (
        cv2.dilate(corner_mask_uint8, None, iterations=1) > 0
    )

    canvas = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    canvas[corner_mask_dilated] = [255, 0, 0]  # Kırmızı noktalar

    return canvas


@st.cache_data(show_spinner=False)
def apply_orb_keypoints(grayscale: np.ndarray, n_features: int) -> tuple[np.ndarray, int]:
    """
    Detect ORB keypoints and draw them on a color version of the grayscale
    image.

    Returns (image_with_keypoints, keypoint_count).
    """

    orb = cv2.ORB_create(nfeatures=n_features)

    # Descriptor (tanımlayıcı) gerekmediği için sadece detect() kullanıyoruz
    keypoints = orb.detect(grayscale, None)

    canvas = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)

    canvas = cv2.drawKeypoints(
        grayscale,
        keypoints,
        canvas,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return canvas, len(keypoints)
