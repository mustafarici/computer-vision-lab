"""Face detection using OpenCV's built-in Haar Cascade classifier."""

import cv2
import numpy as np
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_face_cascade() -> cv2.CascadeClassifier:
    """
    Load the Haar Cascade face detector once and reuse it across
    reruns. This is a model/resource (not data), so it's cached
    with @st.cache_resource rather than @st.cache_data.
    """

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    # ① cv2.CascadeClassifier() doesn't raise on a bad/missing path —
    # it silently returns an unusable classifier. .empty() is the
    # documented way to check the file actually loaded.
    if cascade.empty():
        raise RuntimeError(
            f"Failed to load Haar Cascade classifier from '{cascade_path}'. "
            "The OpenCV installation may be missing its bundled data files."
        )

    return cascade


# By far the most expensive operation in the app (~700 ms on a
# 1920x1080 image), so caching is essential — bounded because the
# scale/neighbors/size sliders have thousands of combinations.
MAX_CACHE_ENTRIES = 8


@st.cache_data(show_spinner=False, max_entries=MAX_CACHE_ENTRIES)
def apply_face_detection(
    image_np: np.ndarray,
    grayscale: np.ndarray,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
    return_boxes: bool = False
):
    """
    Detect faces with the Haar Cascade classifier and draw bounding
    boxes on a color canvas (the original image if it's already
    color, otherwise the grayscale image converted to RGB).

    Returns (image_with_boxes, face_count), or
    (image_with_boxes, face_count, boxes) if return_boxes=True,
    where boxes is a list of (x, y, w, h) tuples in pixel coordinates.
    """

    # ② Parameter validation — cv2.detectMultiScale fails with a
    # cryptic C++ assertion error for invalid values, so we check
    # up front and raise a clear, actionable message instead.
    if scale_factor <= 1.0:
        raise ValueError(
            f"scale_factor must be greater than 1.0, got {scale_factor}."
        )

    if min_neighbors < 0:
        raise ValueError(
            f"min_neighbors must be non-negative, got {min_neighbors}."
        )

    if min_size <= 0:
        raise ValueError(
            f"min_size must be a positive number of pixels, got {min_size}."
        )

    if grayscale.ndim != 2:
        raise ValueError(
            "grayscale must be a single-channel 2D array, "
            f"got shape {grayscale.shape}."
        )

    cascade = load_face_cascade()

    faces = cascade.detectMultiScale(
        grayscale,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size)
    )

    if image_np.ndim == 3 and image_np.shape[2] >= 3:
        canvas = image_np[:, :, :3].copy()
    else:
        canvas = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)

    for (x, y, w, h) in faces:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ③ Optional bounding box output — off by default so existing
    # callers unpacking (canvas, count) keep working unchanged.
    if return_boxes:
        boxes = [tuple(int(v) for v in box) for box in faces]
        return canvas, len(faces), boxes

    return canvas, len(faces)