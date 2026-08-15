"""
Multi-class object detection using OpenCV's bundled Haar Cascade
classifiers (beyond faces: eyes, smiles, full bodies, cat faces,
license plates).
"""

import cv2
import numpy as np
import streamlit as st

# Maps a friendly display name to the bundled cascade filename.
OBJECT_CASCADES = {
    "Eyes": "haarcascade_eye.xml",
    "Smile": "haarcascade_smile.xml",
    "Full Body (Pedestrian)": "haarcascade_fullbody.xml",
    "Cat Face": "haarcascade_frontalcatface.xml",
    "License Plate (Russian)": "haarcascade_russian_plate_number.xml",
}


@st.cache_resource(show_spinner=False)
def load_object_cascade(cascade_filename: str) -> cv2.CascadeClassifier:
    """
    Load a bundled Haar Cascade classifier by filename and reuse it
    across reruns. Cached per filename, since different object
    types need different classifiers loaded.
    """

    cascade_path = cv2.data.haarcascades + cascade_filename
    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        raise RuntimeError(
            f"Failed to load Haar Cascade classifier from '{cascade_path}'. "
            "The OpenCV installation may be missing its bundled data files."
        )

    return cascade


@st.cache_data(show_spinner=False)
def apply_object_detection(
    image_np: np.ndarray,
    grayscale: np.ndarray,
    object_type: str,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
    return_boxes: bool = False
):
    """
    Detect instances of the selected object class with the matching
    Haar Cascade classifier and draw bounding boxes on a color
    canvas (the original image if it's already color, otherwise the
    grayscale image converted to RGB).

    Returns (image_with_boxes, detection_count), or
    (image_with_boxes, detection_count, boxes) if return_boxes=True,
    where boxes is a list of (x, y, w, h) tuples in pixel coordinates.
    """

    if object_type not in OBJECT_CASCADES:
        raise ValueError(
            f"Unknown object_type '{object_type}'. "
            f"Must be one of: {list(OBJECT_CASCADES.keys())}."
        )

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

    cascade = load_object_cascade(OBJECT_CASCADES[object_type])

    detections = cascade.detectMultiScale(
        grayscale,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size)
    )

    if image_np.ndim == 3 and image_np.shape[2] >= 3:
        canvas = image_np[:, :, :3].copy()
    else:
        canvas = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)

    for (x, y, w, h) in detections:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 165, 0), 2)

    if return_boxes:
        boxes = [tuple(int(v) for v in box) for box in detections]
        return canvas, len(detections), boxes

    return canvas, len(detections)
