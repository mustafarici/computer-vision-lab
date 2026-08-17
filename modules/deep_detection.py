"""
Deep-learning based detection: YOLO (ultralytics) and MediaPipe.

Both libraries are OPTIONAL. `ultralytics` pulls in PyTorch and
`mediapipe` ships its own runtime, so neither belongs in the base
requirements of an app whose point is teaching classical OpenCV. They
live in requirements-ml.txt instead, and everything here is written so
that the app runs normally when they aren't installed: availability is
checked with importlib (which doesn't pay the multi-second cost of
actually importing torch), and the stages report what to install
rather than raising.

Models are loaded with @st.cache_resource — they're resources, not
data, and reloading a network on every rerun would make the app
unusable.

Adding your own model: write a load_*() wrapped in @st.cache_resource
and an apply_*() that takes the RGB image plus its parameters and
returns (canvas, count, summary). Register it in modules/stages.py the
same way the two below are.
"""

import importlib.util
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Downloaded model files live here (git-ignored). MediaPipe, unlike
# ultralytics, has no built-in downloader — the caller supplies a path.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class MissingDependencyError(RuntimeError):
    """Raised when a stage needs an optional package that isn't installed."""


# ==================================================
# YOLO (ultralytics)
# ==================================================

# Weights are fetched automatically by ultralytics on first use.
# Nano is the default because it's ~6 MB and runs in well under a
# second on CPU; the larger ones are more accurate and slower.
YOLO_MODELS = {
    "YOLOv8 Nano (fastest)": "yolov8n.pt",
    "YOLOv8 Small": "yolov8s.pt",
    "YOLOv8 Medium (most accurate)": "yolov8m.pt",
}


def is_yolo_available() -> bool:
    """True if ultralytics can be imported, without importing it."""

    return importlib.util.find_spec("ultralytics") is not None


@st.cache_resource(show_spinner="Loading YOLO model…")
def load_yolo_model(weights: str):
    """
    Load (downloading on first use) a YOLO model.

    Cached as a resource so the weights are read once per session
    instead of on every rerun.
    """

    if not is_yolo_available():
        raise MissingDependencyError(
            "The 'ultralytics' package is required for YOLO detection. "
            "Install it with: pip install -r requirements-ml.txt"
        )

    from ultralytics import YOLO

    return YOLO(weights)


def _class_color(class_id: int) -> tuple:
    """
    A stable, well-separated color per class id.

    Spreading hues by the golden ratio keeps neighbouring class ids
    visually distinct instead of nearly the same shade.
    """

    hue = int((class_id * 137.508) % 180)  # OpenCV hue range is 0-179
    hsv = np.uint8([[[hue, 220, 255]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]

    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _draw_label(canvas: np.ndarray, text: str, x: int, y: int, color: tuple):
    """Draw a filled label chip with the text above a box corner."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )

    # Keep the chip on screen when the box starts at the very top.
    top = max(y - text_height - baseline - 4, 0)

    cv2.rectangle(
        canvas,
        (x, top),
        (x + text_width + 6, top + text_height + baseline + 4),
        color,
        thickness=-1,
    )

    cv2.putText(
        canvas,
        text,
        (x + 3, top + text_height + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _as_rgb(image_np: np.ndarray) -> np.ndarray:
    """Normalize any supported input to a 3-channel RGB array."""

    if image_np.ndim == 2:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    return image_np[:, :, :3]


def apply_yolo_detection(
    image_np: np.ndarray,
    model_name: str,
    confidence: float,
    iou: float,
):
    """
    Run YOLO object detection and draw labelled boxes.

    Returns (image_with_boxes, detection_count, summary), where summary
    is a human-readable count per class, e.g. "2 person, 1 dog".
    """

    if model_name not in YOLO_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Must be one of: {list(YOLO_MODELS)}."
        )

    model = load_yolo_model(YOLO_MODELS[model_name])

    rgb = _as_rgb(image_np)

    # ultralytics follows the OpenCV convention and treats a numpy
    # array as BGR, but everything in this app is RGB — so convert on
    # the way in, or every prediction sees swapped color channels.
    predictions = model.predict(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        conf=confidence,
        iou=iou,
        verbose=False,
    )

    canvas = rgb.copy()
    counts: dict = {}

    boxes = predictions[0].boxes
    names = predictions[0].names

    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        class_id = int(box.cls[0])
        score = float(box.conf[0])
        label = names.get(class_id, str(class_id))

        color = _class_color(class_id)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _draw_label(canvas, f"{label} {score:.2f}", x1, y1, color)

        counts[label] = counts.get(label, 0) + 1

    summary = ", ".join(
        f"{count} {label}"
        for label, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )

    return canvas, len(boxes), summary


# ==================================================
# MediaPipe
# ==================================================

# MediaPipe's older `mp.solutions` API (which bundled its models) was
# removed in 0.10.30+, so these use the Tasks API, which needs a
# .task bundle downloaded once and kept on disk.
MEDIAPIPE_TASKS = {
    "Face Mesh": {
        "task": "FaceLandmarker",
        "filename": "face_landmarker.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
        "result_field": "face_landmarks",
        "connections": "FaceLandmarksConnections",
        "point_radius": 1,
    },
    "Hand Landmarks": {
        "task": "HandLandmarker",
        "filename": "hand_landmarker.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        ),
        "result_field": "hand_landmarks",
        "connections": "HandLandmarksConnections",
        "point_radius": 3,
    },
    "Body Pose": {
        "task": "PoseLandmarker",
        "filename": "pose_landmarker_lite.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        ),
        "result_field": "pose_landmarks",
        "connections": "PoseLandmarksConnections",
        "point_radius": 3,
    },
}


def is_mediapipe_available() -> bool:
    return importlib.util.find_spec("mediapipe") is not None


def download_model(url: str, destination: Path) -> Path:
    """
    Fetch a MediaPipe .task bundle once and keep it on disk.

    Written to a temporary name first so an interrupted download can't
    leave a truncated file that looks valid on the next run.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")

    try:
        urllib.request.urlretrieve(url, partial)

    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)

        raise RuntimeError(
            f"Couldn't download the MediaPipe model from {url}. "
            "Check your internet connection, or download the file "
            f"manually and place it at {destination}."
        ) from error

    partial.replace(destination)

    return destination


@st.cache_resource(show_spinner="Loading MediaPipe model…")
def load_mediapipe_landmarker(task_name: str):
    """Load a MediaPipe landmarker, downloading its model if needed."""

    if not is_mediapipe_available():
        raise MissingDependencyError(
            "The 'mediapipe' package is required for this stage. "
            "Install it with: pip install -r requirements-ml.txt"
        )

    spec = MEDIAPIPE_TASKS[task_name]

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = download_model(
        spec["url"], MODELS_DIR / spec["filename"]
    )

    task_class = getattr(vision, spec["task"])
    options_class = getattr(vision, f"{spec['task']}Options")

    options = options_class(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
    )

    return task_class.create_from_options(options)


def _connection_edges(connections_name: str):
    """
    Collect the (start, end) index pairs used to wire landmarks
    together for drawing.

    The Tasks API exposes these as plain lists of Connection objects
    grouped into class attributes (e.g. HAND_PALM_CONNECTIONS), so any
    list-valued attribute on the class is a set of edges.
    """

    from mediapipe.tasks.python import vision

    holder = getattr(vision, connections_name, None)

    if holder is None:
        return []

    edges = []

    for attribute in dir(holder):
        if attribute.startswith("_"):
            continue

        value = getattr(holder, attribute)

        if isinstance(value, list):
            for edge in value:
                start = getattr(edge, "start", None)
                end = getattr(edge, "end", None)

                if start is not None and end is not None:
                    edges.append((start, end))

    return edges


def apply_mediapipe_landmarks(image_np: np.ndarray, task_name: str):
    """
    Detect landmarks with MediaPipe and draw them on the image.

    Returns (image_with_landmarks, detection_count, summary), where the
    count is the number of detected instances (faces, hands or bodies).
    """

    if task_name not in MEDIAPIPE_TASKS:
        raise ValueError(
            f"Unknown MediaPipe task '{task_name}'. "
            f"Must be one of: {list(MEDIAPIPE_TASKS)}."
        )

    spec = MEDIAPIPE_TASKS[task_name]
    landmarker = load_mediapipe_landmarker(task_name)

    import mediapipe as mp

    rgb = _as_rgb(image_np)

    # MediaPipe expects RGB, which is what we already have.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    instances = getattr(result, spec["result_field"], []) or []

    canvas = rgb.copy()
    height, width = canvas.shape[:2]
    edges = _connection_edges(spec["connections"])

    for index, landmarks in enumerate(instances):
        color = _class_color(index * 7)

        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in landmarks
        ]

        for start, end in edges:
            if start < len(points) and end < len(points):
                cv2.line(canvas, points[start], points[end], color, 1,
                         lineType=cv2.LINE_AA)

        for point in points:
            cv2.circle(canvas, point, spec["point_radius"], color, -1,
                       lineType=cv2.LINE_AA)

    noun = {
        "Face Mesh": "face",
        "Hand Landmarks": "hand",
        "Body Pose": "body",
    }[task_name]

    summary = f"{len(instances)} {noun}(s)"

    return canvas, len(instances), summary
