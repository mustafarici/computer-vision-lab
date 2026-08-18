"""
Pre-download the model files the optional deep-learning stages need.

Useful in two places:

- CI, where fetching the weights as its own step means a model-hosting
  outage reads as an infrastructure failure rather than as a mysterious
  assertion error inside a test.
- Locally, when you would rather wait for a few hundred megabytes once,
  up front, than have the first click on the YOLO stage sit there
  downloading.

    python scripts/fetch_models.py

Everything lands in models/ (git-ignored) except the YOLO weights,
which ultralytics manages in its own working directory.
"""

import sys
from pathlib import Path

# Allow running this as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.deep_detection import (  # noqa: E402
    MEDIAPIPE_TASKS,
    MODELS_DIR,
    YOLO_MODELS,
    download_model,
    is_mediapipe_available,
    is_yolo_available,
)

# Only the default weights. The larger YOLO models are a user choice at
# runtime, and pulling all three would triple the download for no extra
# coverage.
DEFAULT_YOLO_WEIGHTS = next(iter(YOLO_MODELS.values()))


def fetch_yolo() -> bool:
    """Fetch the default YOLO weights. Returns False if unavailable."""

    if not is_yolo_available():
        print("- ultralytics not installed, skipping YOLO weights")
        return False

    from ultralytics import YOLO

    print(f"- YOLO weights: {DEFAULT_YOLO_WEIGHTS}")

    # Constructing the model is what triggers the download; ultralytics
    # reuses the file on every later run.
    YOLO(DEFAULT_YOLO_WEIGHTS)

    return True


def fetch_mediapipe() -> bool:
    """Fetch every MediaPipe task bundle. Returns False if unavailable."""

    if not is_mediapipe_available():
        print("- mediapipe not installed, skipping task bundles")
        return False

    for name, spec in MEDIAPIPE_TASKS.items():
        destination = MODELS_DIR / spec["filename"]

        print(f"- MediaPipe {name}: {spec['filename']}")

        download_model(spec["url"], destination)

    return True


def main() -> int:
    print("Fetching optional models…")

    fetched_yolo = fetch_yolo()
    fetched_mediapipe = fetch_mediapipe()

    if not (fetched_yolo or fetched_mediapipe):
        print(
            "\nNeither optional package is installed — nothing to do. "
            "Install them with: pip install -r requirements-ml.txt"
        )
        return 0

    print("\nDone.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
