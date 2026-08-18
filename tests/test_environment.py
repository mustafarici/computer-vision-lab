"""
Guards on the installed environment rather than on our own code.

The same bug has now shipped twice: something resolves to OpenCV 5,
which removed `cv2.CascadeClassifier`, and every Haar Cascade stage
breaks. The first time it was an uncapped `opencv-python` in
requirements.txt. The second time requirements.txt was capped correctly
and `mediapipe` pulled in `opencv-contrib-python` with no upper bound —
a different distribution that installs over the same `cv2` package and
silently undid the cap, so the failure only appeared once the *optional*
ML dependencies were installed.

Neither case is catchable by importing cv2, because `import cv2`
succeeds either way. It has to be asserted about the API.
"""

import cv2


def test_cv2_still_has_the_cascade_classifier_api():
    """
    modules/face_detection.py and modules/object_detection.py are built
    entirely on this. It exists in OpenCV 4 and not in 5.
    """

    assert hasattr(cv2, "CascadeClassifier"), (
        f"cv2 {cv2.__version__} has no CascadeClassifier — an OpenCV 5 "
        "build has been installed over the 4.x one. Check the pins in "
        "requirements.txt and requirements-ml.txt."
    )

    assert hasattr(cv2.data, "haarcascades")


def test_cv2_is_a_major_version_the_app_supports():
    major = int(cv2.__version__.split(".")[0])

    assert major == 4, (
        f"cv2 resolved to {cv2.__version__}. Both requirements.txt "
        "(opencv-python) and requirements-ml.txt (opencv-contrib-python, "
        "via mediapipe) must stay capped below 5."
    )


def test_the_bundled_cascade_files_are_actually_present():
    """
    A cascade is loaded from a path inside the cv2 package, so a build
    that ships the API but not the data files would still fail at
    runtime.
    """

    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    classifier = cv2.CascadeClassifier(path)

    assert not classifier.empty()
