"""
Tests for the deployment manifests.

`packages.txt` is not a config file that something parses politely —
Streamlit Community Cloud pipes it straight into `apt-get install`
through `xargs`. A comment in it is not ignored; it is nineteen
imaginary package names, and the deploy fails before a line of Python
runs:

    E: Unable to locate package #
    E: Unable to locate package opencv-python-headless,

Which is how this file came to exist: the explanation of *why* each
package was needed went into the manifest, and the explanation is what
broke the deploy. Prose belongs in the README. This test keeps the
manifest to bare package names.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

PACKAGES = ROOT / "packages.txt"

# Debian package names: lowercase alphanumerics plus - + . :, at least
# two characters. Deliberately strict — anything surprising in this
# file is far more likely to be a mistake than an exotic package.
PACKAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9+.\-]+[a-z0-9+.\-:]*$")

REQUIREMENTS = [
    ROOT / "requirements.txt",
    ROOT / "requirements-dev.txt",
    ROOT / "requirements-ml.txt",
]


def test_packages_txt_exists():
    assert PACKAGES.is_file()


def test_packages_txt_contains_only_package_names():
    """No comments, no blank-line padding, no explanations."""

    lines = PACKAGES.read_text().splitlines()

    assert lines, "an empty packages.txt is better deleted than empty"

    for number, line in enumerate(lines, start=1):
        assert line == line.strip(), f"line {number} has surrounding space"
        assert not line.startswith("#"), (
            f"line {number} is a comment; apt will treat it as a package "
            "name and the deploy will fail. Explain it in the README."
        )
        assert PACKAGE_NAME.match(line), (
            f"line {number} ({line!r}) doesn't look like a package name"
        )


def test_the_app_still_needs_ffmpeg():
    """
    The one package that isn't optional: without it the processed video
    can only be downloaded, not played in the page.
    """

    assert "ffmpeg" in PACKAGES.read_text().split()


# ==================================================
# REQUIREMENTS
# ==================================================
#
# These *do* take comments — pip understands them — but every
# dependency has been capped for a reason, twice now at the cost of a
# broken build.


@pytest.mark.parametrize(
    "path", REQUIREMENTS, ids=lambda path: path.name
)
def test_every_requirement_has_an_upper_bound(path):
    for line in path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        assert "<" in line, (
            f"{path.name}: {line!r} has no upper bound. OpenCV 5 removed "
            "cv2.CascadeClassifier and shipped through an uncapped "
            "dependency twice; nothing here goes unpinned."
        )


def test_opencv_is_capped_consistently_in_both_files():
    """
    opencv-python-headless and opencv-contrib-python install over the
    same `cv2` package. If their caps ever disagree, installing the
    optional ML extras silently replaces the pinned OpenCV with an
    unpinned one — which is exactly how OpenCV 5 got in the second time.
    """

    base = (ROOT / "requirements.txt").read_text()
    ml = (ROOT / "requirements-ml.txt").read_text()

    caps = re.findall(r"opencv[a-z-]*(>=[\d.]+,<\d+)", base + ml)

    assert len(caps) == 2, caps
    assert caps[0] == caps[1], caps
