"""
Tests for the session scratch space.

These are mostly security tests. The video uploader used to build its
destination path by interpolating the client-supplied filename
straight into a temp path, which meant a request that didn't come from
a browser could choose where the app wrote. On a hosted deployment the
application directory is writable and Streamlit reloads on file
change, so "choose where the app writes" is a short walk from "choose
what the app runs".
"""

import time

import pytest

from modules.workspace import (
    WORKSPACE_PREFIX,
    clear_workspace,
    create_workspace,
    purge_stale_workspaces,
    safe_filename,
    store_upload,
)

# Things a hostile (or merely broken) client might send as a filename.
HOSTILE_NAMES = [
    "../../../../etc/passwd",
    "../app.py",
    "a/../../b.mp4",
    "/absolute/path.mp4",
    "..",
    ".",
    "",
    "....//....//x.mp4",
    "clip\x00.mp4",
    "sub/dir/clip.mp4",
    "\\windows\\style\\path.mp4",
]


# ==================================================
# FILENAME SANITIZING
# ==================================================


@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_a_sanitized_name_is_a_single_harmless_segment(name):
    result = safe_filename(name)

    assert result
    assert "/" not in result
    assert "\\" not in result
    assert "\x00" not in result
    # No traversal, and no hidden files either.
    assert not result.startswith(".")
    assert ".." not in result


def test_an_ordinary_filename_survives_recognizably():
    assert safe_filename("holiday clip-2.mp4") == "holiday_clip-2.mp4"


def test_an_unusable_name_falls_back():
    assert safe_filename("...", fallback="upload") == "upload"


def test_a_non_string_name_is_handled():
    """UploadedFile.name is whatever the client sent; it needn't be a str."""

    assert safe_filename(None)
    assert safe_filename(12345) == "12345"


# ==================================================
# WRITING UPLOADS
# ==================================================


@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_an_upload_always_lands_inside_the_workspace(tmp_path, name):
    workspace = create_workspace(tmp_path)

    destination = store_upload(b"data", name, workspace)

    assert destination.resolve().parent == workspace.resolve()
    assert destination.read_bytes() == b"data"


def test_an_upload_cannot_escape_even_one_level(tmp_path):
    """The specific bug: /tmp/cvlab_input_../../etc/x resolved to /etc/x."""

    workspace = create_workspace(tmp_path)
    sibling = tmp_path / "not_the_workspace"
    sibling.mkdir()

    store_upload(b"data", "../not_the_workspace/stolen.mp4", workspace)

    assert list(sibling.iterdir()) == []


def test_two_uploads_with_the_same_name_do_not_multiply(tmp_path):
    workspace = create_workspace(tmp_path)

    store_upload(b"first", "clip.mp4", workspace)
    store_upload(b"second", "clip.mp4", workspace)

    assert len(list(workspace.iterdir())) == 1


# ==================================================
# CLEANUP
# ==================================================


def test_clearing_a_workspace_empties_it_but_keeps_it(tmp_path):
    workspace = create_workspace(tmp_path)
    (workspace / "old.mp4").write_bytes(b"x")
    (workspace / "nested").mkdir()

    cleared = clear_workspace(workspace)

    assert cleared.is_dir()
    assert list(cleared.iterdir()) == []


def test_stale_workspaces_are_swept(tmp_path):
    old = create_workspace(tmp_path)
    fresh = create_workspace(tmp_path)

    # Age the first one an hour past the cutoff.
    stale_time = time.time() - 7200
    import os

    os.utime(old, (stale_time, stale_time))

    removed = purge_stale_workspaces(tmp_path, max_age_seconds=3600)

    assert removed == 1
    assert not old.exists()
    assert fresh.exists()


def test_the_sweep_only_touches_its_own_directories(tmp_path):
    """
    This runs against the shared system temp directory, so a bug here
    would delete other programs' files, not just ours.
    """

    innocent = tmp_path / "someone_elses_data"
    innocent.mkdir()

    stale_time = time.time() - 100000
    import os

    os.utime(innocent, (stale_time, stale_time))

    purge_stale_workspaces(tmp_path, max_age_seconds=1)

    assert innocent.exists()


def test_the_sweep_is_quiet_when_there_is_nothing_to_sweep(tmp_path):
    assert purge_stale_workspaces(tmp_path / "missing") == 0


def test_workspaces_are_named_so_the_sweep_can_find_them(tmp_path):
    workspace = create_workspace(tmp_path)

    assert workspace.name.startswith(WORKSPACE_PREFIX)
