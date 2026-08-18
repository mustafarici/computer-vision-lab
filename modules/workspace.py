"""
Scratch space on disk for one browser session.

Two problems live here, both of which only bite once the app is
deployed somewhere other than your own laptop.

**Filenames from the client are not names, they are paths.** The video
uploader used to write to `tempfile.gettempdir() / f"cvlab_input_{name}"`
with `name` taken straight from the upload. A browser normally sends
only the basename, but nothing forces a client to be a browser, and
`Path("/tmp") / "cvlab_input_../../../etc/x"` resolves cheerfully
outside `/tmp`. On a hosted deployment the application directory is
writable and Streamlit reloads on file change, so an arbitrary write is
one short step from arbitrary code execution. `safe_filename` exists so
no caller has to remember this.

**Temporary files are not temporary unless someone deletes them.** Each
processed clip wrote a fresh `mkdtemp()` that nothing ever removed, and
uploads sat in `/tmp` forever, with a 200 MB upload cap above them.
That is a slow disk-exhaustion bug on any long-running deployment. A
session gets one directory, reused and cleared between runs, and
directories left behind by sessions that ended are swept on the way in.
"""

import re
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st

# Directories created here all share this prefix, which is what makes
# the sweep able to tell its own leftovers from everything else in the
# system temp directory.
WORKSPACE_PREFIX = "cvlab_session_"

# Sessions that haven't touched their directory in this long are
# assumed to be over. Generous, because the cost of being wrong is a
# user losing a processed clip they might still be looking at.
STALE_AFTER_SECONDS = 6 * 60 * 60

# Anything outside this is replaced. Deliberately a whitelist: a
# blacklist of "../" and "/" would still let through the next
# separator, encoding or normalisation trick.
_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]")

_SESSION_KEY = "_workspace_directory"


def safe_filename(name: object, fallback: str = "upload") -> str:
    """
    Reduce a client-supplied filename to a plain, single path segment.

    Never returns something containing a separator, and never returns
    an empty string or a name starting with a dot, so the result can be
    joined onto a directory without escaping it.
    """

    # PurePath.name drops every directory component, including the ones
    # a caller was hoping we'd keep.
    candidate = Path(str(name)).name

    candidate = _UNSAFE_CHARACTERS.sub("_", candidate)

    # Leading dots would produce hidden files, and "." / ".." survive
    # the character filter untouched.
    candidate = candidate.lstrip(".")

    return candidate or fallback


def create_workspace(root: Path | None = None) -> Path:
    """Make a fresh, empty directory for one session's files."""

    root = Path(root) if root is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)

    return Path(tempfile.mkdtemp(prefix=WORKSPACE_PREFIX, dir=root))


def clear_workspace(directory: Path) -> Path:
    """
    Empty a workspace without discarding it.

    Called before each run so the previous clip's frames and encodes
    don't accumulate across a session.
    """

    directory = Path(directory)

    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def purge_stale_workspaces(
    root: Path | None = None,
    max_age_seconds: int = STALE_AFTER_SECONDS,
    now: float | None = None,
) -> int:
    """
    Delete workspaces left behind by sessions that have ended.

    Returns how many were removed. Errors are swallowed on purpose:
    this is housekeeping, and failing to tidy up is never a reason to
    refuse to process the user's video.
    """

    root = Path(root) if root is not None else Path(tempfile.gettempdir())
    now = time.time() if now is None else now

    removed = 0

    if not root.is_dir():
        return 0

    for candidate in root.glob(f"{WORKSPACE_PREFIX}*"):
        if not candidate.is_dir():
            continue

        try:
            age = now - candidate.stat().st_mtime

        except OSError:
            continue

        if age < max_age_seconds:
            continue

        shutil.rmtree(candidate, ignore_errors=True)

        if not candidate.exists():
            removed += 1

    return removed


def store_upload(data: bytes, name: object, directory: Path) -> Path:
    """
    Write uploaded bytes into `directory` under a sanitized name.

    Asserts the obvious afterwards, because "the sanitizer is correct"
    is exactly the belief that path traversal bugs are made of.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    destination = directory / safe_filename(name)

    resolved_directory = directory.resolve()
    resolved_destination = destination.resolve()

    if resolved_destination.parent != resolved_directory:
        raise ValueError(
            f"Refusing to write outside the workspace: {name!r}."
        )

    destination.write_bytes(data)

    return destination


def session_workspace() -> Path:
    """
    The current session's workspace, created on first use.

    Stored in session state rather than in a cache, because it must be
    one directory per browser session — a `@st.cache_resource` value is
    shared by every session on the server, which is the opposite of
    what's wanted for user-uploaded files.
    """

    existing = st.session_state.get(_SESSION_KEY)

    if existing is not None and Path(existing).is_dir():
        return Path(existing)

    # Sweeping here rather than on a timer keeps it to code that only
    # runs when someone is actually using the video features.
    purge_stale_workspaces()

    directory = create_workspace()
    st.session_state[_SESSION_KEY] = str(directory)

    return directory
