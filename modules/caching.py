"""
A caching decorator that can be switched off for a whole code path.

`@st.cache_data` is a good trade for the interactive path: the user
drags a slider, the same expensive operation runs on the same image
over and over, and a cache hit costs ~1.5 ms instead of ~20 ms.

Video processing inverts every one of those assumptions. Each frame is
a *different* array, so every lookup misses; the cache still pays the
full cost of hashing a ~1.5 MB frame and copying the result out, and
`max_entries` means the entry it just stored is evicted before anything
could ever hit it. Measured on 60 frames of 720p through
grayscale → blur → Canny:

    with @st.cache_data     792 ms   (13.2 ms/frame)
    without                 606 ms   (10.1 ms/frame)   → 31% faster

So the caching decision isn't a property of the operation, it's a
property of how the operation is being called. This module makes that
switchable: modules use `@cache_data(...)` exactly as they used
`@st.cache_data(...)`, and a caller that knows caching cannot pay off
wraps its work in `caching_disabled()`.

The flag is a ContextVar rather than a plain global, so one session
processing a video can't turn caching off for another session's
sliders.
"""

import functools
from contextlib import contextmanager
from contextvars import ContextVar

import streamlit as st

_BYPASS: ContextVar[bool] = ContextVar("cvlab_cache_bypass", default=False)


def caching_is_disabled() -> bool:
    """True when the current code path has opted out of caching."""

    return _BYPASS.get()


@contextmanager
def caching_disabled():
    """
    Run a block with every `@cache_data` function computing directly.

    Use it where the inputs are known to be different every time, so a
    cache can only ever cost and never pay.
    """

    token = _BYPASS.set(True)

    try:
        yield

    finally:
        _BYPASS.reset(token)


def cache_data(**cache_kwargs):
    """
    Drop-in replacement for `st.cache_data(...)`.

    Returns a wrapper that dispatches, per call, to either the cached
    function or the original one. The decision is made at call time
    rather than at import time, which is the whole point — the same
    function is worth caching from the sidebar and not worth caching
    from a video loop.
    """

    def decorator(function):
        cached_function = st.cache_data(**cache_kwargs)(function)

        @functools.wraps(function)
        def dispatch(*args, **kwargs):
            if _BYPASS.get():
                return function(*args, **kwargs)

            return cached_function(*args, **kwargs)

        # Keep the Streamlit cache API reachable (tests and any future
        # "clear caches" control need it), and expose the uncached
        # original for anything that wants to measure the difference.
        dispatch.clear = cached_function.clear
        dispatch.uncached = function

        return dispatch

    return decorator
