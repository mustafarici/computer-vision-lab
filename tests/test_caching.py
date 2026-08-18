"""
Tests for the switchable cache decorator.

The point of `modules/caching.py` is that whether caching pays off is a
property of the *call site*, not of the function: the same Canny call
is worth caching from a slider and a net loss inside a video loop.
"""

import numpy as np

from modules.caching import (
    cache_data,
    caching_disabled,
    caching_is_disabled,
)


def _counting_function():
    """A cached function that records how often it really ran."""

    calls = []

    @cache_data(show_spinner=False, max_entries=4)
    def double(value: int) -> int:
        calls.append(value)
        return value * 2

    # Streamlit keys its cache on the function's module, qualified name
    # and code — all identical for every instance this factory makes —
    # so without clearing, one test's entries are another test's
    # unexplained cache hits.
    double.clear()

    return double, calls


def test_results_are_cached_by_default():
    double, calls = _counting_function()

    assert double(3) == 6
    assert double(3) == 6

    assert calls == [3], "the second call should have been a cache hit"


def test_caching_can_be_switched_off_for_a_block():
    double, calls = _counting_function()

    double(3)

    with caching_disabled():
        double(3)
        double(3)

    assert calls == [3, 3, 3]


def test_the_answer_is_the_same_either_way():
    """Bypassing the cache must change the cost, never the result."""

    double, _calls = _counting_function()

    cached = double(21)

    with caching_disabled():
        uncached = double(21)

    assert cached == uncached == 42


def test_caching_comes_back_after_the_block():
    double, calls = _counting_function()

    with caching_disabled():
        double(5)

    double(5)
    double(5)

    assert calls == [5, 5], "caching should be active again afterwards"


def test_the_flag_is_restored_even_if_the_block_raises():
    assert not caching_is_disabled()

    try:
        with caching_disabled():
            assert caching_is_disabled()
            raise ValueError("boom")

    except ValueError:
        pass

    assert not caching_is_disabled()


def test_nesting_does_not_leave_caching_off():
    with caching_disabled():
        with caching_disabled():
            assert caching_is_disabled()

        assert caching_is_disabled()

    assert not caching_is_disabled()


def test_the_uncached_original_stays_reachable():
    double, calls = _counting_function()

    assert double.uncached(4) == 8
    assert calls == [4]


def test_a_real_module_function_honours_the_switch():
    """
    Not just the toy above: the decorator is applied across modules/,
    so check one of the real ones still behaves.
    """

    from modules.edges import apply_canny

    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 255

    cached = apply_canny(image, 50, 150)

    with caching_disabled():
        uncached = apply_canny(image, 50, 150)

    assert np.array_equal(cached, uncached)
