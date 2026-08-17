import matplotlib.pyplot as plt

from modules.histogram import (
    build_color_histogram_figure,
    build_histogram_figure,
    compute_color_histograms,
    compute_histogram,
    get_figure_download_bytes,
)


def test_compute_histogram_shape_and_total(grayscale_image):
    histogram = compute_histogram(grayscale_image)

    assert histogram.shape == (256, 1)
    assert int(histogram.sum()) == grayscale_image.size


def test_build_histogram_figure_returns_figure(grayscale_image):
    histogram = compute_histogram(grayscale_image)

    figure = build_histogram_figure(histogram)

    assert figure.axes


def test_get_figure_download_bytes_is_png(grayscale_image):
    histogram = compute_histogram(grayscale_image)
    figure = build_histogram_figure(histogram)

    data = get_figure_download_bytes(figure)

    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_color_histograms_cover_all_three_channels(color_image):
    histograms = compute_color_histograms(color_image)

    assert list(histograms) == ["Red", "Green", "Blue"]

    for histogram in histograms.values():
        assert histogram.shape == (256, 1)
        assert int(histogram.sum()) == color_image.shape[0] * color_image.shape[1]


def test_color_histograms_are_empty_for_grayscale(grayscale_only_image):
    assert compute_color_histograms(grayscale_only_image) == {}


def test_color_histogram_counts_the_red_patch(color_image):
    """
    The fixture paints a pure-red rectangle, so the red channel must
    have more fully-saturated pixels than the green one — which also
    proves the channels aren't swapped.
    """

    histograms = compute_color_histograms(color_image)

    assert histograms["Red"][255] > histograms["Green"][255]


def test_color_histogram_figure_has_a_curve_per_channel(color_image):
    figure = build_color_histogram_figure(
        compute_color_histograms(color_image)
    )

    assert len(figure.axes[0].lines) == 3


def test_figures_do_not_leak_into_pyplot_registry(grayscale_image):
    """
    Regression test: build_histogram_figure used to call plt.subplots(),
    which registers every figure in pyplot's global list. Streamlit
    re-runs the script on each interaction, so that leaked one figure
    per rerun until matplotlib warned about it. A bare Figure isn't
    registered at all.
    """

    plt.close("all")
    histogram = compute_histogram(grayscale_image)

    for _ in range(25):
        build_histogram_figure(histogram)

    assert plt.get_fignums() == []
