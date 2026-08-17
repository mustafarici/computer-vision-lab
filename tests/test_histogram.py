import matplotlib.pyplot as plt

from modules.histogram import (
    build_histogram_figure,
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
