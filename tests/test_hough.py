import cv2
import numpy as np

from modules.hough import apply_hough_circles, apply_hough_lines


def test_finds_a_drawn_line():
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(image, (20, 100), (180, 100), 255, 2)

    canvas, count = apply_hough_lines(image, 50, 150, 80, 50, 10)

    assert count >= 1
    assert canvas.shape == (200, 200, 3)


def test_finds_nothing_in_an_empty_image():
    blank = np.zeros((200, 200), dtype=np.uint8)

    canvas, count = apply_hough_lines(blank, 50, 150, 100, 50, 10)

    assert count == 0
    # Even with no detections the caller still gets a displayable image.
    assert canvas.shape == (200, 200, 3)


def test_line_vote_threshold_filters_results():
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(image, (20, 100), (180, 100), 255, 2)

    permissive = apply_hough_lines(image, 50, 150, 40, 20, 10)[1]
    strict = apply_hough_lines(image, 50, 150, 300, 20, 10)[1]

    assert strict <= permissive


def test_finds_a_drawn_circle():
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(image, (100, 100), 40, 255, 3)

    canvas, count = apply_hough_circles(image, 50, 30, 10, 80)

    assert count >= 1
    assert canvas.shape == (200, 200, 3)


def test_no_circles_in_an_empty_image():
    blank = np.zeros((200, 200), dtype=np.uint8)

    canvas, count = apply_hough_circles(blank, 50, 30, 10, 80)

    assert count == 0
    assert canvas.shape == (200, 200, 3)
