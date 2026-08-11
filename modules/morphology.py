"""Erosion, dilation, opening and closing."""

import cv2
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def apply_erosion(binary: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(binary, kernel, iterations=iterations)


@st.cache_data(show_spinner=False)
def apply_dilation(binary: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(binary, kernel, iterations=iterations)


@st.cache_data(show_spinner=False)
def apply_opening(binary: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """Erosion followed by dilation — removes small noise/specks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)


@st.cache_data(show_spinner=False)
def apply_closing(binary: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """Dilation followed by erosion — closes small holes/gaps."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
