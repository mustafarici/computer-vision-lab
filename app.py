import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Computer Vision Lab",
    layout="wide"
)

st.title("🧪 Computer Vision Lab")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Kullanıcının yüklediği resmi Pillow ile aç
    image = Image.open(uploaded_file)

    # Pillow görüntüsünü NumPy dizisine çevir
    image_np = np.array(image)

    # OpenCV ile gri tonlamaya çevir
    grayscale = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )

    # Grayscale histogramını hesapla
    histogram = cv2.calcHist(
        [grayscale],
        [0],
        None,
        [256],
        [0, 256]
    )

    # Orijinal ve grayscale görüntüleri yan yana göster
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.subheader("Grayscale Image")
        st.image(
            grayscale,
            use_container_width=True
        )

    # Histogram grafiğini oluştur
    fig, ax = plt.subplots(figsize=(5, 2.5))

    ax.plot(histogram)
    ax.set_title("Grayscale Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim([0, 256])

    # Histogramı görüntülerin altında göster
    st.subheader("Grayscale Histogram")
    st.pyplot(fig, use_container_width=False)