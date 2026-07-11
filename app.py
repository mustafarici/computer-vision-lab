import streamlit as st
from PIL import Image

st.set_page_config(page_title="Computer Vision Lab", layout="wide")

st.title("🧪 Computer Vision Lab")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)