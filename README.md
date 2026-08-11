# 🧪 Computer Vision Lab

An interactive computer vision and image processing laboratory built with **Python**, **OpenCV**, **NumPy**, **Pillow**, **Matplotlib**, and **Streamlit**.

This project is designed to explore fundamental computer vision and image processing techniques through an interactive web application. New algorithms and processing methods will be added incrementally as the project evolves.

---

## ✨ Current Features

### 📷 Image Input

- Upload JPG, JPEG, and PNG images
- Preserve the original image resolution
- Display image resolution and channel information
- Interactive image preview

### 🖼️ Image Processing

- Grayscale Conversion
- Binary Image Thresholding
- Gaussian Blur
- Canny Edge Detection
- Sobel Edge Detection
- Laplacian Edge Detection
- Morphological Operations (Erosion, Dilation, Opening, Closing)
- Contour Detection
- Grayscale Histogram Analysis

### 🎛️ Interactive Controls

The application provides interactive controls through the sidebar:

- **Threshold Value**
- **Gaussian Blur Kernel Size**
- **Canny Lower / Upper Threshold**
- **Sobel Kernel Size**
- **Laplacian Kernel Size**
- **Morphological Structuring Element Size & Iterations**
- **Contour Retrieval Mode & Approximation Method**
- **Minimum Contour Area**

Each parameter can be adjusted dynamically and the result is updated immediately.

### 🧭 Image Navigation

- Previous / Next navigation
- "Jump to a specific stage" dropdown for direct access to any processing step
- Live stage counter (e.g. `4 / 13`)
- Individual visualization of each processing result

---

## 🔬 Current Processing Pipeline

```text
                                 Input Image
                                      │
                                 Grayscale
                                      │
        ┌───────────┬───────────┬────┴─────┬────────────┬────────────┐
        │           │           │          │            │            │
        ▼           ▼           ▼          ▼            ▼            ▼
    Threshold    Gaussian     Sobel     Laplacian    Histogram    (feeds
        │          Blur                                           binary
        │           │                                            branch)
        │           ▼
        │        Canny Edge
        │        Detection
        ▼
   Binary Image
        │
        ├────────────┬─────────────┬─────────────┐
        ▼            ▼             ▼             ▼
    Erosion      Dilation       Opening       Closing
        │
        ▼
   Contour Detection
```

---

## 🚀 Planned Features

The following features are planned for future development:

- RGB Channel Analysis
- HSV Color Space Analysis
- Color Thresholding
- Feature Detection
- Face Detection
- Object Detection
- Additional Image Filtering Techniques

---

## 📅 Development Roadmap

- [x] Project setup
- [x] GitHub repository
- [x] Streamlit application
- [x] Image upload
- [x] Grayscale conversion
- [x] Histogram analysis
- [x] Image thresholding
- [x] Gaussian blur
- [x] Canny edge detection
- [x] Interactive sidebar controls
- [x] Image navigation
- [x] Sobel edge detection
- [x] Laplacian edge detection
- [x] Morphological operations
- [x] Contour detection
- [ ] RGB / HSV analysis
- [ ] Feature detection
- [ ] Face detection
- [ ] Object detection

---

## 🛠️ Technologies

- 🐍 **Python**
- 👁️ **OpenCV**
- 🌐 **Streamlit**
- 🔢 **NumPy**
- 🖼️ **Pillow**
- 📊 **Matplotlib**

---

## 📂 Project Structure

```text
computer-vision-lab/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── images/
├── modules/
│   ├── __init__.py
│   ├── io_utils.py
│   ├── basic_ops.py
│   ├── edges.py
│   ├── morphology.py
│   ├── contours.py
│   └── histogram.py
├── outputs/
└── pages/
```

---

## 👨‍💻 Author

**Mustafa Arıcı**
M.Sc. Student in Computational Science and Engineering
Izmir Institute of Technology

> This project is actively under development.