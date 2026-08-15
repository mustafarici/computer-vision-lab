# 🧪 Computer Vision Lab

An interactive computer vision and image processing laboratory built with **Python**, **OpenCV**, **NumPy**, **Pillow**, **Matplotlib**, and **Streamlit**.

This project is designed to explore fundamental computer vision and image processing techniques through an interactive web application. New algorithms and processing methods will be added incrementally as the project evolves.

---

## ✨ Current Features

### 📷 Image Input

- Upload JPG, JPEG, and PNG images
- Preserve the original image resolution
- Display image resolution and channel information
- Automatic detection of color vs. grayscale-only input
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
- RGB Channel Analysis
- HSV Color Space Analysis
- HSV Range Color Thresholding
- Feature Detection (Harris Corners, ORB Keypoints)
- Face Detection (Haar Cascade)
- Grayscale Histogram Analysis (with mean / median / min / max statistics)

### 🎛️ Interactive Controls

Sidebar controls are grouped into collapsible sections by category:

- **🔧 Basic Processing** — Threshold Value, Gaussian Blur Kernel Size
- **📐 Edge Detection** — Canny Lower/Upper Threshold, Sobel Kernel Size, Laplacian Kernel Size
- **🧩 Morphological Operations** — Structuring Element Size, Iterations
- **🔲 Contour Detection** — Retrieval Mode, Approximation Method, Minimum Contour Area
- **🎨 Color Analysis** — Hue / Saturation / Value Ranges (requires a color image)
- **🔍 Feature Detection** — Harris Block Size, Sobel Aperture, Sensitivity, Response Threshold, ORB Max Keypoints
- **😀 Face Detection** — Scale Factor, Minimum Neighbors, Minimum Face Size
- **ℹ️ Image Information** — resolution, channel count, color/grayscale type

Each parameter can be adjusted dynamically and the result is updated immediately. Stages that require a color image (RGB/HSV analysis, color thresholding) show a clear warning instead of an error when a grayscale-only image is uploaded.

### 🧭 Image Navigation

- Previous / Next navigation
- "Jump to a specific stage" dropdown for direct access to any processing step
- Live stage counter (e.g. `19 / 20`)
- Each stage displays its category, a short description, and an expandable "Processing Pipeline" breakdown

### ⚡ Performance

- **Lazy stage evaluation**: only the currently displayed stage is computed on each rerun, instead of the full pipeline — moving one slider no longer recomputes unrelated stages.
- Expensive operations (Harris response, Gaussian blur, etc.) are cached independently with `@st.cache_data`.
- Models/resources (the Haar Cascade face detector) are cached with `@st.cache_resource` and loaded only once per session, instead of on every rerun.

---

## 🔬 Current Processing Pipeline

```text
                                 Input Image
                                      │
                          ┌───────────┴───────────┐
                          ▼                        ▼
                     Grayscale          RGB / HSV Channels
                          │                        │
   ┌───────────┬──────────┼────────────┬───────────┴──┐
   │           │          │            │              ▼
   ▼           ▼          ▼            ▼        Color Thresholding
Threshold  Gaussian    Sobel       Laplacian    (Mask + Result)
   │          Blur         │
   │           │           ├──► Harris Corners
   │           │           ├──► ORB Keypoints
   │           │           └──► Face Detection (Haar Cascade)
   │           ▼
   │        Canny Edge
   │        Detection
   ▼
Binary Image
   │
   ├───────────┬────────────┬────────────┐
   ▼           ▼            ▼            ▼
Erosion    Dilation      Opening      Closing
   │
   ▼
Contour Detection

(Grayscale also feeds the Histogram, in parallel with the above.)
```

---

## 🚀 Planned Features

The following features are planned for future development:

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
- [x] RGB / HSV analysis
- [x] Color thresholding
- [x] Feature detection (Harris corners, ORB keypoints)
- [x] Face detection (Haar Cascade)
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
│   ├── color_analysis.py
│   ├── color_threshold.py
│   ├── feature_detection.py
│   ├── face_detection.py
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