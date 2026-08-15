# 🧪 Computer Vision Lab

An interactive computer vision and image processing laboratory built with **Python**, **OpenCV**, **NumPy**, **Pillow**, **Matplotlib**, and **Streamlit**.

This project is designed to explore fundamental computer vision and image processing techniques through an interactive web application. New algorithms and processing methods will be added incrementally as the project evolves.

---

## ✨ Current Features

### 📷 Image Input

- Upload JPG, JPEG, and PNG images
- Automatic downscaling of large images (long side capped at 1920px, aspect ratio preserved) to keep processing fast
- Display of original vs. resized resolution when downscaling occurs
- Display of image resolution and channel information
- Automatic detection of color vs. grayscale-only input
- Interactive image preview

### 🖼️ Image Processing

- Grayscale Conversion
- Binary Image Thresholding
- Gaussian Blur
- Canny Edge Detection (with automatic lower/upper threshold ordering)
- Sobel Edge Detection
- Laplacian Edge Detection
- Morphological Operations (Erosion, Dilation, Opening, Closing)
- Contour Detection
- RGB Channel Analysis
- HSV Color Space Analysis
- HSV Range Color Thresholding
- Feature Detection (Harris Corners, ORB Keypoints)
- Face Detection (Haar Cascade)
- Object Detection (Eyes, Smile, Full Body, Cat Face, License Plate — Haar Cascade)
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
- **🔎 Object Detection** — Object Class, Scale Factor, Minimum Neighbors, Minimum Object Size
- **ℹ️ Image Information** — resolution, channel count, color/grayscale type, resize notice

Each parameter can be adjusted dynamically and the result is updated immediately. Stages that require a color image (RGB/HSV analysis, color thresholding) show a clear warning instead of an error when a grayscale-only image is uploaded. If the Canny lower/upper thresholds are set in the wrong order, the sidebar warns you but the computation still runs correctly with the values auto-sorted.

### 🧭 Image Navigation

- Previous / Next navigation
- "Jump to a specific stage" dropdown for direct access to any processing step
- Live stage counter (e.g. `19 / 21`)
- Each stage displays its category, a short description, and an expandable "Processing Pipeline" breakdown
- **Download button** on every stage to save the current result as a PNG (the histogram downloads as its plotted figure)

### ⚡ Performance

- **Lazy stage evaluation**: only the currently displayed stage is computed on each rerun, instead of the full pipeline — moving one slider no longer recomputes unrelated stages.
- Expensive operations (Harris response, Gaussian blur, etc.) are cached independently with `@st.cache_data`.
- Models/resources (Haar Cascade detectors) are cached with `@st.cache_resource` and loaded only once per session, instead of on every rerun.
- Large uploads are automatically downscaled before processing, keeping CPU-heavy operations (Haar Cascade, Harris, Canny) responsive even on 12MP+ photos.

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
   │           │           ├──► Face Detection (Haar Cascade)
   │           │           └──► Object Detection (Haar Cascade)
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

The following features are potential directions for future development:

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
- [x] Object detection (Haar Cascade, multi-class)

All originally planned features are complete. 🎉

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
│   ├── object_detection.py
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