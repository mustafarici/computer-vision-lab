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
- Grayscale Histogram Analysis

### 🎛️ Interactive Controls

The application provides interactive controls through the sidebar:

- **Threshold Value**
- **Gaussian Blur Kernel Size**
- **Canny Lower Threshold**
- **Canny Upper Threshold**

Each parameter can be adjusted dynamically and the result is updated immediately.

### 🧭 Image Navigation

- Previous / Next navigation
- Interactive processing-stage indicators
- Hover tooltips for each processing stage
- Individual visualization of each processing result

---

## 🔬 Current Processing Pipeline

```text
                    Input Image
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      Grayscale      Thresholding   Gaussian Blur
          │                             │
          │                             ▼
          │                       Canny Edge
          │                       Detection
          │
          ▼
   Grayscale Histogram
```

---

## 🚀 Planned Features

The following features are planned for future development:

- Laplacian Edge Detection
- Morphological Operations
  - Erosion
  - Dilation
  - Opening
  - Closing
- Contour Detection
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
- [ ] Laplacian edge detection
- [ ] Morphological operations
- [ ] Contour detection
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
├── outputs/
└── pages/
```

---

## 👨‍💻 Author

**Mustafa Arıcı**
M.Sc. Student in Computational Science and Engineering
Izmir Institute of Technology

> This project is actively under development.