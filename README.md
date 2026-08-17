# 🧪 Computer Vision Lab

An interactive computer vision and image processing laboratory built with **Python**, **OpenCV**, **NumPy**, **Pillow**, **Matplotlib**, and **Streamlit** — with optional deep-learning stages powered by **YOLOv8** and **MediaPipe**.

This project explores computer vision techniques through an interactive web application: 32 processing stages, from a grayscale conversion up to a neural network detecting 80 object classes, each with its own controls and a short explanation of what it does and why.

---

## 🚀 Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The two deep-learning stages are optional, because they pull in PyTorch and MediaPipe's runtime:

```bash
pip install -r requirements-ml.txt
```

Without them the app runs exactly as before — those two stages just show an install hint instead of a result.

---

## ✨ Features

### 📷 Image Input

- Upload JPG, JPEG, and PNG images
- Automatic downscaling of large images (long side capped at 1920px, aspect ratio preserved) to keep processing fast
- Display of original vs. resized resolution when downscaling occurs
- Display of image resolution and channel information
- Automatic detection of color vs. grayscale-only input

### 🖼️ Processing Stages

**Basic** — Grayscale conversion, fixed-threshold binarization, Gaussian blur

**Thresholding** — Otsu's method (picks the threshold from the histogram automatically), adaptive thresholding (a separate threshold per neighbourhood, for unevenly lit images)

**Noise Filtering** — Median filter (removes salt-and-pepper noise), bilateral filter (edge-preserving smoothing)

**Contrast Enhancement** — Histogram equalization, CLAHE (contrast limited adaptive histogram equalization)

**Edge Detection** — Canny, Sobel, Laplacian

**Morphology** — Erosion, dilation, opening, closing

**Contours** — Contour detection with retrieval-mode, approximation-method and minimum-area controls

**Hough Transform** — Line and circle detection

**Color Analysis** — RGB channel separation, HSV channel separation, HSV-range color masking and thresholding

**Feature Detection** — Harris corners, ORB keypoints

**Classical Object Detection** — Face detection and multi-class detection (eyes, smile, full body, cat face, license plate) via Haar Cascades

**Deep Learning** *(optional)* — YOLOv8 object detection across 80 COCO classes with confidence and NMS controls; MediaPipe landmarks for a 468-point face mesh, 21-point hand skeletons, or 33-point body pose

**Histogram Analysis** — Grayscale histogram with mean/median/min/max statistics, and a per-channel color histogram

### 🎛️ Interactive Controls

Controls are declared as data in `modules/controls.py` and grouped into collapsible sidebar sections. The section belonging to whatever stage is on screen opens automatically; the rest stay collapsed, so you're not scrolling past Hough parameters while tuning a blur.

Every parameter updates the result immediately. Stages that need a color image show a clear explanation rather than an error when a grayscale-only image is uploaded, and Canny warns you if the lower/upper thresholds are inverted while still computing a sensible result.

### 🧭 Navigation

- Previous / Next navigation and a "Jump to a specific stage" dropdown
- Live stage counter
- Each stage shows its category, a short description, and an expandable "Processing Pipeline" breakdown
- **Download** any result as a PNG, or **save a copy to `results/`** directly when running locally

---

## ⚡ Performance

- **Lazy stage evaluation**: only the stage currently displayed is computed on each rerun. Stages that never touch the grayscale conversion (Original, RGB/HSV Channels, Color Mask/Threshold, the filters, YOLO, MediaPipe) skip it entirely.

- **Measured caching policy.** `@st.cache_data` isn't free: it hashes the full input array (~6 MB for a 1920×1080 image) and copies the result back out, costing ~1.5–2.5 ms per call. So caching is applied only where benchmarks showed it wins:

  | Operation | Raw compute | Cache hit | Cached? |
  |---|---|---|---|
  | Grayscale / threshold / blur / morphology | 0.1–0.6 ms | ~1.5–2 ms | ❌ cache cost more than the work |
  | Canny / Sobel / contours | 16–21 ms | ~1.5 ms | ✅ |
  | Harris / ORB / color thresholding | 28–92 ms | ~2–4 ms | ✅ |
  | Bilateral filter / Hough / CLAHE | 10–200 ms | ~2–4 ms | ✅ |
  | Haar Cascade detection | ~700 ms | ~2 ms | ✅ |
  | PNG encoding for the download button | ~46 ms | ~2.6 ms | ✅ |

- **Bounded caches**: every cache sets `max_entries`, so dragging a slider can't accumulate one full-size image per position. (Previously, sweeping the 0–255 threshold slider alone could cache ~500 MB of images.)

- **No figure leak**: histograms are built with matplotlib's object-oriented `Figure` rather than `plt.subplots()`, which would keep every figure ever rendered alive in pyplot's global registry — one per rerun.

- **Models loaded once**: Haar Cascades, YOLO weights and MediaPipe models are cached with `@st.cache_resource`, so they're read once per session rather than on every interaction.

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

162 tests covering the image operations themselves (threshold boundaries, morphology growing/shrinking the foreground, Otsu landing between two intensity clusters, bilateral filtering preserving an edge, contour area filtering, PNG round-trips), parameter validation, the sidebar schema, and every stage handler end-to-end. The sidebar is exercised through Streamlit's own `AppTest`, so a broken widget fails a test rather than only the running app.

Tests run on every push via GitHub Actions (`.github/workflows/tests.yml`) against Python 3.11 and 3.13.

---

## 🔬 Processing Pipeline

```text
                                 Input Image
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
         Grayscale            RGB / HSV Channels      Median / Bilateral
              │                       │                    Filtering
   ┌──────────┼───────────┐           ▼
   │          │           │    Color Thresholding
   │          │           │      (Mask + Result)
   │          │           │
   │          │           ├──► Otsu / Adaptive Threshold
   │          │           ├──► Histogram Equalization / CLAHE
   │          │           ├──► Sobel / Laplacian
   │          │           ├──► Harris Corners / ORB Keypoints
   │          │           ├──► Haar Cascade (faces, objects)
   │          │           └──► Histogram
   │          ▼
   │      Gaussian Blur
   │          │
   │          ▼
   │       Canny Edge ──────► Hough Lines
   │       Detection
   ▼
Binary Image
   │
   ├───────────┬────────────┬────────────┐
   ▼           ▼            ▼            ▼
Erosion    Dilation      Opening      Closing
   │
   ▼
Contour Detection

(The original image also feeds YOLO and MediaPipe directly — neural
networks take the color image, not a preprocessed one.)
```

---

## 📅 Development Roadmap

- [x] Project setup, GitHub repository, Streamlit application
- [x] Image upload and grayscale conversion
- [x] Histogram analysis and image thresholding
- [x] Gaussian blur, Canny / Sobel / Laplacian edge detection
- [x] Interactive sidebar controls and image navigation
- [x] Morphological operations and contour detection
- [x] RGB / HSV analysis and color thresholding
- [x] Feature detection (Harris corners, ORB keypoints)
- [x] Face detection and multi-class object detection (Haar Cascade)
- [x] Modular architecture with a stage registry and declarative sidebar
- [x] Test suite and continuous integration
- [x] Otsu and adaptive thresholding
- [x] Median and bilateral filtering
- [x] Histogram equalization and CLAHE
- [x] Hough line and circle detection
- [x] Color histogram
- [x] YOLOv8 object detection
- [x] MediaPipe face mesh, hands and pose

### 🔭 Possible next steps

- Side-by-side / before-after comparison view
- Stages that take two images: template matching, ORB feature matching, image blending
- Segmentation: watershed, K-means color quantization, GrabCut
- Webcam and video processing
- Training and running a custom model instead of only pre-trained ones

---

## 🛠️ Technologies

- 🐍 **Python**
- 👁️ **OpenCV**
- 🌐 **Streamlit**
- 🔢 **NumPy**
- 🖼️ **Pillow**
- 📊 **Matplotlib**
- 🤖 **Ultralytics YOLOv8** *(optional)*
- ✋ **MediaPipe** *(optional)*

---

## 📂 Project Structure

```text
computer-vision-lab/
│
├── app.py                    # Streamlit entry point / UI shell
├── requirements.txt          # Core dependencies
├── requirements-ml.txt       # Optional: YOLO + MediaPipe
├── requirements-dev.txt      # Optional: pytest
├── pytest.ini
├── README.md
├── .gitignore
│
├── .github/workflows/        # CI: runs the test suite on every push
├── images/                   # Sample images used for local testing
├── modules/                  # All processing logic, one file per topic
│   ├── __init__.py
│   ├── stages.py             # Stage registry: metadata + handler per stage
│   ├── controls.py           # Declarative sidebar schema
│   ├── io_utils.py
│   ├── basic_ops.py
│   ├── thresholding.py
│   ├── filters.py
│   ├── enhancement.py
│   ├── edges.py
│   ├── morphology.py
│   ├── contours.py
│   ├── hough.py
│   ├── color_analysis.py
│   ├── color_threshold.py
│   ├── feature_detection.py
│   ├── face_detection.py
│   ├── object_detection.py
│   ├── deep_detection.py     # YOLO + MediaPipe (optional dependencies)
│   └── histogram.py
├── tests/                    # pytest suite
├── models/                   # Downloaded model weights (git-ignored)
├── assets/                   # Static UI assets
└── results/                  # Locally saved processing results
```

### Adding a new stage

1. Write the operation in a module under `modules/` (or add it to a fitting one).
2. Add a handler and a `Stage(...)` entry in `modules/stages.py`.
3. If it needs parameters, add `Control(...)` entries to `modules/controls.py`.

The sidebar, the params dict, navigation, the download button and the tests that run every handler all pick it up automatically.

---

## 👨‍💻 Author

**Mustafa Arıcı**
M.Sc. Student in Computational Science and Engineering
Izmir Institute of Technology

> This project is actively under development.
