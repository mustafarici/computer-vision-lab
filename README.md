# 🧪 Computer Vision Lab

[![tests](https://github.com/mustafarici/computer-vision-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/mustafarici/computer-vision-lab/actions/workflows/tests.yml)

An interactive computer vision and image processing laboratory built with **Python**, **OpenCV**, **NumPy**, **Pillow**, **Matplotlib** and **Streamlit** — with optional deep-learning stages powered by **YOLOv8** and **MediaPipe**.

33 processing stages, from a grayscale conversion up to a neural network detecting 80 object classes, each with its own controls and a short explanation of what it does and why. Feed it a photo, a webcam snapshot, or a video clip.

<!-- Once deployed to Streamlit Community Cloud (see "Deploying" below),
     replace this comment with:
     **▶️ [Try it live](https://your-app-name.streamlit.app)** -->

![The app with an image loaded](assets/screenshot-app.jpg)

---

## 🚀 Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The two deep-learning stages are optional, because they pull in PyTorch and MediaPipe's runtime:

```bash
pip install -r requirements-ml.txt
python scripts/fetch_models.py   # optional: download the weights up front
```

Without them the app runs exactly as before — those two stages just show an install hint instead of a result.

---

## ✨ Features

### 📥 Three ways in

**Image** — upload a JPG or PNG. Large images are downscaled (long side capped at 1920px, aspect preserved) so nothing chokes on a 20MP photo, and both resolutions are reported when that happens.

**Camera** — take a snapshot from the webcam and process it in place. Nothing is uploaded anywhere; the frame is handled in memory.

**Video** — run any stage over every frame of a clip. A still image tells you what an operation does; a video tells you whether it holds up — detection that flickers between frames, or a threshold tuned for exactly one lighting condition, is invisible in a single frame and obvious in a few seconds of footage.

![Processing a video clip frame by frame](assets/screenshot-video.jpg)

Frame size, frame count and sampling stride are all capped, and the exact plan ("processing 240 of 1,431 frames, covering the first 9.6s") is stated *before* any work starts rather than discovered afterwards.

### 🔍 Before / after / difference

Every stage that returns an image the same size as the input can be viewed three ways: the result on its own, the original next to it, or the per-pixel difference between them.

![Side-by-side comparison of the original and Canny edges](assets/screenshot-side-by-side.jpg)

The difference view also quantifies what changed — the share of pixels the stage touched, and by how much on average. Those two numbers say different things: a contrast tweak nudges every pixel a little, a threshold rewrites most of them completely.

![The difference view showing what CLAHE changed](assets/screenshot-difference.jpg)

### 🧬 Custom pipelines

Every other stage starts from the original image, which is the right shape for learning one operation at a time and the wrong shape for the moment you want to ask "what if I equalize the contrast *first*, then threshold, then close the gaps?"

The Custom Pipeline stage lets you chain operations in whatever order you click them. Any grayscale conversion or threshold a step needs is inserted automatically, and the chain that actually ran is printed underneath the result — so there are no invalid combinations to discover, and no guessing about what the app silently did on your behalf.

![A custom chain: CLAHE, Otsu threshold, opening, contour detection](assets/screenshot-pipeline.jpg)

### 🖼️ Processing stages

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

![YOLOv8 finding all 18 people in a group photo](assets/screenshot-yolo.jpg)

**Histogram Analysis** — Grayscale histogram with mean/median/min/max statistics, and a per-channel color histogram

**Pipeline** — the chain builder described above

### 🎛️ Interactive controls

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

- **Bounded video work**: frame size, frame count and stride are capped before processing rather than after, so a long clip degrades into a shorter preview instead of an unresponsive page.

- **No figure leak**: histograms are built with matplotlib's object-oriented `Figure` rather than `plt.subplots()`, which would keep every figure ever rendered alive in pyplot's global registry — one per rerun.

- **Models loaded once**: Haar Cascades, YOLO weights and MediaPipe models are cached with `@st.cache_resource`, so they're read once per session rather than on every interaction.

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

560 tests covering the image operations themselves (threshold boundaries, morphology growing/shrinking the foreground, Otsu landing between two intensity clusters, bilateral filtering preserving an edge, contour area filtering, PNG round-trips), parameter validation, video framing and encoding, the sidebar schema, and every stage handler end-to-end. The sidebar is exercised through Streamlit's own `AppTest`, so a broken widget fails a test rather than only the running app.

The pipeline builder is tested over **every ordered pair of operations**, because "does this chain work" is a question about combinations, not about individual functions — an operation that quietly rejects a 3-channel array is only wrong in the orderings that hand it one.

A few tests assert about the *installed environment* rather than about this code, because the most expensive bug in this project's history was a dependency quietly resolving to OpenCV 5 — which removed `cv2.CascadeClassifier` and broke every Haar Cascade stage. `import cv2` succeeds either way, so it has to be asserted about the API.

CI runs three jobs on every push:

| Job | What it covers |
|---|---|
| `tests (core deps)` | Python 3.11 and 3.13, base requirements only |
| `tests (optional ML deps)` | The same suite with `ultralytics` and `mediapipe` really installed, so the YOLO and MediaPipe code paths run for real rather than only their "not installed" branch |
| `ruff` | Lint |

Locally, the same lint runs before each commit:

```bash
pre-commit install
```

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
networks take the color image, not a preprocessed one. The Custom
Pipeline stage ignores this diagram entirely: it runs whatever chain
you build, in whatever order you build it.)
```

---

## ☁️ Deploying

The repository is ready for [Streamlit Community Cloud](https://share.streamlit.io) as-is:

- `requirements.txt` pins every dependency below its next major release
- `packages.txt` installs the apt packages OpenCV and the video encoder need (`libgl1`, `libglib2.0-0`, `ffmpeg`)
- `.streamlit/config.toml` gives the deployed app the same theme as the local one

Point Streamlit Cloud at this repository with `app.py` as the entry point. The optional ML dependencies are deliberately left out of `requirements.txt`, so the deployed app starts fast and the two deep-learning stages show their install hint; move them across if you want YOLO in the hosted version too.

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
- [x] Before / after / difference comparison
- [x] Chained custom pipelines
- [x] Video and webcam input
- [x] Linting, pre-commit hooks, and CI coverage for the optional ML stages

### 🔭 Possible next steps

- Stages that take two images: template matching, ORB feature matching, image blending
- Segmentation: watershed, K-means color quantization, GrabCut
- Saving and sharing a custom pipeline as a preset
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
├── requirements-dev.txt      # Optional: pytest, ruff, pre-commit
├── packages.txt              # apt packages for Streamlit Cloud
├── ruff.toml
├── pytest.ini
├── .pre-commit-config.yaml
├── README.md
├── .gitignore
│
├── .github/workflows/        # CI: tests, optional-ML tests, lint
├── .streamlit/               # Theme and server config
├── assets/                   # Screenshots used by this README
├── images/                   # Sample images used for local testing
├── scripts/
│   └── fetch_models.py       # Pre-download the optional model weights
├── modules/                  # All processing logic, one file per topic
│   ├── __init__.py
│   ├── stages.py             # Stage registry: metadata + handler per stage
│   ├── controls.py           # Declarative sidebar schema
│   ├── pipeline.py           # Chainable operations + the chain runner
│   ├── compare.py            # Before/after and difference views
│   ├── video.py              # Frame-by-frame video processing
│   ├── image_utils.py        # Shared shape/representation helpers
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
└── results/                  # Locally saved processing results
```

### Adding a new stage

1. Write the operation in a module under `modules/` (or add it to a fitting one).
2. Add a handler and a `Stage(...)` entry in `modules/stages.py`.
3. If it needs parameters, add `Control(...)` entries to `modules/controls.py`.
4. If it makes sense as a chain step, add an `Operation(...)` entry to `modules/pipeline.py`.

The sidebar, the params dict, navigation, the comparison views, the video mode, the download button and the tests that run every handler all pick it up automatically.

---

## 👨‍💻 Author

**Mustafa Arıcı**
M.Sc. Student in Computational Science and Engineering
Izmir Institute of Technology

> This project is actively under development.
