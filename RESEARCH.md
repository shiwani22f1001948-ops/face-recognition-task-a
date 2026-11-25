# Research

> **Before submitting:** open every link, confirm it says what is claimed, and replace the "my test" rows with
> numbers *you* measured on your own practice data. Delete anything you did not actually read.

## Problem framing
Not "train a model" but **face verification at scale 1:50 with an open set** (some faces are not in the gallery),
one example image per person, fully offline. That points to *pre-trained embeddings + nearest-neighbour + a rejection threshold*.

## Options compared
| Option | Detector / embedder | Offline? | Install effort | Notes | Verdict |
|---|---|---|---|---|---|
| **face_recognition (dlib)** — https://github.com/ageitgey/face_recognition , http://dlib.net | HOG or CNN / ResNet 128-d | Yes, models ship in `face_recognition_models` (pip) | Low with `dlib-bin` wheels | README reports ~99.38% on LFW; HOG weak on non-frontal faces | **Chosen** |
| InsightFace (ArcFace, RetinaFace) — https://github.com/deepinsight/insightface | RetinaFace / ArcFace 512-d | Yes after one-time model download | Higher: needs C++ build tools on Windows, model download | Better accuracy, esp. angles | Rejected for risk; listed as upgrade path |
| OpenCV Zoo YuNet + SFace — https://github.com/opencv/opencv_zoo | YuNet / SFace | Yes, tiny ONNX files | Low, but weights are stored via Git LFS → extra manual download step | Light, fast | Runner-up |
| DeepFace wrapper — https://github.com/serengil/deepface | many backends | Weights downloaded on first use | Pulls TensorFlow (heavy) | Convenient, but heavy + first-run download | Rejected |
| FaceNet / MTCNN (facenet-pytorch) — papers: https://arxiv.org/abs/1503.03832 , https://arxiv.org/abs/1604.02878 | MTCNN / Inception-ResNet | Weights downloaded | PyTorch (heavy) | Good, but heavy | Rejected |
| Cloud APIs (Rekognition, Gemini, ...) | — | **No** | — | Forbidden by the brief | Rejected |
| Train own classifier | — | — | — | 1 image/person; unseen people at demo | Rejected |

Background reading: ArcFace https://arxiv.org/abs/1801.07698 (why angular-margin embeddings separate identities well);
LFW benchmark http://vis-www.cs.umass.edu/lfw/ .

## Data sources for practice
* **LFW** (public, 13k celebrity photos) → `tools/make_demo_data.py` builds the 50-person folder and test collages.
* Your own phone photos of consenting friends/family for realistic group photos (best test of the demo).
* AI-generated faces (allowed by the brief for *practice data only*).

## Decisions that came from the research
* One pip install, models bundled → zero runtime downloads → safest for a live demo on a laptop.
* Threshold stricter than library default (0.6 → 0.5) because "unknown" people are expected.
* One-to-one assignment so a person is not named twice in one photo.

## My own tests (fill in)
| Test | Result |
|---|---|
| Threshold 0.45 / 0.50 / 0.55 / 0.60 on my practice set: wrong names / missed people | _fill in_ |
| HOG vs CNN detector: faces found of 10 | _fill in_ |
