# Who's in the photo? (Take-home Task A)

Open a folder of 50 people (one photo each, file name = person's name), upload a group photo,
and see every face labelled with a name, or **Unknown** if that person is not in the folder.

**Fully offline.** No internet, no API, no cloud. The only network use is the one-time `pip install`; after that nothing leaves the laptop.
The server listens on `127.0.0.1` only.

## Install (Windows / macOS / Linux, Python 3.9 – 3.12)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
pip install face_recognition --no-deps      # --no-deps: we use the pre-built dlib-bin instead of compiling dlib
```

The only internet use is this `pip install`. The face models are inside the `face_recognition_models`
package, so after installing you can turn Wi-Fi off.

## Run

```bash
python app.py
```
Open **http://127.0.0.1:5000**

1. **Choose folder…**, pick the folder with the 50 photos (named `Priya Sharma.jpg`, `Priya_Sharma.jpg`, ...).
   The first load takes roughly 1–2 s per photo; re-opening the same folder is instant (cached in memory).
2. **Drop the group photo.** Names appear on the photo and in the list on the right.
3. Move the **Strictness** slider if people are wrongly named (stricter) or wrongly "Unknown" (looser).
   It re-decides instantly without re-reading the photo.
4. Hover a name in the list to highlight the face. Save the named photo or a CSV with the buttons below.

## Try it with practice data
```bash
# download + extract LFW (http://vis-www.cs.umass.edu/lfw/) once, then
python tools/make_demo_data.py path/to/lfw       # makes demo_data/people/ and demo_data/group_*.jpg
python tests/test_pipeline.py                    # automated smoke test (needs: pip install scikit-image)
```

## Tips for the demo
* Folder photos: one clear, front-facing face each. If a file has several faces, the biggest is used.
* Faces smaller than ~40 px in the group photo can be missed. Use the original photo, not a screenshot.
* Optional slower, more accurate detector for tilted faces: `FACE_DETECTOR=cnn python app.py`
  (Windows PowerShell: `$env:FACE_DETECTOR="cnn"; python app.py`).

## Files
| File | Job |
|---|---|
| `face_engine.py` | detect -> encode -> match (all the recognition logic) |
| `app.py` | small Flask server with 4 endpoints |
| `static/index.html` | the whole UI (no external fonts/scripts, works offline) |
| `tools/`, `tests/` | practice-data builder, smoke test |
| `CODE_WALKTHROUGH.md` | step-by-step explanation of every part |
| `AI_COLLAB.md`, `RESEARCH.md`, `ARCHITECTURE.md` | required docs |
