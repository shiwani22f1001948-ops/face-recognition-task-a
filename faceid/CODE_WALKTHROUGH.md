# Code walkthrough (read this before any interview)

## The one idea
A neural network turns a face into **128 numbers** (an "embedding"). Photos of the *same person* give
numbers that are *close together*; different people are *far apart*. So "who is this?" becomes
"which of the 50 stored sets of numbers is closest, and is it close enough?"

## Step by step, in `face_engine.py`

1. **`load_image`** – opens the bytes with Pillow and calls `exif_transpose`. Phones store photos sideways
   plus a "rotate me" tag; without this, faces would be sideways and the detector would find nothing.
   Big images are shrunk (longest side 1800 px for group, 1000 px for portraits) because detection time
   grows with pixel count.
2. **`detect`** – dlib's HOG + linear-SVM face detector returns boxes `(top, right, bottom, left)`.
   `number_of_times_to_upsample=1` enlarges the image once so it can find faces ≥ ~40 px; if it finds
   nothing we retry with 2. Optional `cnn` detector is more robust to angles but slower.
3. **`encode`** – for each box, dlib finds 5 landmarks, aligns the face, and a ResNet outputs the 128 numbers.
   `num_jitters=2` re-runs it on slightly shifted copies and averages: a little slower, a little more stable.
4. **`Gallery.add_file`** – the filename becomes the name (`name_from_filename`: underscores -> spaces).
   If a portrait has several faces we keep the **largest** (assume it is the named person). Files with no
   face are reported to the user instead of silently ignored. Results are cached by SHA-1 of the file bytes.
5. **`analyse_group`** – same detect + encode on the group photo. Boxes are stored as **fractions (0–1)** of
   image width/height so the browser can draw them at any display size. Faces are sorted into reading order.
6. **`match`** – the decision:
   * `dist` = Euclidean distance between every group face and every known person (a 10 × 50 table, via NumPy broadcasting).
   * Go through all pairs from smallest distance upward. Accept a pair only if `distance <= threshold`
     **and** neither the face nor the person is already taken. (Greedy one-to-one assignment: two faces cannot both be "Priya".)
   * Everything left is `None` -> shown as **Unknown**. We still keep `best_guess` so the UI can say "closest: X, too different".
   * Threshold **0.50** (library default 0.6). Lower = stricter = fewer wrong names but more "Unknown". Because the
     demo includes people *not* in the folder, a stricter-than-default value is the safer choice.

## `app.py`
* `/api/gallery/reset` and `/api/gallery/add` – the browser sends the folder's files 4 at a time (so a progress bar can move).
* `/api/identify` – upload group photo -> detect/encode -> match -> JSON. The face encodings are kept in memory.
* `/api/rematch` – slider moved: re-run only `match()` with the new threshold. Instant, no image processing.
* A `threading.Lock` protects the shared state (single-user local app, but browsers can fire parallel requests).

## `static/index.html`
* Folder picker = `<input type="file" webkitdirectory>`; it reads the folder locally and posts the files to localhost.
* Boxes are absolutely-positioned `div`s using % coordinates over the `<img>`, so they scale with the photo.
* "Save photo" redraws the image + boxes on a `<canvas>` at full resolution.
* Colour is never the only signal: unknown = **dashed** orange box + "Unknown" label; known = solid teal + name.
* Names are HTML-escaped (`esc`) since they come from file names.

## Questions you should be ready for
* *Why not train a classifier on the 50 people?* One photo each is far too little; and the folder changes at demo time.
  Embedding + nearest neighbour needs **no training** and works for any new person instantly.
* *Why 0.50?* Reasoned trade-off; show the slider and how it changes results on your test photos.
* *Where does it fail?* Side-on faces, sunglasses/masks, tiny faces, very different age/lighting between folder photo and group photo,
  look-alikes. Say so honestly; mention CNN detector, more reference photos, or ArcFace models as upgrades.
* *Why dlib instead of ArcFace/InsightFace?* See RESEARCH.md (install risk on a judge's/own laptop vs a few % accuracy).
