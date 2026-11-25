"""
face_engine.py - all the face-recognition logic, with no web code in it.

Pipeline (same for the folder photos and the group photo):
    image file -> fix rotation -> DETECT faces -> ENCODE each face as 128 numbers
Then to answer "who is this?":
    compare the face's 128 numbers to every person in the folder (distance)
    -> closest person wins, but only if the distance is below a threshold,
       otherwise the face is "unknown".

Everything runs locally. The dlib models ship inside the pip package
`face_recognition_models`, so there is nothing to download at runtime.
"""
import base64
import hashlib
import io
import os
import re
from dataclasses import dataclass

import face_recognition
import numpy as np
from PIL import Image, ImageOps

# ---- settings you can tune -------------------------------------------------
DETECTOR = os.environ.get("FACE_DETECTOR", "hog")   # "hog" (fast) or "cnn" (more accurate, slow)
MAX_SIDE_GROUP = 1800        # shrink huge group photos to this (longest side) for speed
MAX_SIDE_ENROLL = 1000       # folder photos are usually one big face, so smaller is fine
DEFAULT_THRESHOLD = 0.50     # max face distance to accept a match (library default is 0.6)
JITTERS = 2                  # re-encode each face this many times with tiny shifts, average


@dataclass
class Person:
    name: str
    encoding: np.ndarray     # shape (128,)
    thumb: str               # data-URL of the face crop, shown in the UI
    file_hash: str


def name_from_filename(filename: str) -> str:
    """'Priya_Sharma.jpg' / 'priya sharma (2).JPG' -> 'Priya Sharma'."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\(\d+\)$", "", stem)
    return re.sub(r"\s+", " ", stem).strip().title() if stem.islower() else re.sub(r"\s+", " ", stem).strip()


def load_image(data: bytes, max_side: int):
    """Bytes -> (RGB numpy array, scale). Applies the phone's EXIF rotation first,
    otherwise sideways photos would have sideways faces that no detector finds."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    scale = 1.0
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return np.array(img), scale


def detect(rgb: np.ndarray):
    """Return face boxes as (top, right, bottom, left). If nothing is found,
    retry with upsampling (looks for smaller faces)."""
    for upsample in (1, 2):
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=upsample, model=DETECTOR)
        if boxes:
            return boxes
        if DETECTOR == "cnn":
            break
    return []


def encode(rgb, boxes):
    return face_recognition.face_encodings(rgb, boxes, num_jitters=JITTERS)


def crop_thumb(rgb: np.ndarray, box, size=112, pad=0.35) -> str:
    """Square-ish crop around a face as a small JPEG data-URL."""
    t, r, b, l = box
    h, w = b - t, r - l
    t, b = max(0, int(t - pad * h)), min(rgb.shape[0], int(b + pad * h))
    l, r = max(0, int(l - pad * w)), min(rgb.shape[1], int(r + pad * w))
    im = Image.fromarray(rgb[t:b, l:r])
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Gallery:
    """The 50 known people. Keeps an on-disk cache so re-opening the same folder is instant."""

    def __init__(self):
        self.people: list[Person] = []
        self.problems: list[dict] = []          # files we could not use, with a reason
        self._cache: dict[str, Person] = {}     # file_hash -> Person (name is re-derived)

    def reset(self):
        self.people, self.problems = [], []

    def add_file(self, filename: str, data: bytes):
        name = name_from_filename(filename)
        h = hashlib.sha1(data).hexdigest()
        if h in self._cache:
            p = self._cache[h]
            self.people.append(Person(name, p.encoding, p.thumb, h))
            return
        try:
            rgb, _ = load_image(data, MAX_SIDE_ENROLL)
        except Exception:
            self.problems.append({"file": filename, "reason": "Not a readable image"})
            return
        boxes = detect(rgb)
        if not boxes:
            self.problems.append({"file": filename, "reason": "No face found"})
            return
        # If a portrait has several faces, assume the biggest one is the named person.
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
        note = "Several faces - used the largest" if len(boxes) > 1 else None
        enc = encode(rgb, [box])[0]
        person = Person(name, enc, crop_thumb(rgb, box), h)
        self._cache[h] = person
        self.people.append(person)
        if note:
            self.problems.append({"file": filename, "reason": note, "warning": True})

    def public(self):
        return [{"name": p.name, "thumb": p.thumb} for p in self.people]


def analyse_group(data: bytes):
    """Detect + encode every face in the group photo. Returns what we need to match later."""
    rgb, scale = load_image(data, MAX_SIDE_GROUP)
    boxes = detect(rgb)
    encs = encode(rgb, boxes) if boxes else []
    faces = []
    for box, enc in zip(boxes, encs):
        t, r, b, l = box
        faces.append({
            "box": {"x": l / rgb.shape[1], "y": t / rgb.shape[0],      # fractions 0..1, so the
                    "w": (r - l) / rgb.shape[1], "h": (b - t) / rgb.shape[0]},  # UI can draw at any size
            "thumb": crop_thumb(rgb, box),
            "enc": enc,
        })
    # reading order: top-to-bottom rows, then left-to-right, so the list matches the photo
    faces.sort(key=lambda f: (round(f["box"]["y"] * 6), f["box"]["x"]))
    return rgb, faces


def match(faces, people, threshold=DEFAULT_THRESHOLD):
    """Decide who is who.

    1. Build a distance table: faces x people (small number = more similar).
    2. Take the smallest distances first (greedy). A pair is accepted only if the
       distance <= threshold AND neither the face nor the person is already used.
       This stops two faces both being called "Priya" - one person can only appear once.
    3. Faces left over are "unknown".
    """
    out = [{"name": None, "distance": None, "match_thumb": None, "best_guess": None} for _ in faces]
    if not faces or not people:
        return out
    known = np.array([p.encoding for p in people])                      # (P, 128)
    probe = np.array([f["enc"] for f in faces])                         # (F, 128)
    dist = np.linalg.norm(probe[:, None, :] - known[None, :, :], axis=2)  # (F, P)

    for fi in range(len(faces)):                       # closest candidate even if rejected
        pj = int(dist[fi].argmin())
        out[fi]["best_guess"] = {"name": people[pj].name, "distance": round(float(dist[fi, pj]), 3)}

    used_f, used_p = set(), set()
    for fi, pj in sorted(np.ndindex(*dist.shape), key=lambda ij: dist[ij]):
        if dist[fi, pj] > threshold:
            break
        if fi in used_f or pj in used_p:
            continue
        used_f.add(fi); used_p.add(pj)
        out[fi].update(name=people[pj].name, distance=round(float(dist[fi, pj]), 3), match_thumb=people[pj].thumb)
    return out
