"""Smoke test of the whole pipeline through the real HTTP endpoints.
Run:  python tests/test_pipeline.py   (needs scikit-image only for the sample face)"""
import io, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import skimage.data as sd
import app as server

def jpg(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "JPEG"); b.seek(0); return b

astro = sd.astronaut()
portrait = astro[40:260, 150:370]                         # crop around the face = "folder photo"
# group photo: astronaut on the left, a mirrored copy on the right (different pixels, same person),
# so we can check detection of 2 faces and that one person is not assigned twice.
group = np.concatenate([astro, astro[:, ::-1]], axis=1)

c = server.app.test_client()
assert c.post("/api/gallery/reset").status_code == 200
r = c.post("/api/gallery/add", data={"files": [(jpg(portrait), "eileen_collins.jpg"),
                                               (jpg(np.zeros((300, 300, 3), np.uint8)), "Blank Person.jpg")]},
           content_type="multipart/form-data").get_json()
print("gallery:", r["count"], [p["name"] for p in r["people"]], r["problems"])
assert r["count"] == 1 and r["people"][0]["name"] == "Eileen Collins" and r["problems"][0]["reason"] == "No face found"

r = c.post("/api/identify", data={"photo": (jpg(group), "g.jpg"), "threshold": "0.5"},
           content_type="multipart/form-data").get_json()["results"]
for x in r: print(x["box"], x["name"], x["distance"], x["best_guess"])
names = [x["name"] for x in r]
assert names.count("Eileen Collins") == 1, "one person must not be matched twice"
r2 = c.post("/api/rematch", json={"threshold": 0.05}).get_json()["results"]
assert all(x["name"] is None for x in r2), "very strict threshold must give all unknown"
print("OK")
