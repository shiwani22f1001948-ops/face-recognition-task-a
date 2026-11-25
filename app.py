"""
app.py - tiny local web server (Flask). The browser UI talks to it on http://127.0.0.1:5000
Nothing leaves your laptop: the server only listens on 127.0.0.1.
"""
import os
import threading

from flask import Flask, jsonify, request, send_from_directory

import face_engine as fe

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024   # a folder of 50 photos can be large

gallery = fe.Gallery()
state = {"faces": []}        # faces of the most recent group photo (kept so the slider can re-match)
lock = threading.Lock()      # dlib is not happy being called from two threads at once

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/gallery/reset")
def gallery_reset():
    with lock:
        gallery.reset()
        state["faces"] = []
    return jsonify(ok=True)


@app.post("/api/gallery/add")
def gallery_add():
    """Receives a small batch of files from the folder picker (the UI sends 4 at a time
    so it can show a progress bar)."""
    with lock:
        for f in request.files.getlist("files"):
            if f.filename.lower().endswith(IMG_EXT):
                gallery.add_file(f.filename, f.read())
        return jsonify(count=len(gallery.people), people=gallery.public(), problems=gallery.problems)


@app.post("/api/identify")
def identify():
    """Upload a group photo -> faces + names."""
    f = request.files.get("photo")
    if f is None:
        return jsonify(error="No photo received"), 400
    if not gallery.people:
        return jsonify(error="Open the folder of people first"), 400
    with lock:
        try:
            _, faces = fe.analyse_group(f.read())
        except Exception:
            return jsonify(error="Could not read that image"), 400
        state["faces"] = faces
        return jsonify(results=_results(float(request.form.get("threshold", fe.DEFAULT_THRESHOLD))))


@app.post("/api/rematch")
def rematch():
    """Slider moved: re-decide names using the face data we already computed (instant)."""
    with lock:
        thr = float(request.get_json(force=True).get("threshold", fe.DEFAULT_THRESHOLD))
        return jsonify(results=_results(thr))


def _results(threshold):
    matches = fe.match(state["faces"], gallery.people, threshold)
    return [{"id": i, "box": f["box"], "face_thumb": f["thumb"], **m}
            for i, (f, m) in enumerate(zip(state["faces"], matches))]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Open http://127.0.0.1:{port} in your browser\n")
    app.run(host="127.0.0.1", port=port, threaded=True)
