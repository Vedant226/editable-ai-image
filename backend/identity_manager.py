"""
Identity Manager — capability detection + face analysis for the Identity
Preservation Engine (Phase 9).

This is the single place that knows WHICH identity method is available and how to
detect/embed faces. It is additive and isolated: nothing here imports or changes
any existing feature module.

Method priority (best available wins):
  1. "instantid"            — ComfyUI InstantID nodes + models present
  2. "ipadapter_faceid"     — ComfyUI IP-Adapter FaceID nodes + models present
  3. "insightface_facelock" — insightface (real detection + ArcFace embeddings)
  4. "opencv_facelock"      — OpenCV Haar cascade (last resort)
  5. "none"                 — no face capability at all

In this environment 1 and 2 are not installed (no custom nodes/models), so the
active method is "insightface_facelock". The IP-Adapter FaceID workflow template
exists and is selected automatically once those nodes/models are added — the
engine never hardcodes the graph.

insightface is initialised lazily (on first use), so importing this module — and
starting the bridge — stays fast.
"""

import logging
import os
import threading

import cv2
import numpy as np

logging.getLogger("insightface").setLevel(logging.ERROR)

_INSIGHTFACE_MODEL = os.environ.get("IDENTITY_INSIGHTFACE_MODEL", "buffalo_l")
_HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

_lock = threading.Lock()
_state = {"app": None, "tried": False, "haar": None, "comfy_nodes": None}


class Face:
    """Minimal face record used by the engine."""

    def __init__(self, bbox, kps=None, embedding=None):
        self.bbox = np.asarray(bbox, dtype=np.float32)        # x1,y1,x2,y2
        self.kps = None if kps is None else np.asarray(kps, dtype=np.float32)  # 5x2
        self.embedding = None if embedding is None else np.asarray(embedding, dtype=np.float32)


# -- insightface (lazy) ------------------------------------------------------

def _get_app():
    """Lazily build the insightface FaceAnalysis app. Returns it or None."""
    if _state["app"] is not None or _state["tried"]:
        return _state["app"]
    with _lock:
        if _state["tried"]:
            return _state["app"]
        _state["tried"] = True
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name=_INSIGHTFACE_MODEL, providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _state["app"] = app
        except Exception as exc:  # noqa: BLE001
            logging.warning("insightface unavailable (%s); falling back to OpenCV", exc)
            _state["app"] = None
    return _state["app"]


def _get_haar():
    if _state["haar"] is None and os.path.exists(_HAAR_PATH):
        _state["haar"] = cv2.CascadeClassifier(_HAAR_PATH)
    return _state["haar"]


# -- detection / embedding ---------------------------------------------------

def analyze(rgb):
    """All faces in an RGB image, richest available representation."""
    app = _get_app()
    if app is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = []
        for f in app.get(bgr):
            out.append(Face(f.bbox, getattr(f, "kps", None), getattr(f, "normed_embedding", None)))
        return out
    haar = _get_haar()
    if haar is not None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rects = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        return [Face((x, y, x + w, y + h)) for (x, y, w, h) in rects]
    return []


def main_face(rgb):
    """The largest face in an image, or None."""
    faces = analyze(rgb)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def similarity(emb_a, emb_b):
    """Cosine similarity of two ArcFace embeddings (None if unavailable)."""
    if emb_a is None or emb_b is None:
        return None
    a, b = np.asarray(emb_a, np.float32), np.asarray(emb_b, np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.dot(a, b) / (na * nb))


# -- capability detection ----------------------------------------------------

_IDENTITY_NODE_HINTS = {
    "instantid": ["InstantID", "ApplyInstantID", "InstantIDFaceAnalysis"],
    "ipadapter_faceid": ["IPAdapterFaceID", "IPAdapterUnifiedLoaderFaceID", "IPAdapterInsightFaceLoader"],
}


def _comfy_identity_nodes(client):
    """Which ComfyUI identity node families are installed (best-effort, cached)."""
    if _state["comfy_nodes"] is not None:
        return _state["comfy_nodes"]
    found = {"instantid": False, "ipadapter_faceid": False}
    if client is not None:
        try:
            oi = client.get_object_info()
            keys = set(oi.keys())
            for fam, hints in _IDENTITY_NODE_HINTS.items():
                found[fam] = any(h in keys for h in hints)
            _state["comfy_nodes"] = found  # only cache on success
        except Exception:  # noqa: BLE001 - ComfyUI offline; don't cache
            pass
    return found


def active_method(client=None):
    nodes = _comfy_identity_nodes(client)
    if nodes.get("instantid"):
        return "instantid"
    if nodes.get("ipadapter_faceid"):
        return "ipadapter_faceid"
    if _get_app() is not None:
        return "insightface_facelock"
    if _get_haar() is not None:
        return "opencv_facelock"
    return "none"


def capabilities(client=None):
    nodes = _comfy_identity_nodes(client)
    return {
        "method": active_method(client),
        "insightface": _get_app() is not None,
        "insightface_model": _INSIGHTFACE_MODEL if _get_app() is not None else None,
        "opencv_haar": _get_haar() is not None,
        "comfy_instantid": nodes.get("instantid", False),
        "comfy_ipadapter_faceid": nodes.get("ipadapter_faceid", False),
        "embeddings": _get_app() is not None,
    }
