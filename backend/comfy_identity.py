"""
Identity Preservation Engine — Phase 9.

A reusable, OPTIONAL identity layer for Change Hair, Change Clothes and Person
Replace. It takes a reference face and a target image and returns the target with
the reference's facial identity imposed on the face region — preserving facial
identity, eye/nose/mouth/jaw shape, skin tone and proportions, while leaving
hair, clothing, accessories, background and pose free to change.

It is fully additive and isolated: it does not import or modify any existing
feature module. It reuses only the shared transport (`ComfyUIClient`), the shared
data-url helpers from comfy_replace, and the capability detection in
identity_manager.

Active method is chosen by identity_manager:
  • IP-Adapter FaceID / InstantID via ComfyUI when those nodes/models exist
    (the graph lives in workflows/identity_preserve.json — never hardcoded), or
  • insightface face-lock: landmark-aligned face transfer + lighting match
    (the active method here; insightface gives real ArcFace embeddings so a true
    cosine face-similarity is reported), or
  • OpenCV Haar fallback.

If no face can be found the target is returned unchanged — so enabling identity
never breaks a result.
"""

import io
import json
import os

import cv2
import numpy as np
from PIL import Image

from comfy_client import ComfyUIClient, ComfyUIError
from comfy_replace import _decode_data_url, _data_url, CKPT_NAME
import identity_manager

WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "workflows", "identity_preserve.json")
RENDER_TIMEOUT = float(os.environ.get("COMFY_IDENTITY_TIMEOUT", "180"))
DEFAULT_STRENGTH = 0.85


class IdentityError(Exception):
    """An identity request that cannot be fulfilled (bad input)."""


# -- face-lock primitives ----------------------------------------------------

def _bbox_transform(rb, tb):
    """Scale+translate mapping reference face bbox onto target face bbox."""
    rcx, rcy = (rb[0] + rb[2]) / 2.0, (rb[1] + rb[3]) / 2.0
    tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
    rs = max(rb[2] - rb[0], rb[3] - rb[1])
    ts = max(tb[2] - tb[0], tb[3] - tb[1])
    s = ts / max(1e-3, rs)
    return np.array([[s, 0, tcx - s * rcx], [0, s, tcy - s * rcy]], np.float32)


def _face_mask(face, shape, expand=1.18, feather=0.18):
    """Feathered elliptical mask over the face (eyes/nose/mouth/jaw; hair excluded)."""
    h, w = shape
    x1, y1, x2, y2 = face.bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ax, ay = max(2.0, (x2 - x1) / 2.0 * expand), max(2.0, (y2 - y1) / 2.0 * expand)
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    k = int(max(3, (ax + ay) * feather)) | 1
    return cv2.GaussianBlur(m, (k, k), 0)


def _match_lighting(src_rgb, ref_rgb, mask):
    """Shift src luminance to match ref within the mask (lighting consistency;
    chroma — and thus skin tone — is preserved)."""
    m = mask > 0.2
    if int(m.sum()) < 16:
        return src_rgb
    s_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dL = float(r_lab[..., 0][m].mean() - s_lab[..., 0][m].mean())
    s_lab[..., 0] = np.clip(s_lab[..., 0] + dL, 0, 255)
    return cv2.cvtColor(s_lab.astype("uint8"), cv2.COLOR_LAB2RGB)


def _facelock(ref_rgb, tgt_rgb, strength):
    """Transfer the reference identity onto the target's face region."""
    method = "insightface_facelock" if identity_manager._get_app() is not None else "opencv_facelock"
    info = {"method": method, "faceFound": False, "similarity": None}
    rf = identity_manager.main_face(ref_rgb)
    tf = identity_manager.main_face(tgt_rgb)
    if rf is None or tf is None:
        return tgt_rgb, info  # graceful no-op: identity never breaks the result
    info["faceFound"] = True

    th, tw = tgt_rgb.shape[:2]
    if rf.kps is not None and tf.kps is not None:
        M, _ = cv2.estimateAffinePartial2D(rf.kps, tf.kps, method=cv2.LMEDS)
    else:
        M = _bbox_transform(rf.bbox, tf.bbox)
    if M is None:
        return tgt_rgb, info

    warped = cv2.warpAffine(ref_rgb, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fmask = _face_mask(tf, (th, tw))
    warped = _match_lighting(warped, tgt_rgb, fmask)
    a = (fmask * float(strength))[:, :, None]
    out = (tgt_rgb * (1 - a) + warped * a).astype("uint8")

    of = identity_manager.main_face(out)
    if of is not None and rf.embedding is not None and of.embedding is not None:
        info["similarity"] = identity_manager.similarity(rf.embedding, of.embedding)
    return out, info


# -- IP-Adapter FaceID / InstantID via ComfyUI (used only if installed) ------

def _run_faceid_workflow(ref_rgb, tgt_rgb, strength, client):
    """Run the diffusion identity workflow. Raises if the nodes aren't present."""
    def _png(rgb):
        b = io.BytesIO(); Image.fromarray(rgb).save(b, format="PNG"); return b.getvalue()

    up_ref = client.upload_image(_png(ref_rgb), "identity_reference.png", overwrite=True)
    up_tgt = client.upload_image(_png(tgt_rgb), "identity_target.png", overwrite=True)

    with open(WORKFLOW_PATH) as fh:
        graph = json.load(fh)
    graph.pop("_comment", None)
    graph["checkpoint"]["inputs"]["ckpt_name"] = CKPT_NAME
    graph["reference_image"]["inputs"]["image"] = up_ref["name"]
    graph["target_image"]["inputs"]["image"] = up_tgt["name"]
    if "ipadapter_faceid" in graph:
        graph["ipadapter_faceid"]["inputs"]["weight"] = float(strength)

    queued = client.queue_prompt(graph)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise ComfyUIError(f"ComfyUI did not queue the identity prompt: {queued}")
    entry = client.await_result(prompt_id, timeout=RENDER_TIMEOUT)
    images = (entry.get("outputs", {}).get("save", {}) or {}).get("images", [])
    if not images:
        raise ComfyUIError("identity workflow produced no image")
    img0 = images[0]
    out = client.get_image(img0["filename"], img0.get("subfolder", ""), img0.get("type", "output"))
    return np.array(Image.open(io.BytesIO(out)).convert("RGB"))


# -- engine entry points -----------------------------------------------------

def preserve_identity(ref_rgb, tgt_rgb, strength=DEFAULT_STRENGTH, client=None):
    """Impose the reference identity on the target's face. Returns (rgb, info)."""
    method = identity_manager.active_method(client)
    if method in ("instantid", "ipadapter_faceid") and client is not None:
        try:
            res = _run_faceid_workflow(ref_rgb, tgt_rgb, strength, client)
            info = {"method": method, "faceFound": True, "similarity": None}
            rf = identity_manager.main_face(ref_rgb)
            of = identity_manager.main_face(res)
            if rf is not None and of is not None:
                info["similarity"] = identity_manager.similarity(rf.embedding, of.embedding)
            return res, info
        except Exception:  # noqa: BLE001 - nodes/models missing -> fall back
            pass
    return _facelock(ref_rgb, tgt_rgb, strength)


def identity_preserve(reference_image, target_image, mask=None, strength=DEFAULT_STRENGTH, client=None):
    """Endpoint-facing API. Images are data URLs; returns a result data URL.

    If the target (or `mask`) carries an alpha footprint, the result is an RGBA
    patch with that same footprint — drop-in for the editor's overlay system.
    """
    if not reference_image or not target_image:
        raise IdentityError("both referenceImage and targetImage are required")
    try:
        strength = max(0.0, min(1.0, float(strength)))
    except (TypeError, ValueError):
        strength = DEFAULT_STRENGTH

    ref = _decode_data_url(reference_image)   # RGBA
    tgt = _decode_data_url(target_image)      # RGBA
    ref_rgb = ref[:, :, :3]
    tgt_rgb = tgt[:, :, :3]

    footprint = None
    if mask:
        footprint = _decode_data_url(mask)[:, :, 0]
    elif (tgt[:, :, 3] < 250).any():
        footprint = tgt[:, :, 3]

    # composite the target over mid-grey for detection robustness at silhouette edges
    detect_rgb = tgt_rgb
    if footprint is not None:
        af = (footprint.astype(np.float32) / 255.0)[:, :, None]
        detect_rgb = (tgt_rgb * af + 128 * (1 - af)).astype("uint8")

    # Portrait crops can be small; upscale to ~512 long-side for reliable face
    # detection / embedding, then bring the result back to the original size.
    th0, tw0 = detect_rgb.shape[:2]
    ps = max(1.0, 512.0 / max(th0, tw0))
    if ps > 1.01:
        tgt_p = cv2.resize(detect_rgb, (round(tw0 * ps), round(th0 * ps)), interpolation=cv2.INTER_CUBIC)
        rs = max(1.0, 512.0 / max(ref_rgb.shape[0], ref_rgb.shape[1]))
        ref_p = cv2.resize(ref_rgb, (round(ref_rgb.shape[1] * rs), round(ref_rgb.shape[0] * rs)),
                           interpolation=cv2.INTER_CUBIC) if rs > 1.01 else ref_rgb
    else:
        tgt_p, ref_p = detect_rgb, ref_rgb

    result_p, info = preserve_identity(ref_p, tgt_p, strength=strength, client=client)
    result_rgb = cv2.resize(result_p, (tw0, th0), interpolation=cv2.INTER_AREA) if ps > 1.01 else result_p

    if footprint is not None:
        out = np.dstack([result_rgb, footprint]).astype("uint8")  # keep the footprint shape
    else:
        out = result_rgb
    return {
        "png": _data_url(out),
        "strength": strength,
        "capabilities": identity_manager.capabilities(client),
        **info,
    }
