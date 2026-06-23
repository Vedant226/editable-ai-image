"""
comfy_capabilities.py — detect which AI assets ComfyUI currently has (Phase 3).

The shared engine upgrades quality opportunistically: it uses ControlNet, an
SDXL inpaint model, soft (differential) inpainting and IP-Adapter FaceID *when
they are installed*, and falls back to the proven base-SDXL pipeline when they
are not. This module is the single source of truth for "what is available".

Detection is done against ComfyUI's live `/object_info` (the authoritative list
of installed nodes and model files) — no filesystem-path guessing, so it works
regardless of where ComfyUI stores its models. Results are cached briefly so a
burst of edits doesn't re-query the catalogue.

Nothing here raises: if ComfyUI is offline, every capability reports False/empty
and callers take the fallback path.
"""

import time

from comfy_client import ComfyUIError

_CACHE = {"t": 0.0, "data": None}
_TTL = 30.0  # seconds

# Substrings that identify a ControlNet model's kind from its filename.
_CN_KEYWORDS = {
    "canny":    ["canny"],
    "tile":     ["tile"],
    "depth":    ["depth"],
    "softedge": ["softedge", "soft_edge", "hed", "pidi", "scribble", "lineart", "line_art", "mlsd"],
    "inpaint":  ["inpaint"],
    "union":    ["union", "promax"],
}


def _classify_controlnets(names):
    """Map available ControlNet model files -> {kind: filename}."""
    out = {}
    for n in names:
        low = str(n).lower()
        for kind, kws in _CN_KEYWORDS.items():
            if any(k in low for k in kws):
                out.setdefault(kind, n)  # first match of each kind wins
    # A union/promax ControlNet can stand in for the per-kind models.
    if "union" in out:
        for k in ("canny", "depth", "softedge", "tile"):
            out.setdefault(k, out["union"])
    return out


def _enum(oi, node, field):
    try:
        return list(oi[node]["input"]["required"][field][0])
    except Exception:  # noqa: BLE001
        return []


def detect(client, force=False):
    """Return the capability dict (cached). Never raises.

    {
      online, controlnet_apply, differential_diffusion, inpaint_conditioning,
      ipadapter_faceid,
      controlnet: {canny: name, tile: name, ...},   # installed ControlNet models by kind
      controlnet_models: [...], checkpoints: [...], inpaint_checkpoints: [...],
    }
    """
    now = time.monotonic()
    if not force and _CACHE["data"] is not None and (now - _CACHE["t"]) < _TTL:
        return _CACHE["data"]

    caps = {
        "online": False,
        "controlnet_apply": False,
        "differential_diffusion": False,
        "inpaint_conditioning": False,
        "ipadapter_faceid": False,
        "controlnet": {},
        "controlnet_models": [],
        "checkpoints": [],
        "inpaint_checkpoints": [],
    }
    try:
        oi = client.get_object_info()
    except ComfyUIError:
        _CACHE.update(t=now, data=caps)
        return caps

    present = set(oi.keys())
    caps["online"] = True
    caps["controlnet_apply"] = ("ControlNetApplyAdvanced" in present and "ControlNetLoader" in present)
    caps["differential_diffusion"] = "DifferentialDiffusion" in present
    caps["inpaint_conditioning"] = "InpaintModelConditioning" in present
    caps["ipadapter_faceid"] = any(k.startswith("IPAdapter") for k in present) and (
        "IPAdapterFaceID" in present or "IPAdapterUnifiedLoaderFaceID" in present
    )

    cn_models = _enum(oi, "ControlNetLoader", "control_net_name")
    caps["controlnet_models"] = cn_models
    caps["controlnet"] = _classify_controlnets(cn_models) if caps["controlnet_apply"] else {}

    ckpts = _enum(oi, "CheckpointLoaderSimple", "ckpt_name")
    caps["checkpoints"] = ckpts
    caps["inpaint_checkpoints"] = [c for c in ckpts if "inpaint" in str(c).lower()]

    _CACHE.update(t=now, data=caps)
    return caps


def controlnet_model_for(kind, client):
    """The installed ControlNet model filename for `kind` (canny/tile/…), or None."""
    if not kind:
        return None
    return detect(client).get("controlnet", {}).get(kind)


def summary(client):
    """Human-facing capability summary for the /comfyui/capabilities route."""
    c = detect(client, force=True)
    cn = c["controlnet"]
    return {
        **c,
        "active_upgrades": {
            "soft_inpaint (DifferentialDiffusion)": c["differential_diffusion"],
            "controlnet_canny": bool(cn.get("canny")),
            "controlnet_tile": bool(cn.get("tile")),
            "controlnet_depth": bool(cn.get("depth")),
            "controlnet_softedge": bool(cn.get("softedge")),
            "sdxl_inpaint_model": bool(c["inpaint_checkpoints"]),
            "ipadapter_faceid": c["ipadapter_faceid"],
        },
    }
