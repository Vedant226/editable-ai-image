"""
comfy_engine.py — the shared AI Editing Engine (Phase 2 refactor).

ONE reusable inpaint pipeline behind every object-level AI feature (Replace,
Remove, Recolor, Change Clothes, Change Hair, Replace Person, …). Previously each
feature module re-implemented the identical orchestration; now they all call
`run_object_edit` and supply only what is genuinely different:

  * region_mask   — the EDITABLE region (footprint, or a derived sub-region such
                    as the hair/clothing area). Full-canvas uint8, 255 = editable.
  * init_builder  — optional: how the masked pixels are pre-filled before SDXL.
                    Replace composites the upload, Recolor does a LAB recolour,
                    Remove/Person-upload do a coarse OpenCV inpaint. Default: the
                    untouched crop (Hair / Clothes / Person-prompt).
  * params/prompt — sampler params (denoise/steps/cfg/…) + positive/negative.
  * postproc      — optional: e.g. Change Hair hard-protects the face alpha.
  * extra         — optional dict merged into the result (mode, targetColor, …).

Everything else — the context crop, mask dilate, SDXL working-size scaling, the
upload, graph injection, queue/await/fetch, scale-back and the feathered RGBA
patch cut — lives here, once. The SDXL graph stays declarative in
workflows/inpaint_engine.json; this module only injects per-request values
(checkpoint, image names, prompts, sampler) into a copy of that template. The
mask node is found by class_type, so the same injector also drives the legacy
per-feature templates unchanged.

Isolated from the LaMa inpaint service (app.py): read-only on the locked layer
artefacts (metadata.json + background.png); ComfyUIClient is pure transport.

  Phase-3 hook: run_object_edit accepts a `controlnet` argument (currently
  unused / None). When ControlNet models + nodes are installed, the injector can
  add structure conditioning here without changing any feature module.

Config (env vars):
  LAYERS_DIR         — layers dir (default ../editable-editor/public/layers)
  COMFY_DEFAULT_CKPT — default SDXL checkpoint (default sd_xl_base_1.0.safetensors)
"""

import base64
import io
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from comfy_client import ComfyUIClient, ComfyUIError
import comfy_capabilities
import comfy_style
import comfy_planner
import comfy_evaluator

LAYERS_DIR = os.path.abspath(
    os.environ.get(
        "LAYERS_DIR",
        os.path.join(os.path.dirname(__file__), "..", "editable-editor", "public", "layers"),
    )
)
CKPT_NAME = os.environ.get("COMFY_DEFAULT_CKPT", "sd_xl_base_1.0.safetensors")
SHARED_TEMPLATE = os.path.join(os.path.dirname(__file__), "workflows", "inpaint_engine.json")

# Working-size bounds shared by every object edit (SDXL works poorly below 768;
# 1024 long-side caps VRAM for an 8GB card — the context-crop keeps us there).
TARGET_MIN = 768
TARGET_MAX = 1024

# Best-of-N hard cap. Candidates render SEQUENTIALLY (one SDXL pass at a time),
# so N raises wall-clock, never peak VRAM — safe on an 8GB card.
N_MAX = 4

# ---- Phase 3: style preservation ----
# The source artwork is a painted illustration, so (per the approved decision)
# edits MATCH that style rather than pushing photoreal. Features pass a content
# prompt without a style word; apply_style() appends the right suffix.
STYLE_SUFFIXES = {
    "preserve": ("in the same painted illustration art style as the original artwork, "
                 "consistent brushwork, texture and colour palette, sharp detail"),
    "photoreal": "photorealistic, realistic photo, sharp focus",
    "none": "",
}


def apply_style(positive, style="preserve"):
    """Append the style-preservation suffix to a content prompt."""
    suffix = STYLE_SUFFIXES.get(style or "preserve", STYLE_SUFFIXES["preserve"])
    if not suffix:
        return positive
    return f"{positive.rstrip(' ,')}, {suffix}"


def _control_hint(kind, init_w):
    """Compute a ControlNet control image (3-channel RGB) from the working init.

    No preprocessor custom-node needed: canny is cv2.Canny; tile is the image
    itself (SDXL tile control). Both run on CPU in a few ms.
    """
    if kind == "tile":
        return init_w  # the tile ControlNet conditions on the image content directly
    # default: canny structure hint
    gray = cv2.cvtColor(init_w, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.dstack([edges, edges, edges])


# ---- Phase 12: quality validation ----
# Conservative, CPU-only checks that catch the common failure modes (a blurry or
# degenerate/flat render) without a model. Model-based identity/style/lighting
# scores plug in here when those models are installed. Thresholds are loose on
# purpose: a normal textured render passes, so the retry never fires for it (and
# the returned patch is byte-for-byte what it would have been pre-Phase-12).
_MIN_BLUR = 6.0   # Laplacian variance below this ≈ blurry/smeared
_MIN_STD = 4.0    # per-channel std below this ≈ flat/degenerate patch


def _validate_patch(patch_rgb):
    """Return a quality report {blur, colorStd, ok, checks} for a generated patch."""
    if patch_rgb.size == 0:
        return {"blur": 0.0, "colorStd": 0.0, "ok": False, "checks": {"blur": False, "notFlat": False}}
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    std = float(patch_rgb.std())
    checks = {"blur": blur >= _MIN_BLUR, "notFlat": std >= _MIN_STD}
    return {"blur": round(blur, 1), "colorStd": round(std, 1),
            "ok": all(checks.values()), "checks": checks}


class EditError(Exception):
    """An edit that cannot be fulfilled for geometric/decode reasons (bad object,
    empty footprint, off-canvas). Feature modules translate this into their own
    error type so the router's HTTP mapping is unchanged."""


# -- image / data-url helpers (single canonical home) ------------------------

def _round8(v):
    return max(8, int(round(v / 8.0)) * 8)


def _data_url(rgb_or_rgba):
    buf = io.BytesIO()
    Image.fromarray(rgb_or_rgba).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _decode_rgba(data_url):
    """Accept a data: URL or a bare base64 string -> RGBA uint8 array."""
    s = data_url or ""
    if "," in s and s.strip().lower().startswith("data:"):
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    return np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))


def _decode_rgb(data_url):
    """Accept a data: URL or a bare base64 string -> RGB uint8 array."""
    s = data_url or ""
    if "," in s and s.strip().lower().startswith("data:"):
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


# Back-compat alias: the original Replace helper returned RGBA under this name.
_decode_data_url = _decode_rgba


def _load_metadata():
    """{id: object} map (the form most feature modules want)."""
    with open(os.path.join(LAYERS_DIR, "metadata.json")) as fh:
        return {int(o["id"]): o for o in json.load(fh)}


def _load_metadata_list():
    """Raw metadata list (Background wants the union over every entry)."""
    with open(os.path.join(LAYERS_DIR, "metadata.json")) as fh:
        return json.load(fh)


def _background_rgb():
    img = Image.open(os.path.join(LAYERS_DIR, "background.png")).convert("RGB")
    return np.array(img)


def _footprint_mask(obj, H, W):
    """Full-canvas binary footprint (255 = object) from the layer PNG's alpha."""
    x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        raise EditError("object footprint is empty / off-canvas")
    png = Image.open(os.path.join(LAYERS_DIR, obj["file"]))
    mask = np.zeros((H, W), dtype="uint8")
    if "A" in png.getbands():
        alpha = np.array(png.split()[-1].resize((w, h)))
        local = (alpha > 12).astype("uint8") * 255
    else:  # opaque crop (e.g. text rect) -> full rectangle
        local = np.full((h, w), 255, dtype="uint8")
    mask[y0:y1, x0:x1] = local[(y0 - y):(y1 - y), (x0 - x):(x1 - x)]
    return mask, (x0, y0, x1, y1)


def _bbox_of(boolmask):
    ys, xs = np.where(boolmask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# -- ComfyUI graph: inject values + run --------------------------------------

def _load_graph(workflow_path):
    with open(workflow_path) as fh:
        graph = json.load(fh)
    graph.pop("_comment", None)
    return graph


def inject_inpaint_graph(graph, *, init_name, mask_name, positive, negative, params, ckpt,
                         soft_inpaint=False, controlnet=None):
    """Inject per-request values into an SDXL masked-inpaint graph (in place).

    Values only for the base graph — its STRUCTURE stays in the JSON template; the
    mask node is found by class_type (LoadImageMask) so this drives the shared and
    the legacy templates alike.

    Phase-3 quality nodes are ADDED here when requested (all ComfyUI core nodes):
      * soft_inpaint  -> a DifferentialDiffusion model patch, so a soft (feathered)
                         mask denoises with a gradient at the boundary = seamless
                         blend (no hard latent seam).
      * controlnet    -> {model, image, strength, start, end}: a ControlNetLoader +
                         ControlNetApplyAdvanced branch conditioning the sampler on
                         a structure/texture hint (perspective/folds/detail kept).
    Both are no-ops when their argument is falsy, so the base path is unchanged.
    """
    def put(node, key, value):
        graph[node]["inputs"][key] = value

    put("checkpoint", "ckpt_name", ckpt)
    put("init_image", "image", init_name)
    mask_node = next(
        (nid for nid, n in graph.items()
         if isinstance(n, dict) and n.get("class_type") == "LoadImageMask"),
        None,
    )
    if mask_node is None:
        raise ComfyUIError("workflow template has no LoadImageMask node")
    put(mask_node, "image", mask_name)
    put("positive_prompt", "text", positive)
    put("negative_prompt", "text", negative)
    put("ksampler", "seed", int(params["seed"]))
    put("ksampler", "steps", int(params["steps"]))
    put("ksampler", "cfg", float(params["cfg"]))
    put("ksampler", "denoise", float(params["denoise"]))
    put("ksampler", "sampler_name", params["sampler"])
    put("ksampler", "scheduler", params["scheduler"])

    # --- soft inpaint: gradient-aware denoise mask (seamless boundary) ---
    if soft_inpaint:
        graph["diff_diff"] = {
            "class_type": "DifferentialDiffusion",
            "inputs": {"model": ["checkpoint", 0]},
        }
        graph["ksampler"]["inputs"]["model"] = ["diff_diff", 0]

    # --- ControlNet: lock structure/texture to the original ---
    if controlnet:
        graph["controlnet_loader"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet["model"]},
        }
        graph["control_image"] = {
            "class_type": "LoadImage",
            "inputs": {"image": controlnet["image"]},
        }
        graph["controlnet_apply"] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["positive_prompt", 0],
                "negative": ["negative_prompt", 0],
                "control_net": ["controlnet_loader", 0],
                "image": ["control_image", 0],
                "strength": float(controlnet.get("strength", 0.6)),
                "start_percent": float(controlnet.get("start", 0.0)),
                "end_percent": float(controlnet.get("end", 0.8)),
            },
        }
        graph["ksampler"]["inputs"]["positive"] = ["controlnet_apply", 0]
        graph["ksampler"]["inputs"]["negative"] = ["controlnet_apply", 1]

    return graph


def submit_and_fetch(client, graph, timeout):
    """Queue a graph, await completion, fetch the SaveImage output -> RGB array.

    Returns (result_rgb_ndarray, prompt_id). Raises ComfyUIError if the queue is
    rejected or no image is produced, TimeoutError if the render stalls.
    """
    queued = client.queue_prompt(graph)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise ComfyUIError(f"ComfyUI did not queue the prompt: {queued}")
    entry = client.await_result(prompt_id, timeout=timeout)
    images = (entry.get("outputs", {}).get("save", {}) or {}).get("images", [])
    if not images:
        raise ComfyUIError(f"ComfyUI produced no image (status={entry.get('status')})")
    img0 = images[0]
    out_bytes = client.get_image(img0["filename"], img0.get("subfolder", ""), img0.get("type", "output"))
    return np.array(Image.open(io.BytesIO(out_bytes)).convert("RGB")), prompt_id


# -- the shared object-edit pipeline -----------------------------------------

def run_object_edit(object_id, bg, region_mask, *, params, positive, negative,
                    slug, ckpt=None, context_margin_frac=0.35, dilate=9, feather=9,
                    target_min=TARGET_MIN, target_max=TARGET_MAX, timeout=180,
                    workflow_path=None, init_builder=None, postproc=None,
                    controlnet=None, soft_inpaint=True, style="auto",
                    validate=True, max_retries=1, plan=True, material_hint=None,
                    harmonize=False, harmonize_strength=None, n=1, evaluator=False,
                    prompt_for_eval=None, identity_face=None, eval_weights=None,
                    client=None, extra=None):
    """Run one masked SDXL edit and return a feathered RGBA patch descriptor.

    bg           — full-canvas RGB (the locked background.png).
    region_mask  — full-canvas uint8, 255 where SDXL may change pixels.
    init_builder — optional fn(ctx) -> init crop RGB. ctx carries crop, dilated
                   mask, crop origin and the region bbox so a feature can
                   composite/recolour/inpaint exactly as before. Default: crop.
    postproc     — optional fn(patch_rgba, ctx) -> patch_rgba (e.g. face protect).
    controlnet   — kind ("canny"/"tile"/…) or {type, strength, start, end}. Used
                   only if that ControlNet model is installed (else ignored).
    soft_inpaint — use DifferentialDiffusion + a soft mask for a seamless boundary
                   when the node is available (it is, in core ComfyUI).
    style        — "preserve" (match the painted artwork) / "photoreal" / "none".
    extra        — dict merged into the returned descriptor.

    Every Phase-3 upgrade degrades gracefully: when an asset is missing the call
    falls back to the proven base pipeline. Returns
    { objectId, x, y, w, h, png, engine, promptId, upgrades, **extra }.
    """
    ckpt = ckpt or CKPT_NAME
    H, W = bg.shape[:2]
    client = client or ComfyUIClient()
    caps = comfy_capabilities.detect(client)

    gbox = _bbox_of(region_mask > 0)
    if gbox is None:
        raise EditError("editable region is empty")
    gx0, gy0, gx1, gy1 = gbox
    gw, gh = gx1 - gx0, gy1 - gy0

    # context crop around the editable region — everything outside stays original
    mx = int(round(gw * context_margin_frac)) + dilate + feather
    my = int(round(gh * context_margin_frac)) + dilate + feather
    cx0, cy0 = max(0, gx0 - mx), max(0, gy0 - my)
    cx1, cy1 = min(W, gx1 + mx), min(H, gy1 + my)
    crop = bg[cy0:cy1, cx0:cx1].copy()
    ch, cw = crop.shape[:2]

    # resolve the style suffix: auto-detect from the surrounding crop (so the edit
    # matches the artwork's style and never introduces a different one), or apply a
    # fixed style ("preserve"/"photoreal"/"none"). Conservative: detection defaults
    # to a painterly "illustration", only flips to photo on strong photographic cues.
    if style == "auto":
        detected_style, _sf = comfy_style.detect_style(crop)
        suffix = comfy_style.STYLE_SUFFIXES.get(detected_style, "")
        positive = f"{positive.rstrip(' ,')}, {suffix}" if suffix else positive
    else:
        detected_style = style
        positive = apply_style(positive, style)

    rm_crop = region_mask[cy0:cy1, cx0:cx1]
    dil = cv2.dilate(rm_crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    if _bbox_of(dil > 0) is None:
        raise EditError("editable region is empty after cropping")

    # ---- Universal AI Edit Planner (Phase 14): material/lighting analysis ->
    # additive prompt optimisation. Gated by `plan` so the raw engine is unchanged. ----
    analysis = None
    if plan:
        analysis = comfy_planner.analyze(crop, dil, material_hint)
        positive, negative = comfy_planner.augment(positive, negative, analysis)

    ctx = SimpleNamespace(
        crop=crop, dil=dil, cx0=cx0, cy0=cy0, gx0=gx0, gy0=gy0,
        gw=gw, gh=gh, ch=ch, cw=cw, H=H, W=W, region_mask=region_mask,
    )
    init = init_builder(ctx) if init_builder is not None else crop

    # scale crop+mask to an SDXL-friendly working size (multiple of 8)
    long_side = max(ch, cw)
    target = min(target_max, max(target_min, long_side))
    scale = target / float(long_side)
    nw, nh = _round8(cw * scale), _round8(ch * scale)
    init_w = cv2.resize(init, (nw, nh), interpolation=cv2.INTER_AREA)

    # ---- Phase-3 upgrades, each gated on availability (else exact base path) ----
    soft_active = bool(soft_inpaint and caps.get("differential_diffusion"))
    if soft_active:
        # soft (feathered) mask -> DifferentialDiffusion blends the boundary in
        # latent space (interior stays 255 = fully regenerated). Seamless seam.
        fsoft = (max(3, feather) * 2) | 1
        soft = cv2.GaussianBlur(dil, (fsoft, fsoft), 0)
        mask_w = cv2.resize(soft, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        mask_w = cv2.resize(dil, (nw, nh), interpolation=cv2.INTER_NEAREST)

    cn_kind = controlnet.get("type") if isinstance(controlnet, dict) else controlnet
    cn_in = controlnet if isinstance(controlnet, dict) else {}
    cn_model = caps.get("controlnet", {}).get(cn_kind) if cn_kind else None

    init_png = io.BytesIO(); Image.fromarray(init_w).save(init_png, format="PNG")
    mask_rgb = np.dstack([mask_w, mask_w, mask_w])
    mask_png = io.BytesIO(); Image.fromarray(mask_rgb).save(mask_png, format="PNG")

    up_init = client.upload_image(init_png.getvalue(), f"{slug}_init_{object_id}.png", overwrite=True)
    up_mask = client.upload_image(mask_png.getvalue(), f"{slug}_mask_{object_id}.png", overwrite=True)

    cn_cfg = None
    if cn_model:
        hint = _control_hint(cn_kind, init_w)
        hint_png = io.BytesIO(); Image.fromarray(hint).save(hint_png, format="PNG")
        up_hint = client.upload_image(hint_png.getvalue(), f"{slug}_ctrl_{object_id}.png", overwrite=True)
        cn_cfg = {
            "model": cn_model, "image": up_hint["name"],
            "strength": float(cn_in.get("strength", 0.6)),
            "start": float(cn_in.get("start", 0.0)),
            "end": float(cn_in.get("end", 0.8)),
        }

    fk = feather | 1
    alpha = cv2.GaussianBlur(dil.astype(np.float32), (fk, fk), 0) / 255.0
    rel_x0, rel_y0 = gx0 - cx0, gy0 - cy0

    # One candidate render: inject -> submit -> scale back -> cut -> refine ->
    # feathered alpha -> postproc. Extracted so the retry loop AND best-of-N call
    # the identical code (pure extraction — behaviour is unchanged).
    def _render_one(gen_params):
        graph = inject_inpaint_graph(
            _load_graph(workflow_path or SHARED_TEMPLATE),
            init_name=up_init["name"], mask_name=up_mask["name"],
            positive=positive, negative=negative, params=gen_params, ckpt=ckpt,
            soft_inpaint=soft_active, controlnet=cn_cfg,
        )
        result_w, prompt_id = submit_and_fetch(client, graph, timeout)
        result = cv2.resize(result_w, (cw, ch), interpolation=cv2.INTER_CUBIC)
        patch_rgb = result[rel_y0:rel_y0 + gh, rel_x0:rel_x0 + gw]
        if plan:  # Planner refinement stage: hue-neutral high-frequency detail recovery
            patch_rgb = comfy_planner.refine(patch_rgb, analysis)
        patch_a = np.clip(alpha[rel_y0:rel_y0 + gh, rel_x0:rel_x0 + gw] * 255.0, 0, 255).astype("uint8")
        patch_rgba = np.dstack([patch_rgb, patch_a]).astype("uint8")
        if postproc is not None:
            patch_rgba = postproc(patch_rgba, ctx)
        # Auto-chained lighting/colour/edge harmonization (Phase 14). Off by
        # default -> this block is skipped and the patch is byte-identical. Runs
        # after postproc so the final footprint alpha drives the ambient match.
        harmonized = False
        if harmonize:
            try:
                import comfy_lighting
                opts = {} if harmonize_strength is None else {"strength": float(harmonize_strength)}
                patch_rgba, _rep = comfy_lighting.harmonize_patch(patch_rgba, crop, rel_x0, rel_y0, opts)
                patch_rgb = patch_rgba[:, :, :3]
                harmonized = True
            except Exception:  # noqa: BLE001 - harmonization never blocks an edit
                pass
        return patch_rgba, patch_rgb, prompt_id, harmonized

    def _augment_with_score(quality, patch_rgb, patch_rgba):
        """Merge the multi-dimensional self-eval score into a quality dict (the
        backward-compatible blur/colorStd/ok/checks fields are left untouched)."""
        rich = comfy_evaluator.evaluate(
            patch_rgb, crop_rgb=crop, patch_left=rel_x0, patch_top=rel_y0,
            footprint=patch_rgba[:, :, 3], prompt=prompt_for_eval,
            original_face_rgb=identity_face, weights=eval_weights, client=client,
        )
        quality["score"] = rich.get("score", 0.0)
        quality["dimensions"] = rich.get("dimensions", {})

    n = max(1, min(int(n or 1), N_MAX))
    want_score = bool(evaluator) or n > 1

    gen_params = dict(params)
    quality = None
    harmonized = False
    if n == 1:
        # Generate; validate; retry once with more steps + a fresh seed if the
        # patch comes back degenerate. A normal render passes first try, so this
        # adds no extra work (and no output change) for the common case.
        attempts = 0
        while True:
            patch_rgba, patch_rgb, prompt_id, harmonized = _render_one(gen_params)
            if not validate:
                break
            quality = _validate_patch(patch_rgb)
            quality["attempts"] = attempts + 1
            if want_score:
                _augment_with_score(quality, patch_rgb, patch_rgba)
            if quality["ok"] or attempts >= max_retries:
                break
            attempts += 1
            gen_params = {**gen_params, "steps": int(gen_params["steps"]) + 8,
                          "seed": int(gen_params["seed"]) + 1}
    else:
        # Best-of-N: render N candidates varying ONLY the seed (candidate-0 uses
        # the base seed, so it is bit-identical to the n=1 render), score each and
        # keep the best. Degenerate-retry is disabled (N already gives robustness).
        # Uploads stay outside this loop -> sequential renders, no extra VRAM.
        base_seed = int(gen_params["seed"])
        best = None
        for i in range(n):
            cand = {**gen_params, "seed": base_seed + i}
            c_rgba, c_rgb, c_pid, c_harm = _render_one(cand)
            q = _validate_patch(c_rgb)
            q["attempts"] = 1
            _augment_with_score(q, c_rgb, c_rgba)
            sval = q.get("score", 0.0)
            if best is None or sval > best["s"]:
                best = {"rgba": c_rgba, "rgb": c_rgb, "pid": c_pid, "harm": c_harm,
                        "q": q, "s": sval, "seed": cand["seed"]}
        patch_rgba, patch_rgb, prompt_id = best["rgba"], best["rgb"], best["pid"]
        harmonized, quality = best["harm"], best["q"]
        quality["selected"] = {"n": n, "seed": best["seed"]}

    out = {
        "objectId": int(object_id),
        "x": gx0, "y": gy0, "w": gw, "h": gh,
        "png": _data_url(patch_rgba),
        "engine": "sdxl",
        "promptId": prompt_id,
        "upgrades": {
            "soft_inpaint": soft_active,
            "controlnet": cn_kind if cn_cfg else None,
            "style": detected_style,
            "harmonized": harmonized,
        },
        "harmonized": harmonized,
    }
    if n > 1:
        out["upgrades"]["bestOfN"] = n
    if quality is not None:
        out["quality"] = quality
    if analysis is not None:
        out["plan"] = analysis
    if extra:
        out.update(extra)
    return out
