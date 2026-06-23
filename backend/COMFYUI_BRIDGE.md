# ComfyUI bridge (AI Integration — Phases 1–2)

The communication layer between this FastAPI backend and a locally running
**ComfyUI** server, plus the first real AI feature built on it.

- **Phase 1 — transport only**: reach ComfyUI, report status, queue/poll/fetch.
- **Phase 2 — Replace Object**: swap one selected object for an uploaded image,
  preserving its footprint, the scene's lighting/perspective/shadows, and every
  surrounding pixel.
- **Phase 3 — Background Replacement**: keep every foreground object and
  regenerate only the background from an uploaded image, harmonising lighting.
- **Phase 4 — Remove Object**: erase a selected object and generate a realistic,
  lighting-consistent background fill over its footprint (no upload).
- **Phase 5 — Recolor Object**: change an object's colour while preserving
  texture, lighting, shadows, highlights and reflections (LAB luminance kept).
- **Phase 6 — Change Clothes**: regenerate only a person's clothing from a
  prompt; face, hair, body pose, legs and background are preserved.
- **Phase 7 — Change Hair**: regenerate only a person's hair from a prompt;
  face, eyes, eyebrows, ears, clothing, body and background are preserved.
- **Phase 8 — Person Replace**: replace the whole person with an uploaded image
  or a prompt-generated one; background, neighbours and perspective are preserved.
- **Phase 9 — Identity Preservation Engine**: an optional, reusable identity layer
  for Hair / Clothes / Person Replace (face identity preserved / matched to a
  reference). Additive — OFF by default, existing features unchanged.
- **Phase 10 — Lighting & Shadow Harmonization Engine**: an optional CPU post-
  process (no SDXL rerender, <2 s) that matches an AI patch's lighting to the
  scene + edge-blends + contact shadows. Additive — "Auto Lighting" OFF by default.

It is fully isolated. The existing inpaint service (`app.py`, `lama_engine.py`,
…) is not imported or modified.

## Files

- `comfy_client.py` — `ComfyUIClient`, the pure HTTP client for ComfyUI.
- `comfy_replace.py` — Replace Object orchestration (mask/crop/composite + run).
- `comfy_background.py` — Background Replacement orchestration (foreground union
  mask + cover-fit upload + re-composite crisp foreground).
- `comfy_remove.py` — Remove Object orchestration (reuses comfy_replace's mask/
  geometry helpers; OpenCV coarse fill + SDXL refine).
- `comfy_recolor.py` — Recolor Object orchestration (reuses comfy_replace's
  helpers; LAB luminance-preserving recolour + low-denoise SDXL refine).
- `comfy_clothes.py` — Change Clothes orchestration (reuses comfy_replace's
  helpers; derives the clothing sub-mask from the person footprint + linked face).
- `comfy_hair.py` — Change Hair orchestration (reuses comfy_replace's helpers and
  comfy_clothes' `_find_face`; hair = head region minus expanded face box).
- `comfy_person.py` — Person Replace orchestration (reuses comfy_replace's
  helpers; upload mode composites a footprint-clipped person, prompt mode
  denoises from the original to inherit geometry).
- `identity_manager.py` — Identity capability detection (IP-Adapter FaceID /
  InstantID nodes vs insightface vs OpenCV) + face detection / ArcFace embeddings.
- `comfy_identity.py` — Identity Preservation Engine (landmark-aligned face
  transfer + lighting match; IP-Adapter FaceID workflow path when available).
- `workflows/identity_preserve.json` — IP-Adapter FaceID (SDXL) template, used
  only when those nodes/models are installed (engine falls back otherwise).
- `lighting_manager.py` — Lighting config loader + scene-light analysis
  (brightness/exposure/white-balance/CCT/contrast/saturation/gamma/histogram/dir).
- `comfy_lighting.py` — Lighting & Shadow Harmonization Engine (CPU/OpenCV;
  LAB tone match + contact shadow + edge blend; no SDXL; safe no-op on failure).
- `workflows/lighting_harmonization.json` — lighting PIPELINE CONFIG (params, not
  an SDXL graph).
- `workflows/replace_object.json`, `workflows/background_replace.json`,
  `workflows/remove_object.json`, `workflows/recolor_object.json`,
  `workflows/change_clothes.json`, `workflows/change_hair.json`,
  `workflows/person_replace.json` — the SDXL workflow templates (graph never
  hardcoded in Python; the orchestration modules inject values by node id).
- `comfy_router.py` — `APIRouter`: `GET /comfyui/status`, `POST /comfyui/replace`,
  `POST /comfyui/background`, `POST /comfyui/remove`, `POST /comfyui/recolor`,
  `POST /comfyui/clothes`, `POST /comfyui/hair`, `POST /comfyui/person`,
  `POST /comfyui/identity`, `GET /comfyui/identity/capabilities`,
  `POST /comfyui/lighting`, `GET /comfyui/lighting/config`.
- `comfy_app.py` — standalone FastAPI app that mounts the router.
- `run_comfy_bridge.sh` — runs the standalone app on `127.0.0.1:8189`.

## Required model

Replace Object uses an SDXL checkpoint loaded by ComfyUI's
`CheckpointLoaderSimple`. Default: `sd_xl_base_1.0.safetensors` in
`ComfyUI/models/checkpoints/` (override with `COMFY_REPLACE_CKPT`). If it is
missing, `POST /comfyui/replace` returns **502** with ComfyUI's
`value_not_in_list` error naming the checkpoint.

## Run

```bash
cd backend
./run_comfy_bridge.sh                 # serves http://127.0.0.1:8189
# then:
curl http://127.0.0.1:8189/health     # bridge process is up
curl http://127.0.0.1:8189/comfyui/status
```

`/comfyui/status` always returns 200. `online: true` with version/device/queue
when ComfyUI is up; `online: false` with an `error` string when it is offline.

### Replace Object

```
POST /comfyui/replace
{
  "objectId": 114,                 # id from layers/metadata.json
  "replacement": "data:image/png;base64,...",   # uploaded image (data URL or base64)
  "denoise": 0.6,                  # optional; how hard SDXL harmonises (0..1)
  "prompt": "...", "negative": "...", "steps": 28, "cfg": 7.0,
  "seed": 0, "sampler": "dpmpp_2m", "scheduler": "karras"   # all optional
}
-> 200 { "objectId", "x", "y", "w", "h", "png" }   # RGBA patch to drop at (x,y)
```

The frontend overlays `png` on the object's footprint only. Errors:
**503** ComfyUI offline · **502** ComfyUI rejected the graph (e.g. checkpoint
missing) · **504** render timed out · **404** unknown/empty object.

### Background Replacement

```
POST /comfyui/background
{
  "replacement": "data:image/png;base64,...",   # uploaded background (data URL or base64)
  "denoise": 0.6,                                # optional; default 0.6
  "prompt": "...", "negative": "...", "steps": 28, "cfg": 7.0,
  "seed": 0, "sampler": "dpmpp_2m", "scheduler": "karras"   # all optional
}
-> 200 { "x":0, "y":0, "w", "h", "png", "full": true }   # full canvas (RGB)
```

Keeps every foreground object (union of all layer alphas) and regenerates only
the background, re-compositing the crisp original foreground at full resolution.
The editor draws `png` as the new base layer. Same error codes as `/replace`.

### Remove Object

```
POST /comfyui/remove
{
  "objectId": 153,          # id from layers/metadata.json
  "denoise": 0.82,          # optional; higher = more regeneration
  "prompt": "...", "negative": "...", "steps": 28, "cfg": 7.0,
  "seed": 0, "sampler": "dpmpp_2m", "scheduler": "karras"   # all optional
}
-> 200 { "objectId", "x", "y", "w", "h", "png" }   # RGBA footprint patch
```

Erases the object: an OpenCV coarse fill removes it, then SDXL refines a
realistic, lighting-matched background over the (slightly expanded) footprint.
The frontend drops `png` on the footprint via the same overlay as `/replace`.
Same error codes as `/replace`.

### Recolor Object

```
POST /comfyui/recolor
{
  "objectId": 153,
  "targetColor": "#1565ff",   # '#rgb' / '#rrggbb' / 'rgb(r,g,b)' / [r,g,b]
  "denoise": 0.28,            # optional; low by default so texture is kept
  "prompt": "...", "negative": "...", "steps": 24, "cfg": 6.0,
  "seed": 0, "sampler": "dpmpp_2m", "scheduler": "karras"   # all optional
}
-> 200 { "objectId", "targetColor", "x", "y", "w", "h", "png" }   # RGBA patch
```

Recolours in LAB space (luminance kept → texture/lighting/shadows/highlights/
reflections preserved; chroma rotated to the target hue, rolled off in
highlights so speculars stay white), then a low-denoise SDXL pass refines.
Frontend uses the native color picker and the same overlay as `/replace`. Same
error codes as `/replace`.

### Change Clothes

```
POST /comfyui/clothes
{
  "objectId": 157,
  "prompt": "a blue denim jacket",   # describe the new clothing (required)
  "denoise": 0.9,                     # optional; high — clothing is generated
  "negative": "...", "steps": 30, "cfg": 7.0,
  "seed": 0, "sampler": "dpmpp_2m", "scheduler": "karras"   # all optional
}
-> 200 { "objectId", "x", "y", "w", "h", "png", "usedFace" }   # RGBA clothing patch
```

Regenerates ONLY the clothing region. The region is derived from the person
footprint: the head is excluded via the person's linked FACE object (metadata
parent/children, else a head-fraction fallback), and a lower strip is excluded
as legs for tall full-body figures. Face, hair, pose, legs and background are
preserved exactly. Frontend uses a prompt dialog and the same overlay as
`/replace`. Same error codes as `/replace`.

### Change Hair

```
POST /comfyui/hair
{
  "objectId": 157,
  "prompt": "long blonde wavy hair",   # hairstyle and/or colour (required)
  "denoise": 0.85,                      # optional; high — hair is generated
  "negative": "...", "steps": 30, "cfg": 7.0,
  "seed": 0, "sampler": "dpmpp_2m", "scheduler": "karras"   # all optional
}
-> 200 { "objectId", "x", "y", "w", "h", "png", "usedFace" }   # RGBA hair patch
```

Regenerates ONLY the hair region = the head region of the footprint (above the
chin) MINUS an expanded face box (eyes/eyebrows/nose/mouth/ears, via the linked
face object). The patch's alpha is forced to 0 inside the face box, so the face
stays byte-identical; clothing, body and background are untouched. Frontend uses
a prompt dialog and the same overlay as `/replace`. Same error codes as `/replace`.

### Person Replace

```
POST /comfyui/person
{
  "objectId": 157,
  "image": "data:image/png;base64,...",   # EITHER an uploaded replacement person
  "prompt": "a young queen in a green gown", # OR a description to generate one
  "denoise": null,                          # optional; default 0.55 upload / 0.82 prompt
  "negative": "...", "steps": 30, "cfg": 7.0,
  "seed": 0, "sampler": "dpmpp_2m", "scheduler": "karras"   # all optional
}
-> 200 { "objectId", "mode", "x", "y", "w", "h", "png" }   # RGBA footprint patch
```

Replaces the whole person within its footprint. **Upload mode** removes the old
person (coarse inpaint), composites the uploaded person clipped to the footprint
silhouette — which discards the upload's own background (transparent PNGs use
their alpha; opaque PNG/JPG are clipped) — and SDXL harmonises it (low denoise).
**Prompt mode** denoises from the original person, so the new one inherits its
scale, camera angle and body orientation (higher denoise). Background and
neighbouring objects are never touched. `400` if neither `image` nor `prompt` is
given; otherwise same error codes as `/replace`.

### Identity Preservation Engine (optional)

```
POST /comfyui/identity
{
  "referenceImage": "data:image/png;base64,...",   # the reference face
  "targetImage": "data:image/png;base64,...",       # image to receive the identity
  "mask": "data:image/png;base64,...",              # optional footprint (else target alpha)
  "strength": 0.9                                    # 0..1
}
-> 200 { "png", "method", "faceFound", "similarity", "strength", "capabilities" }

GET /comfyui/identity/capabilities -> { method, insightface, comfy_ipadapter_faceid, ... }
```

Imposes the reference face's identity onto the target's face region (eye/nose/
mouth/jaw/skin-tone/proportions preserved; hair/clothing/accessories/background/
pose free to change). Method is chosen automatically: IP-Adapter FaceID /
InstantID via ComfyUI when those nodes+models exist (graph in
`identity_preserve.json`), else **insightface face-lock** (landmark-aligned face
transfer + lighting match; real ArcFace `similarity`), else OpenCV. If no face is
found the target is returned unchanged, so enabling identity never breaks a
result.

**Optional integration (frontend):** a single "Preserve Identity" toggle (OFF by
default) in the toolbar. When ON, Change Hair / Change Clothes / Person Replace
additionally call `/comfyui/identity` after their patch is produced (reference =
the original person, or the uploaded image for Person Replace upload mode). When
OFF, those features behave exactly as before. Existing feature modules and
workflow JSONs are not modified.

### Lighting & Shadow Harmonization (optional)

```
POST /comfyui/lighting
{
  "patch": "data:image/png;base64,...",     # the AI patch (RGBA) or a full image
  "reference": "data:image/png;base64,...",  # original scene (margined crop)
  "patchLeft": 60, "patchTop": 60,           # patch offset within the reference
  "strength": 0.7                            # optional global blend
}
-> 200 { "png", "ok", "report", "capabilities" }   # always 200; on failure ok=false + original png

GET /comfyui/lighting/config -> the pipeline config
```

A fast CPU/OpenCV post-process (no SDXL rerender, <2 s) that matches the patch's
lighting to the surrounding scene (brightness/exposure/white-balance/colour-temp/
contrast/gamma/saturation), extends a conservative contact shadow, and soft-blends
the edge — geometry/texture/alpha preserved (global LAB affine). `report` carries
before/after brightness, colour temperature, histogram distance, etc.

**Optional integration (frontend):** a single "Auto Lighting" toggle (OFF by
default). When ON, Replace / Hair / Clothes / Background / Person pass their patch
through `/comfyui/lighting` after the AI edit (and after identity). Remove and
Recolor never do. When OFF, behaviour is identical. On any failure the original
patch is returned, so editing is never blocked.

The Vite dev server proxies `/comfyui/*` → `:8189`, so the editor calls
`/comfyui/replace` and `/comfyui/background` with no CORS setup.

## Config (env vars)

- `COMFYUI_URL` — ComfyUI base URL (default `http://127.0.0.1:8188`)
- `COMFYUI_TIMEOUT` — per-request timeout in seconds (default `5.0`)

## Client surface (`ComfyUIClient`)

`is_available()`, `status()`, `system_stats()`, `queue_prompt(graph)`,
`get_history(prompt_id)`, `await_result(prompt_id)`, `get_queue()`,
`get_image(filename, …)`, `upload_image(bytes, filename)`, `interrupt()`,
`get_object_info()`. Offline/timeout failures raise `ComfyUIUnavailable`;
reached-but-errored responses raise `ComfyUIRequestError`.

## Merging into the main service (later, your call)

To expose status through the existing Vite `/api` proxy on `:8000`, add one
additive line to `app.py` — no existing code changes:

```python
from comfy_router import router as comfy_router
app.include_router(comfy_router)
```
