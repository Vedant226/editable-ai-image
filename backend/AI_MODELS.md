# AI model provisioning — Phase 3 quality upgrades

The ComfyUI editing engine (`comfy_engine.py`) **auto-detects** what's installed
and upgrades quality opportunistically, falling back to the proven base-SDXL
pipeline when an asset is missing. Nothing here is required to run; each item
unlocks a specific quality gain.

Check what's currently active any time:

```bash
curl -s localhost:8189/comfyui/capabilities | python3 -m json.tool
```

Install with the helper (downloads into `ComfyUI/models/…`), then **restart ComfyUI**:

```bash
./install_ai_models.sh controlnet     # recommended
./install_ai_models.sh ipadapter      # optional (identity via diffusion)
```

## What's active out of the box (no download)

| Upgrade | Mechanism | Status |
|---|---|---|
| **Seamless blending** | `DifferentialDiffusion` + soft mask (core node) | ✅ active now |
| **Style preservation** | painted-style prompts (drop "photorealistic") | ✅ active now |
| **Better object removal** | LaMa fill (via the `:8000` service) → low-denoise SDXL refine | ✅ active when `:8000` is up |
| **Identity** | insightface ArcFace face-lock (pip) | ✅ active now |

## What a download unlocks

| Model | File → folder | Size | Unlocks | VRAM (4060 8 GB) |
|---|---|---|---|---|
| ControlNet **union** (xinsir promax) | `models/controlnet/controlnet-union-sdxl-promax.safetensors` | ~2.5 GB | **canny/structure** lock for replace, clothes, hair, person (perspective, folds, head shape) | SDXL+1 CN fits with model offloading; use `--lowvram` if OOM |
| ControlNet **tile** (xinsir) | `models/controlnet/controlnet-tile-sdxl.safetensors` | ~2.5 GB | **texture / high-frequency** lock for recolor + detail recovery (recolor can raise denoise without flattening) | as above |
| IP-Adapter FaceID + CLIP-ViT-H | `models/ipadapter/`, `models/clip_vision/` + `custom_nodes/ComfyUI_IPAdapter_plus` | ~2 GB | **diffusion** identity transfer (vs the insightface fallback) | load FaceID on CPU provider; one adapter at a time |
| SDXL **inpaint** checkpoint (optional) | `models/checkpoints/*inpaint*.safetensors` | ~6 GB | proper inpaint conditioning (`InpaintModelConditioning`) — strongest seams | heavy; DifferentialDiffusion already covers most of this |

## How the engine chooses (per edit)

- `comfy_capabilities.detect()` reads ComfyUI's live `/object_info` (authoritative).
- Each feature requests a ControlNet **kind** (replace/clothes/hair/person → `canny`;
  recolor → `tile`); the engine uses it **only if that model is installed**.
- The control **hint** is computed in Python (cv2 Canny; tile = the image), so **no
  preprocessor custom node is needed** — only the ControlNet model file.
- `DifferentialDiffusion` is used whenever present (it is, in core ComfyUI).
- Every per-request `upgrades` field in the response reports what was actually applied.

## 8 GB guidance

- The engine already processes only a **context crop** (≤1024 long-side), not the
  full canvas — the main reason SDXL + a ControlNet fits 8 GB.
- Never stack ControlNets (the engine applies at most one per render).
- If you hit CUDA OOM, launch ComfyUI with `--lowvram` (or `--medvram`).
- Keep the `:8000` LaMa service running for best object removal.
