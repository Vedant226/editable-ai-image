# Inpaint backend

Thin FastAPI service that fills the hole an object leaves when it is lifted as a
Smart Object in the editor.

## Run

```bash
cd backend
./run.sh           # serves http://127.0.0.1:8000
```

The Vite dev server proxies `/api/*` → `http://127.0.0.1:8000`, so the frontend
calls `/api/inpaint` and `/api/health` with no CORS setup.

## Endpoints

- `GET /health` → `{ engine: "lama" | "opencv", device, objects }`
- `POST /inpaint` `{ "objectId": <int> }` → `{ x, y, w, h, png }`
  where `png` is a base64 data URL of the inpainted patch cropped to the
  object's footprint, to be placed at `(x, y)` on the repairs layer.

## Engine

Loads `~/.cache/torch/hub/checkpoints/big-lama.pt` directly via `torch.jit`
(lama-cleaner's `ModelManager` is incompatible with the installed
`huggingface_hub`). Falls back to OpenCV `cv2.inpaint` if the LaMa weights can't
be loaded. Check which is active via `/health`.

## Config (env vars)

- `LAYERS_DIR` — path to the layers dir (default `../editable-editor/public/layers`)
- `MASK_DILATE` — mask dilation px (default 7)
- `CONTEXT_MARGIN` — context px around the footprint fed to LaMa (default 64)
- `LAMA_JIT_PATH` — override the big-lama.pt path
