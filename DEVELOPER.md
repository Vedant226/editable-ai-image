# Editable AI Image Engine — Developer Guide

Turn a single AI-generated image into a Photoshop-Smart-Object-style editor:
every meaningful element (people, faces, crowns, ornaments, borders, emblems,
text, …) becomes individually selectable, and editing it feels like editing the
**original image itself** — no visible extraction, duplication, seams, or
"pasted PNG" look.

---

## 1. System overview

```
                        OFFLINE (per image, GPU)                         RUNTIME (browser + backend)
 image ─▶ Universal Extraction Pipeline ─▶ layers/ + metadata.json ─▶ React editor ─▶ FastAPI (/lift, /inpaint)
          (SAM-everything + DINO + CLIP            (RGBA cutouts,        (Object Manager,    (matte + LaMa inpaint
           + OCR + matting + hierarchy)             bitmap text)          Visual Resolver,    of the original)
                                                                          Illusion Engine)
```

Two halves:

- **Extraction (offline, Python, `pipeline/`)** — decomposes one image into a
  rich, labeled, deduplicated, hierarchical object set with soft RGBA mattes and
  bitmap text. Run once per image.
- **Editor (runtime, React + FastAPI)** — loads that metadata and lets the user
  lift/move/resize/rotate/delete/edit objects while preserving the illusion of
  editing the original.

The original image is always the visual source of truth; it is **occluded,
never reconstructed**.

---

## 2. Runtime architecture (editor)

Three pure, separately-testable subsystems plus a thin React/Konva shell and a
FastAPI backend.

| Subsystem | File(s) | Owns | Pure? |
|---|---|---|---|
| **Object Manager (OM)** | `editable-editor/src/objectManager/` | identity, classification, dedup, grouping, selection (`pickAt` + escalation), session transitions | yes |
| **Visual Object Resolver (VOR)** | `editable-editor/src/visualResolver/` | which pixels represent an object; the pixel-ownership partition (anti-duplicate) | yes |
| **Editing Illusion Engine (EIE)** | `editable-editor/src/editingIllusion/` | the perceptual transition over *time* (RESTING→LIFTING→FLOATING→SETTLING→PLACED / DELETING); the six invariants | stateful state-machine, framework-agnostic |
| **Alpha hit-tester** | `editable-editor/src/useAlphaHitTester.js` | pixel-perfect selection from PNG alpha | React hook |
| **Shell** | `editable-editor/src/App.jsx` | Konva stage, RAF clock, session (undo/redo), toolbar, inline text editor, prefetch/IO | React |
| **Backend** | `backend/` | `/lift`, `/inpaint`, `/health` — matting + LaMa inpaint of the original | FastAPI |

**Pipeline of responsibility:** OM (which object) → VOR (which pixels / mask
geometry) → EIE (how it transitions, masked by motion/cross-fade) →
React/Konva draw + FastAPI/LaMa pixel work.

### The six illusion invariants (enforced by the EIE)
1. **Continuity** — no frame shows a hole/seam/duplicate/raw edge.
2. **Identity-at-rest** — idle canvas is pixel-identical to the original.
3. **No premature displacement** — an object's pixels never leave their origin
   until the footprint is covered (inpaint patch or temp fill).
4. **Single instance** — base *or* float, never both/neither visible.
5. **Edge integrity** — feathered, decontaminated mattes; feathered fill collars.
6. **Masked swaps** — every base↔float swap rides motion or a cross-fade.

### Selection model
`pickAt(point)` returns the most-specific editable object whose **alpha** covers
the point; re-clicking the selection escalates (leaf → person). Activation lifts
the object as a Smart Object; the `/lift` cutout (refined, decontaminated)
replaces the SAM PNG in place, a feathered fill covers the footprint, a matte-
hugging glow marks selection.

---

## 3. Extraction pipeline (`pipeline/`)

Production entry point is `engine.py` (single process, all models loaded once).
The per-phase `run_xN.py` scripts are diagnostic tools that persist intermediate
artifacts + HTML reports to `pipeline/_work/` (gitignored).

| Phase | Module | Does |
|---|---|---|
| X1 | `proposals.py` | **Universal proposals** — SAM `AutomaticMaskGenerator` ("segment everything", recall) ∪ GroundingDINO→SAM (semantic anchor). Tagged by `source`. |
| X2 | `clip_labeler.py` · `geometry.py` · `fusion.py` | **Semantic Fusion Labeling** — CLIP zero-shot + DINO label + geometry/position fused into `{category, confidence, importance, editable, evidence}`. DINO is additive-only. |
| X3 | `hierarchy.py` | **Dedup** (mask-IoU NMS, keep best, alias rest) + **hierarchy** (part → smallest containing person). |
| X4 | (matting) | **Soft RGBA cutouts** — guided-filter matte (`backend/matting.py`), cropped to footprint. |
| X5 | `run_x5.py` (`estimate_style`) | **BitmapTextObjects** — EasyOCR; store original text pixels + string + polygon + baseline + font estimation. |
| X6 | `run_x6.py` (`om_type`, `zindex`) | **Backward-compatible metadata** — superset schema; `type` = OM-known token, precise label in `category`. |
| X7 | (editor) | editor renders the text **bitmap** until edited, then synthesizes React Text. |

**Models** (all in the venv; CLIP weights download once): SAM `vit_h` (GPU),
GroundingDINO SwinT (GPU), CLIP ViT-B/32 via `transformers` (**CPU**), EasyOCR
(**CPU**). CLIP/OCR run on CPU so SAM's encoder fits an ~8GB GPU.

**Run (any image):**
```bash
python -m pipeline.run_all --image path/to/image.png --out pipeline/_out/layers
cp pipeline/_out/layers/* editable-editor/public/layers/        # deploy
```

---

## 4. Metadata schema (`public/layers/metadata.json`)

A JSON array of objects. The editor reads the first block; the rest is additive
(ignored by older readers).

```jsonc
{
  "id": 42, "file": "42_crown.png", "type": "crown",      // type = OM-known editable token
  "x": 875, "y": 230, "width": 52, "height": 46, "rotation": 0, "zIndex": 6,
  // --- additive (from the fusion/hierarchy pipeline) ---
  "category": "crown",            // precise label (decor kept distinct)
  "confidence": 0.78, "importance": 0.74, "editable": true,
  "kind": "object",               // "object" | "bitmap_text"
  "parent": 54, "children": [16,17],
  "source": "sam" | "dino" | "ocr" | "base",
  "evidence": { "clip": 0.86, "dino": null, "geometry": 0.60 },
  // --- bitmap_text only ---
  "text": "CROWNED HEADS", "fontFamily": "serif", "fontSize": 57,
  "fontWeight": "bold", "fontColor": "#e3c38c", "baseline": 96, "polygon": [[..]]
}
```
`background.png` is object id `99999` (`type:"background"`, `editable:false`).
The original `metadata.json` is never hand-edited; the pipeline regenerates it.

---

## 5. Backend workflow (`backend/`)

```bash
cd backend && ./run.sh            # serves http://127.0.0.1:8000 ; Vite proxies /api -> here
```
- `GET /health` → `{engine: "lama"|"opencv", device, objects}`
- `POST /lift {objectId}` → Lift Package: `cutout` (original × refined
  decontaminated matte, footprint-cropped), `fill` (LaMa inpaint of footprint
  ∪ shadow, feathered collar), `shadow` (synthesized soft matte). Returns 404 if
  the object/layer file is missing.
- `POST /inpaint {objectId}` → footprint inpaint patch.

LaMa is loaded directly from `~/.cache/torch/hub/checkpoints/big-lama.pt` via
`torch.jit` (lama-cleaner's `ModelManager` is incompatible with the installed
`huggingface_hub`); OpenCV `cv2.inpaint` is the automatic fallback.

---

## 6. Benchmark methodology (`pipeline/benchmark.py`)

Category-agnostic, **no-reference** (no ground truth needed) so it works on any
AI image. Drop images in `pipeline/benchmark_images/`, run
`python -m pipeline.benchmark`. Per-image metrics → `pipeline/_work/benchmark/`
(JSON + HTML):

| Metric | Proxy for |
|---|---|
| `proposals`, `canonical`, `dedup_ratio` | proposal quality, duplicate suppression |
| `editable`, `editable_coverage` | editable coverage (% canvas owned by editable masks) |
| `avg_confidence`, `pct_uncertain` | semantic accuracy |
| `avg_solidity` | mask quality |
| `bitmap_text` | text preservation |
| `timings` | performance |

No-reference metrics measure *internal* quality/robustness; absolute semantic
accuracy across categories requires a labeled set (see limitations).

---

## 7. Deployment

1. Extract: `python -m pipeline.run_all --image IMG --out pipeline/_out/layers`
2. Deploy assets: `cp pipeline/_out/layers/* editable-editor/public/layers/`
3. Backend: `cd backend && ./run.sh`
4. Editor: `cd editable-editor && npm run dev` (build: `npm run build`)

The editor degrades gracefully if the backend is down (HUD shows
`lift: offline`; objects still select/move using the SAM stand-in).

---

## 8. Tests

```bash
cd editable-editor
npm run om:test      # 19 Object Manager + Visual Resolver checks
npm run eie:test     # 28 Editing Illusion Engine state-machine checks
npm run om:report    # semantic breakdown of the deployed metadata
npm run build        # production build
python -m pipeline.benchmark   # pipeline benchmark (from repo root)
```

---

## 9. Repo layout

```
pipeline/            extraction pipeline (engine.py = production; run_xN = diagnostics)
  benchmark_images/  drop images here for the benchmark
backend/             FastAPI: app.py, lama_engine.py, matting.py, shadow.py
editable-editor/     Vite React app
  src/objectManager/ | visualResolver/ | editingIllusion/ | App.jsx
  public/layers/     deployed extraction output (metadata.json + PNGs)
*.py (repo root)     LEGACY one-shot extraction experiments, superseded by pipeline/
```

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| CUDA OOM during extraction | SAM `vit_h` is heavy; keep `crop_n_layers=0`, run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. CLIP/OCR already on CPU. |
| `lift: offline` in editor | backend not running → `cd backend && ./run.sh`. Editor still works (SAM stand-in, no clean fill). |
| `/health` shows `engine: opencv` | `big-lama.pt` not loadable; inpaint falls back to OpenCV (lower quality but functional). |
| Text looks like the original until edited | intended — bitmap text preserves original pixels; React Text is synthesized only on edit. |
| New categories not editable | extraction maps `type` to an OM-known token via `pipeline/run_x6.py: OM_TYPE`; add a mapping there (don't change the OM). |
| Moving a large central figure leaves a faint ghost | LaMa limitation on large person-shaped holes; small objects fill cleanly. |
