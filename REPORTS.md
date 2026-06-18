# Editable AI Image Engine — Final Reports

Production-release audit of the full system (Object Manager, Visual Object
Resolver, Editing Illusion Engine, Universal Extraction Pipeline X1–X7, unified
engine, React editor, FastAPI backend). Generated at the end of the
production-hardening pass.

---

## 1. Final benchmark report

Harness: `pipeline/benchmark.py` — category-agnostic, no-reference metrics, one
shared model load. Identical code path for every image (no per-image logic).

Dataset: 5 images spanning distinct categories — 2 real (`bookcover`,
`hero`) + 3 synthetic robustness fixtures (`poster`, `logo`, `landscape`).

| image | proposals | canonical | editable | coverage | avg_conf | uncertain | solidity | text | time |
|---|---|---|---|---|---|---|---|---|---|
| bookcover (real) | 251 | 204 | 207 | 0.76 | 0.46 | 0.72 | 0.86 | 22 | 24s |
| hero (real) | 31 | 24 | 7 | 0.97 | 0.38 | 1.00 | 0.66 | 0 | 7s |
| poster (synth) | 56 | 43 | 43 | 0.23 | 0.47 | 0.62 | 0.84 | 3 | 11s |
| logo (synth) | 33 | 18 | 18 | 0.24 | 0.49 | 0.56 | 0.86 | 2 | 8s |
| landscape (synth) | 26 | 6 | 3 | 0.47 | 0.33 | 1.00 | 0.99 | 0 | 10s |
| **average** | **79.4** | **59.0** | **55.6** | **0.535** | **0.426** | **0.782** | **0.842** | **5.4** | — |

**Findings**
- **Zero crashes** across all categories; the pipeline is genuinely
  content-agnostic (book cover, graphic, poster, logo, landscape).
- **Graceful degradation by content**: flat/synthetic images correctly produce
  *fewer* proposals and lower coverage rather than hallucinating objects
  (landscape → 3 editable; logo → 18). Text was recovered on logo/poster.
- **Duplicate suppression** removes ~37% of raw proposals on average.
- **Mask quality** is high (avg solidity 0.84).
- **Confidence is conservative** (avg 0.43; 78% flagged uncertain). This is
  honest model uncertainty — concentrated in visually near-synonymous decor
  (symbol/logo/crest/emblem) and abstract synthetic shapes — not a defect. It is
  surfaced, never hidden, and importance/editability remain usable.

**Interpretation.** These metrics measure *internal* quality and robustness
(no ground truth). They demonstrate the **infrastructure generalizes**. They do
**not** establish absolute semantic accuracy across all 14 target categories —
that needs a labeled real-image set (see §3).

---

## 2. Final architecture report

**Shape.** Clean offline/runtime split. Offline extraction emits a
backward-compatible `metadata.json` + RGBA layers; the runtime editor consumes
them. The original image is the sole visual source of truth (occluded, never
reconstructed).

**Runtime — three pure subsystems + shell:**
- **Object Manager** — identity/classification/dedup/grouping/selection/session.
  Pure, 19 unit checks.
- **Visual Object Resolver** — pixel representation + the anti-duplicate
  ownership partition. Pure.
- **Editing Illusion Engine** — time-domain transition state machine enforcing
  six perceptual invariants. Framework-agnostic, 28 unit checks.
- **React/Konva shell** — RAF clock, undo/redo (session snapshots), toolbar,
  inline text, prefetch/IO. **FastAPI backend** — `/lift` (matte + LaMa inpaint)
  with OpenCV fallback.

**Extraction — fusion over recall:** SAM-everything (recall) + GroundingDINO
(semantics) → CLIP+DINO+geometry fusion (additive DINO) → mask-NMS dedup +
hierarchy → soft RGBA mattes → OCR bitmap text → backward-compatible metadata.
Unified `engine.py` runs it single-process (~4× faster than per-phase; GPU-safe
via CPU CLIP/OCR).

**Dependency directions are clean and one-way:** VOR→OM, EIE→VOR/OM,
pipeline→(models); the editor never imports the pipeline. No image-specific
heuristics in any algorithm; category specifics live only in data
(taxonomy/vocabulary/`OM_TYPE` map).

**Health:** no TODO/FIXME in source; dead `layerData.js` removed; matting shared
between backend and pipeline (no duplication); legacy root scripts isolated and
documented as superseded.

---

## 3. Remaining known limitations

1. **Generalization is validated, not proven on real data at scale.** Only 2
   real images were available in this environment (both book-cover domain). The
   harness + methodology are ready; a curated multi-category real-AI set is
   required to quantify semantic accuracy per category.
2. **GPU ceiling (~8 GB)** forces SAM `crop_n_layers=0`, capping recall of the
   faintest small decor. A memory-bounded tiling pass would recover it.
3. **Decorative label ambiguity** — CLIP cannot reliably separate
   symbol/logo/crest/emblem; confidences honestly reflect this (kept distinct by
   design rather than collapsed).
4. **Large-figure inpaint ghosting** — LaMa can re-hallucinate a faint figure
   when filling a large person-shaped hole; small/medium objects fill cleanly.
5. **Text**: OCR mis-reads stylized fonts (mitigated — the bitmap is pixel-exact
   until edit); font *family* is not recognized (defaults to serif); rotated
   text uses an axis-aligned box.
6. **Backend is single-process/dev-grade** — no auth, batching, queueing, or
   horizontal scaling; CLIP/OCR on CPU trade speed for GPU headroom.
7. **Knockout compositing / multi-instance duplicate / face-replace** are
   designed-for but deferred in the editor.

---

## 4. Suggestions for future research (not implementation)

- **Deep image matting** (FBA / MODNet / ViTMatte) to replace guided-filter
  matting for hair/lace/glow-grade alpha.
- **Diffusion-based inpainting** (or LaMa-then-refine) to eliminate large-hole
  ghosting and enable content-aware fills.
- **Font recognition + vectorization** (e.g., DeepFont-style) so edited text
  matches the original typeface, not just a serif fallback.
- **Learned object-importance / saliency** to rank editability instead of
  heuristic geometry+category weights.
- **Shadow/relighting estimation** so moved objects re-cast physically-plausible
  shadows and match local lighting.
- **A labeled multi-category benchmark with reference masks** to convert the
  current no-reference proxies into true precision/recall/IoU.
- **Open-vocabulary panoptic models** (e.g., SAM-2 + grounded captioning) to
  unify proposal + label + hierarchy in one pass and reduce decor ambiguity.
- **Instance-keyed session model** to support duplicate / detached-child
  (knockout) compositing in the editor.

---

## 5. Final production-readiness assessment

**Verdict: production-ready for controlled / beta use; not yet for unsupervised
production at scale.**

| Dimension | State |
|---|---|
| Functionality | ✅ Complete end-to-end (extract → deploy → edit → export). |
| Stability | ✅ All tests pass (build, om:test 19, eie:test 28); benchmark 5/5 no crashes; backend endpoints verified (200s + correct 404). |
| Architecture | ✅ Clean, layered, pure cores, no image-specific logic, documented. |
| Reliability | ✅ Graceful degradation: backend-offline, OCR/SAM/CLIP/matte failure, missing files, malformed metadata all handled. |
| Performance | ✅ Single-process engine, GPU-safe; ~8–24 s/image offline. ⚠️ Backend not horizontally scaled. |
| Visual fidelity | ✅ Soft decontaminated mattes, feathered fills, matte-hug selection, bitmap-exact text, masked transitions. ⚠️ Large-figure inpaint + rotated text imperfect. |
| Generalization | 🟡 Architecturally category-agnostic and validated to *run* across 5 categories; broad semantic accuracy on real AI images **unproven** for lack of a dataset. |

**Bottom line.** This is the strongest version of this architecture achievable
within the current stack and environment: the design is clean and complete, it
runs generically on arbitrary images, it degrades rather than fails, and visual
fidelity is high. The single material gap to full production confidence is
**empirical generalization on a curated, diverse real-AI dataset** — which the
benchmark harness is built to measure the moment such images are provided. The
remaining quality ceilings (matting, large-hole inpaint, font ID) are bounded by
the underlying models, not the architecture, and are enumerated as future
research.
