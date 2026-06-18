"""
Phase X2 runner — Semantic Fusion Labeling over the X1 proposal pool.

Loads the candidate masks/manifest from pipeline/_work, labels every proposal by
fusing CLIP + DINO + geometry, writes a NEW semantic metadata file (never
touches metadata.json), prints statistics, and renders an HTML validation
report (original crop + mask + category + confidence + editable + evidence).

  python -m pipeline.run_x2      (from the repo root)
"""

import base64
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from . import config as C
from . import clip_labeler
from .geometry import features as geom_features, category_scores as geom_scores
from .fusion import fuse, map_phrase_to_category

WORK = C.WORK_DIR
MASKS_DIR = os.path.join(WORK, "cand_masks")
CROPS_DIR = os.path.join(WORK, "crops")


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def main():
    os.makedirs(CROPS_DIR, exist_ok=True)
    image = np.array(Image.open(C.IMAGE_PATH).convert("RGB"))
    H, W = image.shape[:2]
    manifest = json.load(open(os.path.join(WORK, "candidates.json")))
    dino_cands = [c for c in manifest if c["source"] == "dino"]

    print(f"device: {C.DEVICE}  candidates: {len(manifest)}  (loading CLIP, first run downloads weights)")
    results = []
    for c in manifest:
        cid, (x, y, w, h) = c["id"], c["bbox"]
        crop_mask = np.array(Image.open(os.path.join(MASKS_DIR, f"cand_{cid}.png")).convert("L")) > 0
        img_crop = image[y : y + h, x : x + w]

        # CLIP on the masked object (neutral background)
        masked = np.where(crop_mask[:, :, None], img_crop, 128).astype("uint8")
        clip_probs = clip_labeler.classify(Image.fromarray(masked))

        # geometry / position
        gf = geom_features(crop_mask, c["bbox"], W, H)
        gs = geom_scores(gf)

        # DINO evidence: own label, else inherit from the best-overlapping DINO box
        if c["source"] == "dino":
            dino_cat = map_phrase_to_category(c["label"])
            dino_strength = float(c["score"])
        else:
            best, best_iou = None, 0.0
            for d in dino_cands:
                o = iou(c["bbox"], d["bbox"])
                if o > best_iou:
                    best, best_iou = d, o
            if best and best_iou > 0.4:
                dino_cat = map_phrase_to_category(best["label"])
                dino_strength = float(best["score"]) * best_iou
            else:
                dino_cat, dino_strength = None, 0.0

        decided = fuse(clip_probs, dino_cat, dino_strength, gf, gs)
        results.append(
            {
                "id": cid,
                "source": c["source"],
                "bbox": c["bbox"],
                "area": c["area"],
                "dinoLabel": c["label"],
                **decided,
                "geometry": gf,
            }
        )
        Image.fromarray(img_crop).save(os.path.join(CROPS_DIR, f"crop_{cid}.png"))

    json.dump(results, open(os.path.join(WORK, "semantic.json"), "w"), indent=1)

    # ---- statistics ----
    per_cat = Counter(r["category"] for r in results)
    editable = [r for r in results if r["editable"]]
    ignored = [r for r in results if not r["editable"]]
    uncertain = [r for r in results if r["confidence"] < C.UNCERTAIN_THRESHOLD]
    bitmap_text = [r for r in results if r["kind"] == "bitmap_text"]
    avg_conf = sum(r["confidence"] for r in results) / max(1, len(results))

    print("\n" + "=" * 64)
    print(f"PHASE X2 — SEMANTIC FUSION LABELING   ({len(results)} proposals)")
    print("=" * 64)
    print(f"  editable: {len(editable)}   ignored: {len(ignored)}   bitmap_text: {len(bitmap_text)}")
    print(f"  uncertain (<{C.UNCERTAIN_THRESHOLD}): {len(uncertain)}   avg confidence: {avg_conf:.3f}")
    print("  per-category (count):")
    for cat, n in per_cat.most_common():
        ed = sum(1 for r in results if r["category"] == cat and r["editable"])
        print(f"      {cat:14} {n:3}  ({ed} editable)")

    _html(results, per_cat, editable, ignored, uncertain, bitmap_text, avg_conf)
    print(f"\n  semantic metadata -> pipeline/_work/semantic.json")
    print(f"  HTML report       -> pipeline/_work/x2_report.html")
    print("=" * 64 + "\n")


def _thumb(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _html(results, per_cat, editable, ignored, uncertain, bitmap_text, avg_conf):
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    order = [c for c, _ in per_cat.most_common()]

    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{background:#15110d;color:#e8d9bf;font:13px/1.4 system-ui,sans-serif;margin:18px}",
        "h2{color:#d8b36a;border-bottom:1px solid #3a2f1c;padding-bottom:4px;margin-top:26px}",
        ".grid{display:flex;flex-wrap:wrap;gap:10px}",
        ".card{background:#211a12;border:2px solid #3a2f1c;border-radius:8px;padding:8px;width:190px}",
        ".card.edit{border-color:#4a7a43}.card.ign{border-color:#555}.card.unc{box-shadow:0 0 0 2px #c98a2e inset}",
        ".imgs{display:flex;gap:6px}.imgs img{width:84px;height:84px;object-fit:contain;background:#0c0a07;border-radius:4px}",
        ".cat{color:#d8b36a;font-weight:700;font-size:14px}.muted{color:#9a8a70}.bar{height:5px;background:#0c0a07;border-radius:3px;margin:2px 0}",
        ".bar>span{display:block;height:100%;background:#d8b36a;border-radius:3px}",
        "</style></head><body>",
        f"<h1 style='color:#d8b36a'>Phase X2 — Semantic Fusion Labeling</h1>",
        f"<p>{len(results)} proposals · editable {len(editable)} · ignored {len(ignored)} · "
        f"bitmap_text {len(bitmap_text)} · uncertain(&lt;{C.UNCERTAIN_THRESHOLD}) {len(uncertain)} · "
        f"avg confidence {avg_conf:.3f}</p>",
    ]
    for cat in order:
        rs = sorted(by_cat[cat], key=lambda r: -r["importance"])
        parts.append(f"<h2>{cat} &middot; {len(rs)}</h2><div class='grid'>")
        for r in rs:
            cls = "edit" if r["editable"] else "ign"
            if r["confidence"] < C.UNCERTAIN_THRESHOLD:
                cls += " unc"
            ev = r["evidence"]
            crop = _thumb(os.path.join(CROPS_DIR, f"crop_{r['id']}.png"))
            mask = _thumb(os.path.join(MASKS_DIR, f"cand_{r['id']}.png"))
            parts.append(
                f"<div class='card {cls}'><div class='imgs'><img src='{crop}'><img src='{mask}'></div>"
                f"<div class='cat'>{r['category']}  <span class='muted'>#{r['id']} {r['source']}</span></div>"
                f"<div>conf <b>{r['confidence']}</b> · imp {r['importance']} · {'editable' if r['editable'] else 'ignored'}"
                f"{' · <b style=color:#c98a2e>uncertain</b>' if r['confidence']<C.UNCERTAIN_THRESHOLD else ''}</div>"
                f"<div class='muted'>clip {ev['clip']} · dino {ev['dino']} · geom {ev['geometry']}</div>"
                f"<div class='bar'><span style='width:{int(r['confidence']*100)}%'></span></div></div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    with open(os.path.join(WORK, "x2_report.html"), "w") as f:
        f.write("".join(parts))


if __name__ == "__main__":
    main()
