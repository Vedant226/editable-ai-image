"""
Semantic Fusion Labeling Engine.

Combines independent evidence — CLIP zero-shot, GroundingDINO label, and
geometry/position — into one calibrated decision per proposal:
  { category, confidence, importance, editable, kind, evidence{clip,dino,geometry} }

DINO is ADDITIVE-only: a DINO match boosts its own category but can never lower
the winner's confidence (so a wrong/loose DINO box can't penalise a correct CLIP
call). Decorative subcategories (ornament/embroidery/border/emblem/crest/symbol)
are kept distinct — never collapsed into a generic "decoration". Parent context
is accepted (for X3) but contributes nothing until hierarchy exists.
"""

from . import config as C

W_CLIP, W_GEOM = 0.70, 0.30  # base evidence (sum to 1)
DINO_BONUS = 0.25            # max additive boost a DINO match gives its category
AGREE_BONUS = 0.05          # extra when DINO independently agrees with the winner

PARENT_PRIORS = {
    "person": ("face", "head", "hair", "crown", "hat", "clothing", "robe", "collar",
               "cape", "armor", "sword", "weapon", "jewelry", "hand", "eye"),
}


def map_phrase_to_category(phrase):
    if not phrase:
        return None
    low = phrase.lower().replace(",", " ").replace("/", " ")
    toks = low.split()
    for t in toks:
        if t in C.TAXONOMY:
            return t
    for t in toks:
        if t in C.SYNONYMS:
            return C.SYNONYMS[t]
    for k, v in C.SYNONYMS.items():
        if k in low:
            return v
    for t in toks:
        for cat in C.TAXONOMY:
            if t in cat or cat in t:
                return cat
    return None


def fuse(clip_probs, dino_cat, dino_strength, gfeat, gscores, parent_category=None):
    base = {
        c: W_CLIP * clip_probs.get(c, 0.0) + W_GEOM * gscores.get(c, 0.4)
        for c in C.TAXONOMY
    }

    has_dino = bool(dino_cat) and dino_cat in C.TAXONOMY
    if has_dino:
        ds = max(0.0, min(1.0, dino_strength))
        base[dino_cat] = min(1.0, base[dino_cat] + DINO_BONUS * ds)  # additive only

    if parent_category:
        for c in PARENT_PRIORS.get(parent_category, ()):
            base[c] = min(1.0, base[c] + 0.05)

    winner = max(base, key=base.get)
    conf = min(1.0, base[winner])
    if has_dino and dino_cat == winner:
        conf = min(1.0, conf + AGREE_BONUS)

    af = gfeat["areaFraction"]
    centrality = 1.0 - min(1.0, (((gfeat["cx"] - 0.5) ** 2 + (gfeat["cy"] - 0.5) ** 2) ** 0.5) / 0.7)
    size_norm = min(1.0, af / 0.08)
    cat_w = C.IMPORTANCE_WEIGHTS.get(winner, 0.6)
    importance = cat_w * (0.45 + 0.30 * size_norm + 0.25 * centrality) * (0.55 + 0.45 * conf)

    editable = (
        winner not in C.IGNORE_CATEGORIES
        and C.EDITABLE_MIN_AREA_FRAC <= af <= C.EDITABLE_MAX_AREA_FRAC
        and gfeat["solidity"] > 0.12
    )

    return {
        "category": winner,
        "confidence": round(conf, 3),
        "importance": round(min(1.0, importance), 3),
        "editable": bool(editable),
        "kind": "bitmap_text" if winner == "text" else "object",
        "evidence": {
            "clip": round(clip_probs.get(winner, 0.0), 3),
            "dino": round(max(0.0, min(1.0, dino_strength)), 3) if (has_dino and dino_cat == winner) else None,
            "geometry": round(gscores.get(winner, 0.4), 3),
        },
    }
