"""
Phase X3 — duplicate suppression (mask-NMS) + semantic hierarchy.

Dedup: greedy mask-IoU NMS. Two proposals whose masks overlap by IoU > tau are
the SAME region (whether same category or a relabel) -> keep the higher-quality
one, alias the other (retained, flagged, never discarded).

Hierarchy: a part (face/crown/clothing/sword/...) becomes a child of the
smallest person/figure that CONTAINS it. Containment (small-inside-large) is
hierarchy, not duplication, so both survive.
"""

import numpy as np

PARENT_CATS = {"person"}
PART_CATS = {
    "face", "head", "hair", "crown", "hat", "helmet", "clothing", "robe", "collar",
    "cape", "armor", "sword", "weapon", "shield", "staff", "jewelry", "hand", "eye", "badge",
}


def bbox_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def mask_iou(mi, mj):
    inter = int(np.logical_and(mi, mj).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(mi, mj).sum())
    return inter / union if union else 0.0


def containment(inner, outer):
    a = int(inner.sum())
    if a == 0:
        return 0.0
    return int(np.logical_and(inner, outer).sum()) / a


def quality(o):
    return (0.5 if o["editable"] else 0.0) + 0.6 * o["confidence"] + 0.4 * o["importance"]


def dedup(objs, masks, tau=0.80):
    """Returns (canonical_ids:set, alias:{dup_id -> canonical_id})."""
    order = sorted(range(len(objs)), key=lambda i: -quality(objs[i]))
    alive, alias = [], {}
    for i in order:
        oid = objs[i]["id"]
        dup_of = None
        for j in alive:
            jid = objs[j]["id"]
            if not bbox_overlap(objs[i]["bbox"], objs[j]["bbox"]):
                continue
            if mask_iou(masks[oid], masks[jid]) > tau:
                dup_of = jid
                break
        if dup_of is None:
            alive.append(i)
        else:
            alias[oid] = dup_of
    return {objs[i]["id"] for i in alive}, alias


def build_hierarchy(objs_by_id, canonical_ids, masks, contain_thresh=0.60):
    canon = [objs_by_id[i] for i in canonical_ids]
    parents = [o for o in canon if o["category"] in PARENT_CATS]
    areas = {o["id"]: int(masks[o["id"]].sum()) for o in canon}
    for o in canon:
        o.setdefault("parent", None)
        o.setdefault("children", [])
    for o in canon:
        if o["category"] not in PART_CATS:
            continue
        best, best_area = None, None
        for p in parents:
            if p["id"] == o["id"] or not bbox_overlap(o["bbox"], p["bbox"]):
                continue
            if containment(masks[o["id"]], masks[p["id"]]) > contain_thresh:
                if best is None or areas[p["id"]] < best_area:
                    best, best_area = p, areas[p["id"]]
        if best is not None:
            o["parent"] = best["id"]
            best["children"].append(o["id"])
    return parents
