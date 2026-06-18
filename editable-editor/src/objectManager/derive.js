/**
 * Derivation pipeline: raw metadata -> ManagedObject[] (+ Groups).
 *
 * Pure and deterministic. Produces TIER 2 entirely in memory; the raw input
 * is never mutated and no derivative is written to disk. Stages:
 *   1. classify           — semantic category/role/specificity/editability
 *   2. text preference     — OCR-backed text wins; DINO text crops demoted
 *   3. soft demotion       — a container-dominated, oversized "person" -> ignored
 *   4. duplicate collapse  — same-category boxes with IoU > t -> one logical object
 *   5. grouping            — parts (face/crown/...) become children of a person
 *
 * Ignored objects are RETAINED and promotable; nothing is discarded.
 */

import { ROLE, DEFAULT_CONFIG } from "./vocabulary.js";
import { classify, isContainerDominated } from "./classify.js";
import { area, iou, containment, unionBox } from "./geometry.js";

const num = (v) => Number(v) || 0;

function reasonFromRole(role) {
  if (role === ROLE.CONTAINER) return "container";
  if (role === ROLE.SCENE) return "scene";
  return "unrecognized";
}

export function deriveModel(rawObjects, options = {}) {
  const config = { ...DEFAULT_CONFIG, ...(options.config || {}) };
  const objects = new Map();
  const order = [];
  const stats = { demotions: [], collapsed: [], groups: [] };

  // ---- 1. build base ManagedObjects + classify ----
  for (const raw of rawObjects) {
    if (!raw || raw.id == null || !raw.file) continue; // defensive: skip malformed entries
    const id = raw.id;
    const bbox = { x: num(raw.x), y: num(raw.y), w: num(raw.width), h: num(raw.height) };
    const c = classify(raw.type);
    const hasString = !!(raw.text && String(raw.text).trim());
    const isOcrText = c.role === ROLE.TEXT && (hasString || /^text_/i.test(raw.file || ""));

    const obj = {
      id,
      file: raw.file,
      rawType: raw.type,
      text: hasString ? String(raw.text) : "",
      isOcrText,
      // typography passthrough for text objects (so an activated text node
      // renders with its original styling); null for non-text.
      style:
        c.role === ROLE.TEXT
          ? {
              fontFamily: raw.fontFamily,
              fontWeight: raw.fontWeight,
              fontColor: raw.fontColor,
              strokeColor: raw.strokeColor,
              strokeWidth: raw.strokeWidth,
              textAlign: raw.textAlign,
              letterSpacing: raw.letterSpacing,
              shadowBlur: raw.shadowBlur,
            }
          : null,
      bbox,
      area: area(bbox),
      areaFraction: 0, // filled once image size known
      rotation: num(raw.rotation),
      zIndex: num(raw.zIndex),
      category: c.category,
      role: c.role,
      specificity: c.specificity,
      tokens: c.tokens,
      matched: c.matched,
      editable: c.editable,
      ignoredReason: c.editable ? null : reasonFromRole(c.role),
      parentId: null,
      childIds: [],
      groupId: null,
      aliasOf: null,
      duplicateIds: [],
    };
    objects.set(id, obj);
    order.push(id);
  }

  // ---- image size: prefer the background object, else union of all boxes ----
  let imageWidth = options.imageWidth;
  let imageHeight = options.imageHeight;
  if (!imageWidth || !imageHeight) {
    const bg = [...objects.values()].find((o) => o.category === "background");
    if (bg) {
      imageWidth = bg.bbox.w;
      imageHeight = bg.bbox.h;
    } else {
      const u = unionBox([...objects.values()].map((o) => o.bbox));
      imageWidth = u.x + u.w;
      imageHeight = u.y + u.h;
    }
  }
  const imageArea = Math.max(1, imageWidth * imageHeight);
  for (const o of objects.values()) o.areaFraction = o.area / imageArea;

  // ---- 2. text preference: OCR-backed text only; demote stringless DINO text ----
  for (const o of objects.values()) {
    if (o.role === ROLE.TEXT && !o.isOcrText) {
      o.editable = false;
      o.ignoredReason = "text-no-string";
    }
  }

  // ---- 3. demote container-dominated "persons" (framed-portrait panels) ----
  //   Purely semantic: a parent whose label is dominated by frame/portrait/scene
  //   tokens is structure, not a person — regardless of its size. A real person
  //   with a single stray structural token is safe (needs struct > edit).
  for (const o of objects.values()) {
    if (o.editable && o.role === ROLE.PARENT && isContainerDominated(o.matched)) {
      o.editable = false;
      o.ignoredReason = "demoted-container";
      stats.demotions.push({ id: o.id, type: o.rawType, areaFraction: o.areaFraction });
    }
  }

  // ---- 4. near-duplicate collapse (same category, IoU > threshold) ----
  const byCategory = new Map();
  for (const o of objects.values()) {
    if (!o.editable) continue;
    if (!byCategory.has(o.category)) byCategory.set(o.category, []);
    byCategory.get(o.category).push(o);
  }

  for (const group of byCategory.values()) {
    if (group.length < 2) continue;
    // union-find over the category
    const parent = new Map(group.map((o) => [o.id, o.id]));
    const find = (x) => {
      while (parent.get(x) !== x) {
        parent.set(x, parent.get(parent.get(x)));
        x = parent.get(x);
      }
      return x;
    };
    const union = (a, b) => parent.set(find(a), find(b));
    for (let i = 0; i < group.length; i++) {
      for (let j = i + 1; j < group.length; j++) {
        if (iou(group[i].bbox, group[j].bbox) > config.duplicateIoU) {
          union(group[i].id, group[j].id);
        }
      }
    }
    // gather clusters
    const clusters = new Map();
    for (const o of group) {
      const root = find(o.id);
      if (!clusters.has(root)) clusters.set(root, []);
      clusters.get(root).push(o);
    }
    for (const members of clusters.values()) {
      if (members.length < 2) continue;
      // canonical = most specific, then largest, then lowest id (deterministic)
      members.sort(
        (a, b) =>
          b.specificity - a.specificity || b.area - a.area || a.id - b.id
      );
      const canonical = members[0];
      const aliases = members.slice(1);
      canonical.duplicateIds = aliases.map((a) => a.id);
      for (const al of aliases) {
        al.editable = false;
        al.aliasOf = canonical.id;
        al.ignoredReason = `duplicate-of:${canonical.id}`;
      }
      stats.collapsed.push({
        canonicalId: canonical.id,
        category: canonical.category,
        aliasIds: canonical.duplicateIds,
      });
    }
  }

  // ---- 5. grouping: parts -> children of best-containing person ----
  const canonicalEditable = [...objects.values()].filter((o) => o.editable && !o.aliasOf);
  const parents = canonicalEditable.filter((o) => o.role === ROLE.PARENT);
  const parts = canonicalEditable.filter((o) => o.role === ROLE.PART);

  for (const part of parts) {
    let best = null;
    let bestScore = config.childContainment;
    for (const p of parents) {
      const score = containment(part.bbox, p.bbox);
      if (score > bestScore) {
        bestScore = score;
        best = p;
      }
    }
    if (best) {
      part.parentId = best.id;
      best.childIds.push(part.id);
    }
  }

  const groups = new Map();
  // parent-rooted groups
  for (const p of parents) {
    const memberIds = [p.id, ...p.childIds];
    const gid = `g-${p.id}`;
    const bbox = unionBox(memberIds.map((id) => objects.get(id).bbox));
    groups.set(gid, { id: gid, parentId: p.id, memberIds, bbox, type: p.category });
    for (const id of memberIds) objects.get(id).groupId = gid;
  }
  // standalone editable objects (decor, parentless parts, text) -> singleton groups
  for (const o of canonicalEditable) {
    if (o.groupId) continue;
    const gid = `g-${o.id}`;
    groups.set(gid, { id: gid, parentId: o.id, memberIds: [o.id], bbox: o.bbox, type: o.category });
    o.groupId = gid;
  }

  for (const g of groups.values()) {
    if (g.memberIds.length > 1) {
      stats.groups.push({ groupId: g.id, type: g.type, memberIds: g.memberIds });
    }
  }

  return { objects, groups, order, imageSize: { width: imageWidth, height: imageHeight }, config, stats };
}
