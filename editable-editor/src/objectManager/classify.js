/**
 * Semantic classification of a raw detection.
 *
 * The `type` string is a token bag (underscore/space joined). We pick the
 * single highest-specificity token that exists in the vocabulary; that token
 * decides category / role / editability. Size is NOT consulted here.
 */

import { VOCABULARY, MODIFIERS, ROLE } from "./vocabulary.js";

export function tokenize(type) {
  return String(type || "")
    .toLowerCase()
    .split(/[_\s/]+/)
    .filter(Boolean);
}

/**
 * @returns {{ category, role, specificity, editable, tokens, matched }}
 *   matched = every vocabulary hit (used by isContainerDominated for demotion).
 */
export function classify(rawType) {
  const tokens = tokenize(rawType);
  const matched = [];
  let best = null;

  for (const t of tokens) {
    if (MODIFIERS.has(t)) continue;
    const entry = VOCABULARY[t];
    if (!entry) continue;
    const hit = { token: t, ...entry };
    matched.push(hit);
    if (!best || hit.specificity > best.specificity) best = hit;
  }

  if (!best) {
    // Nothing recognized -> treat as inert scene, retained but not editable.
    return {
      category: "unknown",
      role: ROLE.SCENE,
      specificity: 0,
      editable: false,
      tokens,
      matched,
    };
  }

  return {
    category: best.category,
    role: best.role,
    specificity: best.specificity,
    editable: best.editable,
    tokens,
    matched,
  };
}

/**
 * True when an object's label is dominated by container/scene tokens.
 * Combined with a large area fraction, this is the ONLY place size enters —
 * to soft-demote a "person" that is really a portrait/frame panel. Size is
 * never the sole reason and the object is retained either way.
 */
export function isContainerDominated(matched) {
  let structural = 0;
  let editable = 0;
  for (const m of matched) {
    if (m.role === ROLE.CONTAINER || m.role === ROLE.SCENE) structural += 1;
    if (m.editable) editable += 1;
  }
  return structural > editable;
}
