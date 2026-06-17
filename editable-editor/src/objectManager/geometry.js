/**
 * Geometry primitives. A bbox is { x, y, w, h } in NATIVE image pixels.
 * Phase 0 operates at the bounding-box level; pixel/alpha refinement
 * (isOpaqueAt) is supplied by React in Phase 2.
 */

export function area(b) {
  return Math.max(0, b.w) * Math.max(0, b.h);
}

export function intersectionArea(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return ix * iy;
}

/** Intersection-over-union of two boxes (0..1). */
export function iou(a, b) {
  const inter = intersectionArea(a, b);
  const union = area(a) + area(b) - inter;
  return union <= 0 ? 0 : inter / union;
}

/** Fraction of `inner`'s area that lies inside `outer` (0..1). */
export function containment(inner, outer) {
  const a = area(inner);
  return a <= 0 ? 0 : intersectionArea(inner, outer) / a;
}

/** Is the point (px, py) inside the box? */
export function contains(b, px, py) {
  return px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h;
}

export function center(b) {
  return { x: b.x + b.w / 2, y: b.y + b.h / 2 };
}

/** Smallest box covering all given boxes. */
export function unionBox(boxes) {
  if (!boxes.length) return { x: 0, y: 0, w: 0, h: 0 };
  let x1 = Infinity,
    y1 = Infinity,
    x2 = -Infinity,
    y2 = -Infinity;
  for (const b of boxes) {
    x1 = Math.min(x1, b.x);
    y1 = Math.min(y1, b.y);
    x2 = Math.max(x2, b.x + b.w);
    y2 = Math.max(y2, b.y + b.h);
  }
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}
