/**
 * useAlphaHitTester — pixel-perfect hit predicate for the Object Manager.
 *
 * The OM ranks/selects logical objects (semantics, escalation) but stays pure;
 * it asks React "is object X opaque at this point?" via the callback this hook
 * returns. We decode each editable object's PNG into an offscreen canvas once
 * and sample its alpha channel. Until an image is decoded (or if decoding
 * fails) we fall back to the bounding box, so selection always works.
 */

import { useCallback, useEffect, useRef } from "react";

const ALPHA_THRESHOLD = 12; // treat alpha <= this as transparent (not a hit)

export function useAlphaHitTester(om) {
  // id -> { data, w, h } when decoded; undefined while loading; null on failure
  const cacheRef = useRef(new Map());

  useEffect(() => {
    if (!om) return;
    let cancelled = false;
    const cache = cacheRef.current;

    for (const o of om.getEditableObjects()) {
      if (cache.has(o.id)) continue;
      cache.set(o.id, undefined); // mark as loading

      const img = new window.Image();
      img.onload = () => {
        if (cancelled) return;
        try {
          const w = img.naturalWidth || o.bbox.w;
          const h = img.naturalHeight || o.bbox.h;
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext("2d", { willReadFrequently: true });
          ctx.drawImage(img, 0, 0, w, h);
          const { data } = ctx.getImageData(0, 0, w, h);
          cache.set(o.id, { data, w, h });
        } catch {
          cache.set(o.id, null); // tainted/failed -> bbox fallback
        }
      };
      img.onerror = () => {
        if (!cancelled) cache.set(o.id, null);
      };
      img.src = `/layers/${encodeURIComponent(o.file)}`;
    }

    return () => {
      cancelled = true;
    };
  }, [om]);

  return useCallback(
    (id, point) => {
      if (!om) return true;
      const o = om.getObject(id);
      if (!o) return false;

      const lx = point.x - o.bbox.x;
      const ly = point.y - o.bbox.y;
      if (lx < 0 || ly < 0 || lx >= o.bbox.w || ly >= o.bbox.h) return false;

      const entry = cacheRef.current.get(id);
      if (!entry) return true; // loading or failed -> bbox hit

      const px = Math.min(entry.w - 1, Math.floor((lx * entry.w) / o.bbox.w));
      const py = Math.min(entry.h - 1, Math.floor((ly * entry.h) / o.bbox.h));
      const alpha = entry.data[(py * entry.w + px) * 4 + 3];
      return alpha > ALPHA_THRESHOLD;
    },
    [om]
  );
}
