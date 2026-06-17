/**
 * Visual Object Resolver (VOR) — core (Phase 2).
 *
 * Maps a logical object (from the Object Manager) to a VisualObject: the
 * concrete plan for what pixels represent it on screen. Pure planner — React
 * executes it (loads images, draws, rasterizes). One-way dependency on the OM.
 *
 * Phase 2 covers SINGLE mode (one PNG per object) and text (dynamic). The
 * pixel-ownership partition is trivially satisfied here: inactive objects draw
 * nothing (the original base owns their pixels), and two active objects in the
 * same parent/child hierarchy are mutually exclusive (enforced in OM.activate).
 * Knockout compositing, inpaint-mask geometry and face-replace arrive later.
 */

import { ROLE } from "../objectManager/vocabulary.js";

export function createVisualResolver(om) {
  const defaultTransform = (o) => ({
    x: o.bbox.x,
    y: o.bbox.y,
    width: o.bbox.w,
    height: o.bbox.h,
    rotation: o.rotation,
  });

  /** Resolve one logical object → VisualObject. */
  const resolve = (objectId, session) => {
    const o = om.getObject(objectId);
    if (!o) return null;

    const entry = session?.entries?.[objectId];
    const isText = o.role === ROLE.TEXT;

    return {
      objectId,
      mode: isText ? "dynamic" : "single",
      isText,
      category: o.category,
      file: o.file,
      text: entry?.text ?? o.text,
      style: o.style || null,
      bbox: { ...o.bbox },
      transform: entry?.transform || defaultTransform(o),
      caps: {
        textEditable: isText,
        deletable: true,
        exportable: true,
        // face-replace targets (used by a later phase); empty action surface for now
        faceReplaceTargets: om
          .getChildren(objectId)
          .filter((c) => c.category === "face")
          .map((c) => c.id),
      },
    };
  };

  /** Resolve the whole scene: base + repair patches + active VisualObjects. */
  const resolveScene = (session) => {
    const activeVisuals = [];
    const repairs = [];
    for (const entry of Object.values(session.entries)) {
      const repairReady = entry.repair?.status === "ready" && entry.repair.dataUrl;
      const repairPatch = repairReady
        ? { objectId: entry.objectId, dataUrl: entry.repair.dataUrl, bbox: entry.repair.bbox }
        : null;

      // A deleted object renders ONLY its repair patch: the footprint stays
      // filled with clean background, so the object reads as removed.
      if (entry.deleted) {
        if (repairPatch) repairs.push(repairPatch);
        continue;
      }
      if (entry.state !== "active") continue;

      const v = resolve(entry.objectId, session);
      if (v) {
        v.z = entry.z || 0;
        activeVisuals.push(v);
      }
      // The patch covers the object's original footprint, so moving the lifted
      // object no longer reveals the original beneath it.
      if (repairPatch) repairs.push(repairPatch);
    }

    activeVisuals.sort((a, b) => (a.z || 0) - (b.z || 0)); // low z = back
    const { base } = om.getRenderModel(session);
    return { base, repairs, activeVisuals };
  };

  return { resolve, resolveScene };
}
