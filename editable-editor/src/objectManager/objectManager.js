/**
 * Object Manager — the public, framework-agnostic API.
 *
 * Owns object IDENTITY, relationships, selection and the session lifecycle.
 * Pure: never mutates raw metadata, performs no I/O. React holds the session
 * (TIER 3) and calls these transitions, which return new session objects.
 *
 * Visual representation (which PNGs to draw) and inpaint-mask geometry are NOT
 * here — those belong to the Visual Object Resolver (Phase 2+).
 */

import { ROLE } from "./vocabulary.js";
import { contains } from "./geometry.js";
import { deriveModel } from "./derive.js";

// Initial transform = the object sitting exactly on its native footprint, so
// the lift is visually seamless. Geometry is width/height (not scale) to match
// the canvas's transform model.
const IDENTITY = (o) => ({ x: o.bbox.x, y: o.bbox.y, width: o.bbox.w, height: o.bbox.h, rotation: o.rotation });

export function createObjectManager(rawMetadata, options = {}) {
  const model = deriveModel(rawMetadata, options);
  const { objects, groups, order, imageSize } = model;

  const backgroundFile =
    [...objects.values()].find((o) => o.category === "background")?.file || "background.png";

  // ---------- helpers ----------
  const resolve = (id) => {
    const o = objects.get(id);
    return o && o.aliasOf != null ? objects.get(o.aliasOf) : o;
  };

  const editableCanonical = () =>
    order.map((id) => objects.get(id)).filter((o) => o.editable && o.aliasOf == null);

  // selection ranking: most specific, then smallest, then top-most, then id
  const rank = (a, b) =>
    b.specificity - a.specificity ||
    a.area - b.area ||
    b.zIndex - a.zIndex ||
    a.id - b.id;

  // ---------- queries ----------
  const api = {
    getImageSize: () => ({ ...imageSize }),
    getAllObjects: () => order.map((id) => objects.get(id)),
    getEditableObjects: () => editableCanonical(),
    getIgnoredObjects: () => order.map((id) => objects.get(id)).filter((o) => !o.editable),
    getObject: (id) => objects.get(id) || null,
    getGroup: (gid) => groups.get(gid) || null,
    getGroupOf: (id) => {
      const o = objects.get(id);
      return o && o.groupId ? groups.get(o.groupId) : null;
    },
    getChildren: (id) => (objects.get(id)?.childIds || []).map((c) => objects.get(c)),
    getParent: (id) => {
      const p = objects.get(id)?.parentId;
      return p != null ? objects.get(p) : null;
    },
    getDuplicates: (id) => (objects.get(id)?.duplicateIds || []).map((d) => objects.get(d)),

    // ---------- hit testing & selection ----------
    /** Editable objects whose bbox (and optional alpha) covers the point, ranked. */
    getHitCandidates: (point, isOpaqueAt) =>
      editableCanonical()
        .filter(
          (o) =>
            contains(o.bbox, point.x, point.y) && (!isOpaqueAt || isOpaqueAt(o.id, point))
        )
        .sort(rank),

    /**
     * Resolve a click to the best object. Re-clicking the current selection
     * escalates to its parent (leaf -> person).
     */
    pickAt: (point, { currentSelectionId = null, isOpaqueAt } = {}) => {
      const cands = api.getHitCandidates(point, isOpaqueAt);
      if (!cands.length) return null;
      const top = cands[0];
      if (currentSelectionId != null && currentSelectionId === top.id) {
        const up = api.escalateSelection(top.id);
        return up != null ? objects.get(up) : top;
      }
      return top;
    },

    /** One level up the hierarchy: a part -> its person; a person -> null. */
    escalateSelection: (id) => {
      const o = objects.get(id);
      if (!o) return null;
      return o.parentId != null ? o.parentId : null;
    },

    // ---------- session (pure transitions over TIER 3) ----------
    createSession: () => ({ entries: {}, selectedId: null }),

    /** Lift an object. Returns the new session + a repair request descriptor. */
    activate: (session, id) => {
      const obj = resolve(id);
      if (!obj || !obj.editable) return { session, repairRequest: null };
      // v1 rule: never lift an object together with its own parent or child
      // (knockout compositing is deferred). Deactivate conflicting relatives.
      const entries = { ...session.entries };
      for (const key of Object.keys(entries)) {
        const other = objects.get(Number(key));
        if (!other || other.id === obj.id) continue;
        if (other.parentId === obj.id || obj.parentId === other.id) delete entries[key];
      }
      const maxZ = Object.values(entries).reduce((m, e) => Math.max(m, e.z || 0), 0);
      const entry = entries[obj.id] || { objectId: obj.id, transform: IDENTITY(obj) };
      entries[obj.id] = { ...entry, state: "active", deleted: false, z: entry.z ?? maxZ + 1 };
      const next = { ...session, selectedId: obj.id, entries };
      // mask geometry is the Visual Object Resolver's job (Phase 3); bbox-level for now.
      const repairRequest = { objectId: obj.id, bbox: { ...obj.bbox }, reason: "activate-lift" };
      return { session: next, repairRequest };
    },

    deactivate: (session, id) => {
      const entries = { ...session.entries };
      delete entries[id];
      return { ...session, entries, selectedId: session.selectedId === id ? null : session.selectedId };
    },

    select: (session, id) => ({ ...session, selectedId: id }),

    applyTransform: (session, id, transform) => {
      const entry = session.entries[id];
      if (!entry) return session;
      return {
        ...session,
        entries: { ...session.entries, [id]: { ...entry, transform: { ...entry.transform, ...transform } } },
      };
    },

    setText: (session, id, text) => {
      const entry = session.entries[id] || { objectId: id, transform: IDENTITY(objects.get(id)) };
      return { ...session, entries: { ...session.entries, [id]: { ...entry, text } } };
    },

    softDelete: (session, id) => {
      const entry = session.entries[id] || { objectId: id, transform: IDENTITY(objects.get(id)) };
      return {
        ...session,
        entries: { ...session.entries, [id]: { ...entry, state: "deleted", deleted: true } },
      };
    },

    // Attach (or update) the inpaint repair patch for a lifted object.
    attachRepair: (session, id, repair) => {
      const entry = session.entries[id];
      if (!entry) return session;
      return {
        ...session,
        entries: { ...session.entries, [id]: { ...entry, repair } },
      };
    },

    bringToFront: (session, id) => {
      const entry = session.entries[id];
      if (!entry) return session;
      const maxZ = Object.values(session.entries).reduce((m, e) => Math.max(m, e.z || 0), 0);
      return { ...session, entries: { ...session.entries, [id]: { ...entry, z: maxZ + 1 } } };
    },

    sendToBack: (session, id) => {
      const entry = session.entries[id];
      if (!entry) return session;
      const minZ = Object.values(session.entries).reduce((m, e) => Math.min(m, e.z || 0), 0);
      return { ...session, entries: { ...session.entries, [id]: { ...entry, z: minZ - 1 } } };
    },

    // ---------- render model (logical; VOR resolves visuals in Phase 2) ----------
    getRenderModel: (session) => {
      const base = { file: backgroundFile, bbox: { x: 0, y: 0, w: imageSize.width, h: imageSize.height } };
      const activeObjects = [];
      const repairs = [];
      for (const entry of Object.values(session.entries)) {
        if (entry.deleted) continue;
        if (entry.state !== "active") continue;
        const o = objects.get(entry.objectId);
        if (!o) continue;
        activeObjects.push({
          id: o.id,
          file: o.file,
          bbox: o.bbox,
          transform: entry.transform,
          text: entry.text ?? o.text,
          category: o.category,
          role: o.role,
          isText: o.role === ROLE.TEXT,
        });
        if (entry.repair) repairs.push(entry.repair);
      }
      const hitProxies = editableCanonical().map((o) => ({ id: o.id, file: o.file, bbox: o.bbox }));
      return { base, repairs, activeObjects, hitProxies };
    },

    // ---------- diagnostics (used by the dev report) ----------
    getStats: () => {
      const all = api.getAllObjects();
      const ignored = all.filter((o) => !o.editable);
      const ignoredByReason = {};
      for (const o of ignored) {
        const key = o.ignoredReason?.startsWith("duplicate-of") ? "duplicate" : o.ignoredReason || "?";
        ignoredByReason[key] = (ignoredByReason[key] || 0) + 1;
      }
      const categoryCounts = {};
      for (const o of editableCanonical()) categoryCounts[o.category] = (categoryCounts[o.category] || 0) + 1;
      return {
        rawCount: all.length,
        editableCount: editableCanonical().length,
        ignoredCount: ignored.length,
        ignoredByReason,
        categoryCounts,
        demotions: model.stats.demotions,
        collapsed: model.stats.collapsed,
        groups: model.stats.groups,
      };
    },
  };

  return api;
}
