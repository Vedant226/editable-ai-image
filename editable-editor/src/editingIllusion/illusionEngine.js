/**
 * Editing Illusion Engine — core (Phase 7).
 *
 * Framework-agnostic state machine that owns the *perceptual transition* of an
 * object between "part of the original image" and "floating editable object".
 * It performs no I/O and decodes no pixels: it consumes intents + asset-status
 * + a clock, and emits a per-object `ContinuityFrame` describing HOW to render
 * this instant (occlusion, fill source/opacity, cutout opacity + lift offset,
 * shadow, selection glow). React drives the clock and executes the frame; the
 * backend produces the actual cutout/fill/shadow pixels.
 *
 * It encodes the invariants:
 *   I2 identity-at-rest  — RESTING shows the base, never the cutout.
 *   I3 no premature displacement — never leaves RESTING into a lifted phase
 *      unless the footprint can be covered (assets ready OR a temp fill exists).
 *   I4 single instance   — once lifted, the base footprint is occluded by the
 *      fill while the cutout floats; the two never both "show" the object.
 *   I6 masked swaps      — every base↔float swap rides a lift motion or a
 *      cross-fade; opacities/offsets here are what mask it.
 */

export const PHASE = Object.freeze({
  RESTING: "resting", // object is still part of the base image
  LIFTING: "lifting", // motion-masked swap base -> cutout+fill (rising)
  FLOATING: "floating", // elevated, following the cursor
  SETTLING: "settling", // easing back down onto the canvas
  PLACED: "placed", // grounded lifted object, idle
  DELETING: "deleting", // cutout fading out; fill remains (erased)
});

const DEFAULTS = Object.freeze({
  glowMs: 150,
  liftMs: 180,
  settleMs: 160,
  fillFadeMs: 200, // temp -> lama cross-fade
  deleteMs: 180,
  liftPx: 6, // native-px rise while elevated
  liftScale: 1.02,
  shadowOpacity: 0.4,
  placedShadow: 0.7, // shadow opacity multiplier when grounded
  tempFillAvailable: true, // can React always produce an instant temp fill?
});

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const easeOut = (t) => 1 - Math.pow(1 - clamp01(t), 3);

export function createEditingIllusionEngine(config = {}) {
  const C = { ...DEFAULTS, ...config };
  const lifts = new Map(); // id -> state
  let selectedId = null;

  const ensure = (id, now) => {
    let l = lifts.get(id);
    if (!l) {
      l = {
        phase: PHASE.RESTING,
        since: now,
        selected: false,
        selectedSince: now,
        deselectedSince: now,
        assets: "none", // none | pending | ready | failed
        fill: "none", // none | temp | lama
        fillSwapSince: null, // when a temp->lama swap began (for cross-fade)
      };
      lifts.set(id, l);
    }
    return l;
  };

  const setPhase = (l, phase, now) => {
    l.phase = phase;
    l.since = now;
  };

  const coverable = (l) => l.assets === "ready" || C.tempFillAvailable;

  function fillDirective(l, now) {
    const cf = l.fillSwapSince != null ? clamp01((now - l.fillSwapSince) / C.fillFadeMs) : 1;
    return {
      visible: true,
      source: l.fill,
      opacity: 1,
      crossfadeFrom: l.fillSwapSince != null && cf < 1 ? "temp" : null,
      crossfade: cf,
    };
  }

  function computeFrame(id, now) {
    const l = lifts.get(id);
    if (!l) return null;

    const glow = l.selected
      ? easeOut((now - l.selectedSince) / C.glowMs)
      : clamp01(1 - (now - l.deselectedSince) / C.glowMs);

    const rest = {
      objectId: id,
      phase: l.phase,
      selection: { glow },
      showBaseAtFootprint: true,
      cutout: { visible: false, opacity: 0, dy: 0, scale: 1 },
      fill: { visible: false, source: null, opacity: 0, crossfadeFrom: null, crossfade: 1 },
      shadow: { visible: false, opacity: 0, growth: 0 },
      done: false,
    };

    if (l.phase === PHASE.RESTING) return rest;

    const lifted = { ...rest, showBaseAtFootprint: false, fill: fillDirective(l, now) };

    if (l.phase === PHASE.LIFTING) {
      const t = easeOut((now - l.since) / C.liftMs);
      return {
        ...lifted,
        cutout: { visible: true, opacity: 1, dy: -C.liftPx * t, scale: 1 + (C.liftScale - 1) * t },
        shadow: { visible: true, opacity: C.shadowOpacity * t, growth: t },
      };
    }
    if (l.phase === PHASE.FLOATING) {
      return {
        ...lifted,
        cutout: { visible: true, opacity: 1, dy: -C.liftPx, scale: C.liftScale },
        shadow: { visible: true, opacity: C.shadowOpacity, growth: 1 },
      };
    }
    if (l.phase === PHASE.SETTLING) {
      const t = easeOut((now - l.since) / C.settleMs);
      return {
        ...lifted,
        cutout: { visible: true, opacity: 1, dy: -C.liftPx * (1 - t), scale: C.liftScale - (C.liftScale - 1) * t },
        shadow: { visible: true, opacity: C.shadowOpacity * (1 - 0.3 * t), growth: 1 - 0.5 * t },
      };
    }
    if (l.phase === PHASE.PLACED) {
      return {
        ...lifted,
        cutout: { visible: true, opacity: 1, dy: 0, scale: 1 },
        shadow: { visible: true, opacity: C.shadowOpacity * C.placedShadow, growth: 0.5 },
      };
    }
    if (l.phase === PHASE.DELETING) {
      const t = clamp01((now - l.since) / C.deleteMs);
      return {
        ...lifted,
        cutout: { visible: true, opacity: 1 - t, dy: 0, scale: 1 },
        shadow: { visible: true, opacity: C.shadowOpacity * (1 - t), growth: 0.5 },
        done: t >= 1,
      };
    }
    return rest;
  }

  return {
    PHASE,
    config: C,
    has: (id) => lifts.has(id),
    getState: (id) => lifts.get(id) || null,
    selectedId: () => selectedId,

    /** Select an object (single-selection). Creates a RESTING lift; React should prefetch its assets. */
    select(id, now) {
      if (selectedId != null && selectedId !== id) {
        const prev = lifts.get(selectedId);
        if (prev && prev.selected) {
          prev.selected = false;
          prev.deselectedSince = now;
        }
      }
      const l = ensure(id, now);
      if (!l.selected) {
        l.selected = true;
        l.selectedSince = now;
      }
      selectedId = id;
    },

    deselect(now) {
      if (selectedId != null) {
        const prev = lifts.get(selectedId);
        if (prev && prev.selected) {
          prev.selected = false;
          prev.deselectedSince = now;
        }
      }
      selectedId = null;
    },

    setAssets(id, status, now) {
      const l = ensure(id, now);
      // if already lifted on a temp fill and the real fill just arrived, cross-fade it in
      if (status === "ready" && l.fill === "temp" && l.phase !== PHASE.RESTING) {
        l.fill = "lama";
        l.fillSwapSince = now;
      }
      l.assets = status;
    },

    /** True if we may displace this object (I3): real fill ready, or a temp fill exists. */
    canLift(id) {
      const l = lifts.get(id);
      return coverable(l || { assets: "none" });
    },

    /** Begin the lift. Returns false (and stays RESTING) if the footprint can't be covered. */
    beginLift(id, now) {
      const l = ensure(id, now);
      if (l.phase === PHASE.LIFTING || l.phase === PHASE.FLOATING) return true;
      if (!coverable(l)) return false; // degradation: never displace over a hole
      l.fill = l.assets === "ready" ? "lama" : "temp";
      l.fillSwapSince = null;
      l.selected = true;
      setPhase(l, PHASE.LIFTING, now);
      selectedId = id;
      return true;
    },

    drop(id, now) {
      const l = lifts.get(id);
      if (l && l.phase === PHASE.FLOATING) setPhase(l, PHASE.SETTLING, now);
    },

    delete(id, now) {
      const l = ensure(id, now);
      setPhase(l, PHASE.DELETING, now);
    },

    remove(id) {
      lifts.delete(id);
      if (selectedId === id) selectedId = null;
    },

    clear() {
      lifts.clear();
      selectedId = null;
    },

    /** Advance time-based transitions (call each animation frame). */
    tick(now) {
      for (const l of lifts.values()) {
        if (l.phase === PHASE.LIFTING && now - l.since >= C.liftMs) setPhase(l, PHASE.FLOATING, now);
        else if (l.phase === PHASE.SETTLING && now - l.since >= C.settleMs) setPhase(l, PHASE.PLACED, now);
      }
    },

    ids: () => [...lifts.keys()],

    /** Create a lift already in PLACED (used to reconcile objects that appear
     *  active without a fresh lift, e.g. after undo/redo). No animation. */
    ensurePlaced(id, now) {
      let l = lifts.get(id);
      if (!l) {
        l = {
          phase: PHASE.PLACED,
          since: now,
          selected: false,
          selectedSince: now - C.glowMs,
          deselectedSince: now - C.glowMs,
          assets: "ready",
          fill: "lama",
          fillSwapSince: null,
        };
        lifts.set(id, l);
      }
      return l;
    },

    /** True while any transition or glow/cross-fade is mid-flight (drives RAF). */
    isAnimating(now) {
      for (const l of lifts.values()) {
        if (l.phase === PHASE.LIFTING || l.phase === PHASE.SETTLING || l.phase === PHASE.DELETING) return true;
        if (l.selected && now - l.selectedSince < C.glowMs) return true;
        if (!l.selected && now - l.deselectedSince < C.glowMs) return true;
        if (l.fillSwapSince != null && now - l.fillSwapSince < C.fillFadeMs) return true;
      }
      return false;
    },

    frame: computeFrame,
    frames(now) {
      const out = [];
      for (const id of lifts.keys()) out.push(computeFrame(id, now));
      return out;
    },
  };
}
