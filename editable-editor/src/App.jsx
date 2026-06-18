import React, {
  useEffect,
  useState,
  useRef,
  useMemo,
  useReducer,
  useCallback,
} from "react";

import { Stage, Layer, Image, Transformer, Text } from "react-konva";
import useImage from "use-image";

import { createObjectManager } from "./objectManager";
import { createVisualResolver } from "./visualResolver";
import { useAlphaHitTester } from "./useAlphaHitTester";
import { createEditingIllusionEngine, PHASE } from "./editingIllusion";

/* Decode an image URL/data-URL to a ready-to-paint HTMLImageElement (or null on
   failure). Used so lift bitmaps are decoded BEFORE a node depends on them — a
   node never has to mount against a not-yet-loaded image (which would blank a
   frame and, mid-drag, orphan the Konva node). */
function decodeImage(url) {
  return new Promise((resolve) => {
    if (!url) return resolve(null);
    const img = new window.Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

/* ==========================
   EDITABLE LAYER
   The node for the SELECTED and/or LIFTED object. While RESTING-selected it
   draws the object's ORIGINAL pixels in place at full opacity (visually
   identical to the base — selection never changes pixels) with a matte-hugging
   glow. It only becomes draggable/transformable once its footprint can be
   covered (`manipulable` = lift package ready); the actual lift (fill + float)
   begins on the FIRST drag/transform via `onLiftStart`, never on selection.
========================== */

function EditableLayer({ shapeProps, isSelected, isEditing, edited, textFade, manipulable, glow, opacity, cutoutImg, onChange, onStartTextEdit, onLiftStart }) {
  // The resting highlight draws the ORIGINAL /layers PNG (closest to the base);
  // the refined /lift cutout (decoded element, passed in) is used only once
  // lifted. Because both are READY HTMLImageElements, swapping between them
  // keeps the SAME Konva node mounted — a drag in progress is never orphaned.
  const [layerImage] = useImage(`/layers/${encodeURIComponent(shapeProps.file || "")}`);
  const image = cutoutImg || layerImage;

  const shapeRef = useRef(null); // the <Image> (bitmap / cutout)
  const textRef = useRef(null); // the synthesized <Text> (separate ref: unmounting the Image must not null it)
  const trRef = useRef(null);
  const isText = shapeProps.type?.toLowerCase().includes("text");
  // tf=0 → show original bitmap; tf=1 → show synthesized text; in between = crossfade
  const tf = isText ? (textFade ?? (edited ? 1 : 0)) : 0;
  const imageReady = !!image;
  // the node the transformer/handlers act on: the synth Text once editing has
  // taken over, otherwise the bitmap/cutout Image
  const activeNode = () => (isText && edited && textRef.current ? textRef.current : shapeRef.current);

  // Transformer is an affordance on the selected object, but only once the lift
  // package is ready (so acting on a handle covers the footprint, never a hole)
  // and only once a real node exists (imageReady) → never an empty transformer
  // box; re-attaches when the node appears or the Image↔Text node changes.
  useEffect(() => {
    const node = activeNode();
    if (isSelected && manipulable && !isEditing && imageReady && trRef.current && node) {
      trRef.current.nodes([node]);
      trRef.current.getLayer()?.batchDraw();
    }
  }, [isSelected, isEditing, manipulable, imageReady, edited]);

  const glowProps =
    glow > 0.01
      ? { shadowColor: "#d8b36a", shadowBlur: 6 + glow * 16, shadowOpacity: glow, shadowForStrokeEnabled: false }
      : {};

  const liftStart = () => onLiftStart?.(shapeProps.id);
  const updatePosition = (e) => onChange({ ...shapeProps, x: e.target.x(), y: e.target.y() });

  const handleTransform = () => {
    const node = activeNode();
    if (!node) return;
    const scaleX = node.scaleX();
    const scaleY = node.scaleY();
    node.scaleX(1);
    node.scaleY(1);
    onChange({
      ...shapeProps,
      x: node.x(),
      y: node.y(),
      rotation: node.rotation(),
      width: Math.max(10, node.width() * scaleX),
      height: Math.max(10, node.height() * scaleY),
    });
  };

  return (
    <>
      {/* image, OR text's ORIGINAL bitmap (identical to source; fades out as synth text fades in) */}
      {(!isText || tf < 1) && image && (
        <Image
          ref={shapeRef}
          image={image}
          visible={!(isText && isEditing)}
          x={shapeProps.x}
          y={shapeProps.y}
          width={shapeProps.width}
          height={shapeProps.height}
          rotation={shapeProps.rotation || 0}
          opacity={(opacity ?? 1) * (isText ? 1 - tf : 1)}
          draggable={manipulable}
          onDblClick={isText ? () => onStartTextEdit?.(shapeProps.id) : undefined}
          onDblTap={isText ? () => onStartTextEdit?.(shapeProps.id) : undefined}
          onDragStart={liftStart}
          onDragEnd={updatePosition}
          onTransformStart={liftStart}
          onTransformEnd={handleTransform}
          {...glowProps}
        />
      )}

      {/* once the user edits text, synthesize typography from the font estimation */}
      {isText && edited && (
        <Text
          ref={textRef}
          visible={!isEditing}
          text={shapeProps.text || "Edit me"}
          x={shapeProps.x}
          y={shapeProps.y}
          width={shapeProps.width}
          height={shapeProps.height}
          rotation={shapeProps.rotation || 0}
          opacity={(opacity ?? 1) * tf}
          fontFamily={shapeProps.fontFamily || "Cinzel"}
          fontStyle="bold"
          fontSize={Math.max(14, shapeProps.height * 0.55)}
          fill={shapeProps.fontColor || "#d8b36a"}
          stroke={shapeProps.strokeColor || "#5a2e12"}
          strokeWidth={shapeProps.strokeWidth || 1}
          align="center"
          draggable={manipulable}
          onDblClick={() => onStartTextEdit?.(shapeProps.id)}
          onDblTap={() => onStartTextEdit?.(shapeProps.id)}
          onDragStart={liftStart}
          onDragEnd={updatePosition}
          onTransformStart={liftStart}
          onTransformEnd={handleTransform}
          {...glowProps}
        />
      )}

      {/* transformer on the SELECTED object once its footprint can be covered
          (lift package ready) AND a real node exists (no empty box) */}
      {isSelected && manipulable && !isEditing && imageReady && <Transformer ref={trRef} rotateEnabled />}
    </>
  );
}

/* ==========================
   INLINE TEXT EDITOR (unchanged from Phase 4)
========================== */

function InlineTextEditor({ rect, fontSize, fontFamily, color, initialValue, onCommit, onCancel }) {
  const ref = useRef(null);
  const doneRef = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (el) {
      el.focus();
      el.select();
    }
  }, []);

  const finish = (commit) => {
    if (doneRef.current) return;
    doneRef.current = true;
    if (commit) onCommit(ref.current?.value ?? initialValue);
    else onCancel();
  };

  return (
    <textarea
      ref={ref}
      defaultValue={initialValue}
      spellCheck={false}
      onKeyDown={(e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          finish(true);
        } else if (e.key === "Escape") {
          e.preventDefault();
          finish(false);
        }
      }}
      onBlur={() => finish(true)}
      style={{
        position: "fixed",
        left: `${rect.left}px`,
        top: `${rect.top}px`,
        width: `${Math.max(40, rect.width)}px`,
        height: `${Math.max(rect.height, fontSize * 1.3)}px`,
        fontSize: `${fontSize}px`,
        fontFamily,
        fontWeight: "bold",
        color,
        textAlign: "center",
        lineHeight: 1.05,
        // translucent so the ORIGINAL bitmap (still in the base) stays visible
        // behind the edit field until the user confirms
        background: "rgba(10,10,10,0.35)",
        border: "1px solid #d8b36a",
        outline: "none",
        margin: 0,
        padding: 0,
        resize: "none",
        overflow: "hidden",
        transformOrigin: "top left",
        transform: rect.rotation ? `rotate(${rect.rotation}deg)` : "none",
        zIndex: 20,
      }}
    />
  );
}

/* ==========================
          APP
========================== */

const FALLBACK_IMAGE_SIZE = { width: 1408, height: 768 };

const chip = {
  background: "rgba(0,0,0,0.45)",
  color: "#d8b36a",
  font: "13px/1.4 monospace",
  padding: "6px 10px",
  borderRadius: 6,
};
const btn = (enabled = true) => ({
  padding: "7px 12px",
  background: enabled ? "#d8b36a" : "#5a5039",
  color: "#1a1a1a",
  border: "none",
  borderRadius: 6,
  fontWeight: 600,
  cursor: enabled ? "pointer" : "default",
  opacity: enabled ? 1 : 0.6,
});

export default function App() {
  const [rawMetadata, setRawMetadata] = useState(null);
  const [history, setHistory] = useState({ past: [], present: { entries: {}, selectedId: null }, future: [] });
  const session = history.present;

  const [viewport, setViewport] = useState({ w: window.innerWidth, h: window.innerHeight });
  const [liftEngine, setLiftEngine] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [, forceTick] = useReducer((c) => c + 1, 0);

  const [backgroundImage] = useImage("/layers/background.png");
  const stageRef = useRef(null);
  const backendReadyRef = useRef(false);
  const liftAssetsRef = useRef(new Map()); // id -> { cutout:{url}, fill:{url,x,y,w,h}, shadow, readyAt } | {pending|failed}
  const prevSelRef = useRef(null);
  const lastHoverRef = useRef(0);
  const editedAtRef = useRef(new Map()); // text id -> time of first edit (bitmap→text crossfade)
  const TEXT_FADE_MS = 260;

  // Editing Illusion Engine (stateful; created once)
  const eieRef = useRef(null);
  if (!eieRef.current) eieRef.current = createEditingIllusionEngine();
  const eie = eieRef.current;

  const clockRef = useRef(performance.now());
  const rafRunningRef = useRef(false);
  const kick = useCallback(() => {
    if (rafRunningRef.current) return;
    rafRunningRef.current = true;
    const step = () => {
      const now = performance.now();
      clockRef.current = now;
      eie.tick(now);
      forceTick();
      let textFading = false;
      for (const ea of editedAtRef.current.values()) {
        if (now - ea < TEXT_FADE_MS) {
          textFading = true;
          break;
        }
      }
      if (eie.isAnimating(now) || textFading) requestAnimationFrame(step);
      else rafRunningRef.current = false;
    };
    requestAnimationFrame(step);
  }, [eie]);

  // ---- Object Manager + Visual Resolver ----
  const om = useMemo(() => (rawMetadata ? createObjectManager(rawMetadata) : null), [rawMetadata]);
  const vr = useMemo(() => (om ? createVisualResolver(om) : null), [om]);
  const isOpaqueAt = useAlphaHitTester(om);

  useEffect(() => {
    fetch("/layers/metadata.json")
      .then((r) => r.json())
      .then((d) => setRawMetadata(Array.isArray(d) ? d : [])) // defensive: tolerate malformed metadata
      .catch((err) => {
        console.error("Failed to load metadata.json", err);
        setRawMetadata([]);
      });
  }, []);

  useEffect(() => {
    const onResize = () => setViewport({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((h) => {
        if (cancelled || !h) return;
        backendReadyRef.current = true;
        setLiftEngine(h.engine);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // ---- single coordinate space ----
  const imageSize = om ? om.getImageSize() : FALLBACK_IMAGE_SIZE;
  const scale = Math.min(viewport.w / imageSize.width, viewport.h / imageSize.height);
  const stageWidth = imageSize.width * scale;
  const stageHeight = imageSize.height * scale;

  // ---- history helpers ----
  const commit = useCallback((updater) => {
    setHistory((h) => {
      const next = updater(h.present);
      if (next === h.present) return h;
      return { past: [...h.past, h.present], present: next, future: [] };
    });
  }, []);
  const update = useCallback((updater) => {
    setHistory((h) => {
      const next = updater(h.present);
      if (next === h.present) return h;
      return { ...h, present: next };
    });
  }, []);
  const undo = useCallback(() => {
    setEditingId(null);
    setHistory((h) =>
      h.past.length
        ? { past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] }
        : h
    );
  }, []);
  const redo = useCallback(() => {
    setEditingId(null);
    setHistory((h) =>
      h.future.length ? { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) } : h
    );
  }, []);

  // ---- prefetch the Lift Package (cutout + fill + shadow) for an object ----
  const prefetchLift = useCallback(
    async (id) => {
      if (!backendReadyRef.current) return;
      const cached = liftAssetsRef.current.get(id);
      if (cached && cached.readyAt) {
        eie.setAssets(id, "ready", performance.now());
        return;
      }
      if (cached && cached.pending) return;
      liftAssetsRef.current.set(id, { pending: true });
      try {
        const res = await fetch("/api/lift", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ objectId: id }),
        });
        if (!res.ok) throw new Error(`lift ${res.status}`);
        const p = await res.json();
        // DECODE the bitmaps before exposing the object as ready. fillReady (and
        // hence draggable/transformable + the synth-text swap) must mean "can be
        // PAINTED this frame", not merely "URL known" — otherwise the footprint
        // fill / refined cutout blanks for a frame on first use (orphaned drag,
        // ghosted text). We hand React the decoded elements, so no async re-load.
        const [cutoutImg, fillImg, shadowImg] = await Promise.all([
          decodeImage(p.cutout?.png),
          decodeImage(p.fill?.png),
          decodeImage(p.shadow?.png),
        ]);
        liftAssetsRef.current.set(id, {
          cutout: cutoutImg ? { url: p.cutout.png, img: cutoutImg } : null,
          fill: fillImg
            ? { url: p.fill.png, img: fillImg, x: p.fill.x, y: p.fill.y, w: p.fill.w, h: p.fill.h }
            : null,
          shadow: shadowImg && p.shadow ? { ...p.shadow, img: shadowImg } : null,
          readyAt: performance.now(),
        });
        eie.setAssets(id, "ready", performance.now());
        kick();
      } catch (err) {
        console.warn("lift failed", err);
        liftAssetsRef.current.set(id, { failed: true });
        eie.setAssets(id, "failed", performance.now());
      }
    },
    [eie, kick]
  );

  // ---- selection (NEVER lifts; selection must not modify pixels) ----
  const handleStageClick = useCallback(
    (e) => {
      if (!om) return;
      const stage = e.target.getStage();
      if (!stage) return;
      const parent = e.target.getParent && e.target.getParent();
      if (parent && parent.className === "Transformer") return;
      const point = stage.getRelativePointerPosition();
      if (!point) return;

      const picked = om.pickAt(point, { currentSelectionId: session.selectedId, isOpaqueAt });
      // Clicking only SELECTS — the object stays part of the original image (no
      // activate, no fill, no cutout swap). Lifting happens on drag/transform.
      update((s) => om.select(s, picked ? picked.id : null));
      if (picked) prefetchLift(picked.id); // warm the lift package so a later drag is instant
    },
    [om, isOpaqueAt, session.selectedId, update, prefetchLift]
  );

  // Warm the lift package for the object under the cursor so the lift is instant
  // on click/drag (the footprint fill is ready before displacement is allowed).
  const handleStageHover = useCallback(
    (e) => {
      if (!om || !backendReadyRef.current) return;
      const t = performance.now();
      if (t - lastHoverRef.current < 100) return;
      lastHoverRef.current = t;
      const stage = e.target.getStage();
      const point = stage && stage.getRelativePointerPosition();
      if (!point) return;
      const picked = om.pickAt(point, { isOpaqueAt });
      if (picked) prefetchLift(picked.id);
    },
    [om, isOpaqueAt, prefetchLift]
  );

  // ---- lift: the SELECTED→LIFTING transition, triggered by the FIRST drag or
  // transform (never by selection). The footprint fill is already loaded (the
  // node is only manipulable once it is), so beginLift covers it before the
  // object can be displaced — no hole, no duplicate. `update` (not `commit`)
  // keeps the activation out of history; the committed drop is the undo step.
  const liftStart = useCallback(
    (id) => {
      if (id == null || !om) return;
      update((s) => {
        const sel = om.select(s, id);
        return s.entries[id]?.state === "active" ? sel : om.activate(sel, id).session;
      });
      eie.beginLift(id, performance.now());
      kick();
    },
    [om, update, eie, kick]
  );

  const handleObjectChange = useCallback(
    (id, attrs, prevText) => {
      commit((s) => {
        let next = om.applyTransform(s, id, {
          x: attrs.x,
          y: attrs.y,
          width: attrs.width,
          height: attrs.height,
          rotation: attrs.rotation,
        });
        if (typeof attrs.text === "string" && attrs.text !== prevText) next = om.setText(next, id, attrs.text);
        return next;
      });
      eie.drop(id, performance.now()); // FLOATING → SETTLING → PLACED
      kick();
    },
    [om, commit, eie, kick]
  );

  // ---- inline text editing ----
  const startTextEdit = useCallback((id) => setEditingId(id), []);
  const cancelText = useCallback(() => setEditingId(null), []);
  const commitText = useCallback(
    (value) => {
      const id = editingId;
      setEditingId(null);
      if (id == null || !om) return;
      const v = (value ?? "").trim();
      if (!v) return;
      // Confirming the edit lifts the text object so its footprint fill occludes
      // the ORIGINAL baked bitmap; only then does the bitmap cross-fade to the
      // synthesized typography (the render gates this on the fill being ready,
      // so the original never ghosts through). Selection alone never does this.
      prefetchLift(id);
      commit((s) => om.setText(om.activate(s, id).session, id, v));
      eie.beginLift(id, performance.now());
      kick();
    },
    [editingId, om, commit, eie, kick, prefetchLift]
  );

  // ---- toolbar actions ----
  const deleteSelected = useCallback(() => {
    const id = session.selectedId;
    if (id == null || !om) return;
    prefetchLift(id); // ensure a footprint fill exists so deletion never reveals a hole
    let tries = 0;
    const run = () => {
      if (liftAssetsRef.current.get(id)?.fill) {
        eie.delete(id, performance.now());
        kick();
        setTimeout(() => commit((s) => om.select(om.softDelete(s, id), null)), eie.config.deleteMs + 40);
      } else if (tries++ < 25 && backendReadyRef.current) {
        setTimeout(run, 120); // wait for the fill
      }
      // else (no backend / fill unavailable): leave the object in place rather than expose a hole
    };
    run();
  }, [session.selectedId, om, eie, kick, commit, prefetchLift]);

  // z-order acts on a lifted object; a resting selection is activated first so
  // the button is never a dead affordance (re-ordering doesn't displace the
  // object, so it needs no fill — identity at rest is preserved either way).
  const reorder = useCallback(
    (which) => {
      const id = session.selectedId;
      if (id == null || !om) return;
      commit((s) => {
        const base = s.entries[id]?.state === "active" ? s : om.activate(s, id).session;
        return which === "front" ? om.bringToFront(base, id) : om.sendToBack(base, id);
      });
    },
    [session.selectedId, om, commit]
  );
  const bringToFront = useCallback(() => reorder("front"), [reorder]);
  const sendToBack = useCallback(() => reorder("back"), [reorder]);

  // ---- keyboard ----
  useEffect(() => {
    const onKey = (e) => {
      if (editingId != null) return;
      const tag = (e.target && e.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if (meta && e.key.toLowerCase() === "y") {
        e.preventDefault();
        redo();
      } else if (e.key === "Delete" || e.key === "Backspace") {
        if (session.selectedId != null) {
          e.preventDefault();
          deleteSelected();
        }
      } else if (e.key === "Escape") {
        update((s) => (om ? om.select(s, null) : s));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editingId, session.selectedId, undo, redo, deleteSelected, update, om]);

  // ---- reconcile the illusion engine with the session (selection + lifts) ----
  useEffect(() => {
    if (!om) return;
    const now = performance.now();
    const active = new Set(
      Object.values(session.entries)
        .filter((e) => e.state === "active" && !e.deleted)
        .map((e) => e.objectId)
    );
    for (const id of active) if (!eie.has(id)) eie.ensurePlaced(id, now);
    for (const id of eie.ids()) {
      if (id === session.selectedId) continue; // keep the selected object's RESTING glow
      if (!active.has(id) && !session.entries[id]?.deleted) eie.remove(id);
    }
    if (prevSelRef.current !== session.selectedId) {
      if (session.selectedId != null) eie.select(session.selectedId, now);
      else eie.deselect(now);
      prevSelRef.current = session.selectedId;
    }
    kick();
  }, [session, om, eie, kick]);

  // ---- export at native resolution ----
  const handleExport = useCallback(() => {
    const stage = stageRef.current;
    if (!stage) return;
    update((s) => (om ? om.select(s, null) : s));
    requestAnimationFrame(() => {
      const uri = stage.toDataURL({ pixelRatio: 1 / scale });
      const link = document.createElement("a");
      link.download = "edited.png";
      link.href = uri;
      link.click();
    });
  }, [scale, om, update]);

  const scene = vr ? vr.resolveScene(session) : { activeVisuals: [] };
  const now = clockRef.current;
  const selected = session.selectedId != null && om ? om.getObject(session.selectedId) : null;
  const selectedVisual = selected && vr ? vr.resolve(selected.id, session) : null;
  const selectedLifted = selected ? !!liftAssetsRef.current.get(selected.id)?.fill : false;

  // Render the SELECTED object even when it has not been lifted, so it can show
  // its silhouette glow and be grabbed — drawn from its ORIGINAL pixels in place
  // (identical to the base), with no fill behind it (the base still owns those
  // pixels). It joins the lifted objects from the scene.
  const renderList = [...scene.activeVisuals];
  if (selected && vr) {
    const selEntry = session.entries[selected.id];
    if ((!selEntry || selEntry.state !== "active") && !selEntry?.deleted) {
      const sv = vr.resolve(selected.id, session);
      if (sv) renderList.push(sv);
    }
  }

  // fills: cover lifted footprints (active) and erased footprints (deleted)
  const fills = [];
  for (const v of scene.activeVisuals) {
    const a = liftAssetsRef.current.get(v.objectId);
    if (a && a.fill) fills.push({ id: v.objectId, ...a.fill });
  }
  for (const entry of Object.values(session.entries)) {
    if (entry.deleted) {
      const a = liftAssetsRef.current.get(entry.objectId);
      if (a && a.fill) fills.push({ id: entry.objectId, ...a.fill });
    }
  }

  // attached grounding shadows — invisible at rest (identity preserved), fade in
  // as an object is dragged off its footprint (the "lifted off the image" cue)
  const shadows = [];
  for (const v of scene.activeVisuals) {
    const a = liftAssetsRef.current.get(v.objectId);
    const o = om && om.getObject(v.objectId);
    if (a && a.shadow && a.shadow.img && o) {
      const dx = v.transform.x - o.bbox.x;
      const dy = v.transform.y - o.bbox.y;
      const op = (a.shadow.opacity ?? 0.4) * Math.min(1, Math.hypot(dx, dy) / 40);
      if (op > 0.01) {
        shadows.push({ id: v.objectId, img: a.shadow.img, x: a.shadow.x + dx, y: a.shadow.y + dy, w: a.shadow.w, h: a.shadow.h, opacity: op });
      }
    }
  }

  // inline editor screen rect
  let editor = null;
  if (editingId != null && om && stageRef.current) {
    const o = om.getObject(editingId);
    if (o) {
      const entry = session.entries[editingId];
      const t = entry?.transform || { x: o.bbox.x, y: o.bbox.y, width: o.bbox.w, height: o.bbox.h, rotation: o.rotation };
      const cont = stageRef.current.container().getBoundingClientRect();
      editor = {
        rect: {
          left: cont.left + t.x * scale,
          top: cont.top + t.y * scale,
          width: t.width * scale,
          height: t.height * scale,
          rotation: t.rotation || 0,
        },
        fontSize: Math.max(14, t.height * 0.55) * scale,
        fontFamily: o.style?.fontFamily || "Cinzel",
        color: o.style?.fontColor || "#d8b36a",
        initialValue: entry?.text ?? o.text,
      };
    }
  }

  return (
    <div
      style={{
        background: "#111",
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div style={{ ...chip, position: "absolute", top: 12, left: 12, zIndex: 10, pointerEvents: "none" }}>
        {selected ? `selected: ${selected.category}#${selected.id}` : "click an object to select it"}
        {selected ? (selectedLifted ? "  ·  drag to move" : "  ·  preparing…") : ""}
        {selected?.role === "text" ? "  ·  double-click to edit" : ""}
        {`  ·  lift: ${liftEngine || "offline"}`}
      </div>

      <div style={{ position: "absolute", top: 12, right: 12, zIndex: 10, display: "flex", gap: 8 }}>
        <button onClick={undo} disabled={!history.past.length} style={btn(history.past.length > 0)}>
          ↶ Undo
        </button>
        <button onClick={redo} disabled={!history.future.length} style={btn(history.future.length > 0)}>
          ↷ Redo
        </button>
        <button onClick={handleExport} style={btn(true)}>
          Export PNG
        </button>
      </div>

      {selected && !editingId && (
        <div
          style={{
            position: "absolute",
            bottom: 16,
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 10,
            display: "flex",
            gap: 8,
            padding: 8,
            background: "rgba(0,0,0,0.55)",
            borderRadius: 10,
          }}
        >
          {selectedVisual?.caps?.textEditable && (
            <button onClick={() => startTextEdit(selected.id)} style={btn(true)}>
              Edit text
            </button>
          )}
          <button onClick={bringToFront} style={btn(true)}>
            Bring to front
          </button>
          <button onClick={sendToBack} style={btn(true)}>
            Send to back
          </button>
          {selectedVisual?.caps?.deletable && (
            <button onClick={deleteSelected} style={{ ...btn(true), background: "#c0563f", color: "#fff" }}>
              Delete
            </button>
          )}
        </div>
      )}

      <Stage
        ref={stageRef}
        width={stageWidth}
        height={stageHeight}
        scaleX={scale}
        scaleY={scale}
        onClick={handleStageClick}
        onTap={handleStageClick}
        onMouseMove={handleStageHover}
      >
        <Layer>
          {/* BASE — the original image, the only visual source of truth */}
          <Image image={backgroundImage} x={0} y={0} width={imageSize.width} height={imageSize.height} listening={false} />

          {/* REPAIR FILLS — inpainted patches covering lifted/erased footprints
              (decoded elements, so a fill paints the SAME frame it is requested) */}
          {fills.map((f) => (
            <Image key={`fill-${f.id}`} image={f.img} x={f.x} y={f.y} width={f.w} height={f.h} listening={false} />
          ))}

          {/* GROUNDING SHADOWS — attached, fade in on drag */}
          {shadows.map((s) => (
            <Image key={`shadow-${s.id}`} image={s.img} x={s.x} y={s.y} width={s.w} height={s.h} opacity={s.opacity} listening={false} />
          ))}

          {/* SELECTED / LIFTED SMART OBJECTS */}
          {renderList.map((v) => {
            const t = v.transform;
            const fr = eie.frame(v.objectId, now);
            const a = liftAssetsRef.current.get(v.objectId);
            const entry = session.entries[v.objectId];
            const isActive = entry?.state === "active";
            const fillReady = !!(a && a.fill);
            // text becomes synthesized typography only once edited AND its
            // footprint fill is PAINTABLE (decoded) — else the original ghosts
            // through (fillReady = decoded element present, not just URL known)
            const editedFlag = entry?.text != null && (!v.isText || fillReady);
            // start the bitmap→text cross-fade exactly when the synth text first
            // shows, so it eases in; clear it when not shown so a later
            // (re)appearance (redo / re-edit) eases again rather than snapping
            if (editedFlag && !editedAtRef.current.has(v.objectId)) editedAtRef.current.set(v.objectId, now);
            else if (!editedFlag && editedAtRef.current.has(v.objectId)) editedAtRef.current.delete(v.objectId);
            const ea = editedAtRef.current.get(v.objectId);
            const textFade = editedFlag ? (ea ? Math.min(1, (now - ea) / TEXT_FADE_MS) : 1) : 0;
            // Opacity follows the engine only for a genuinely lifted phase; a
            // RESTING frame (selection highlight OR active-but-not-yet-lifted)
            // draws at full opacity (≡ the base). DELETING fades even on a never-
            // activated object (delete a resting selection → it still dissolves).
            const opacity = fr && fr.phase !== PHASE.RESTING ? fr.cutout.opacity : 1;
            const shapeProps = {
              id: v.objectId,
              file: v.file,
              type: v.isText ? "text" : v.category,
              text: v.text,
              x: t.x,
              y: t.y,
              width: t.width,
              height: t.height,
              rotation: t.rotation,
              fontFamily: v.style?.fontFamily,
              fontColor: v.style?.fontColor,
              strokeColor: v.style?.strokeColor,
              strokeWidth: v.style?.strokeWidth,
            };
            return (
              <EditableLayer
                key={v.objectId}
                shapeProps={shapeProps}
                isSelected={session.selectedId === v.objectId}
                isEditing={editingId === v.objectId}
                edited={editedFlag}
                textFade={textFade}
                manipulable={fillReady}
                glow={fr ? fr.selection.glow : session.selectedId === v.objectId ? 1 : 0}
                opacity={opacity}
                cutoutImg={!v.isText && isActive && a && a.cutout ? a.cutout.img : null}
                onChange={(attrs) => handleObjectChange(v.objectId, attrs, v.text)}
                onStartTextEdit={startTextEdit}
                onLiftStart={liftStart}
              />
            );
          })}
        </Layer>
      </Stage>

      {editor && (
        <InlineTextEditor
          rect={editor.rect}
          fontSize={editor.fontSize}
          fontFamily={editor.fontFamily}
          color={editor.color}
          initialValue={editor.initialValue}
          onCommit={commitText}
          onCancel={cancelText}
        />
      )}
    </div>
  );
}
