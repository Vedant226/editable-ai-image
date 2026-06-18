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
import { createEditingIllusionEngine } from "./editingIllusion";

/* A Konva image bound to a (possibly dynamic) URL; renders nothing until loaded. */
function KImage({ url, ...props }) {
  const [img] = useImage(url);
  if (!img) return null;
  return <Image image={img} {...props} />;
}

/* ==========================
   EDITABLE LAYER
   One lifted Smart Object. Its pixels are the refined /lift CUTOUT once ready
   (a drop-in for the old SAM PNG at the same footprint), else the SAM PNG as an
   instant stand-in. A matte-hugging glow (Konva shadow over the alpha) marks
   selection. Drag/transform model is unchanged from Phase 5.
========================== */

function EditableLayer({ shapeProps, isSelected, isEditing, edited, glow, opacity, cutoutUrl, onChange, onStartTextEdit }) {
  const [image] = useImage(cutoutUrl || `/layers/${encodeURIComponent(shapeProps.file || "")}`);

  const shapeRef = useRef(null);
  const trRef = useRef(null);
  const isText = shapeProps.type?.toLowerCase().includes("text");

  useEffect(() => {
    if (isSelected && !isEditing && trRef.current && shapeRef.current) {
      trRef.current.nodes([shapeRef.current]);
      trRef.current.getLayer()?.batchDraw();
    }
  }, [isSelected, isEditing]);

  const glowProps =
    glow > 0.01
      ? { shadowColor: "#d8b36a", shadowBlur: 6 + glow * 16, shadowOpacity: glow, shadowForStrokeEnabled: false }
      : {};

  const updatePosition = (e) => onChange({ ...shapeProps, x: e.target.x(), y: e.target.y() });

  const handleTransform = () => {
    const node = shapeRef.current;
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
      {/* image, OR unedited text rendered as its ORIGINAL bitmap (identical to source) */}
      {(!isText || !edited) && image && (
        <Image
          ref={shapeRef}
          image={image}
          visible={!(isText && isEditing)}
          x={shapeProps.x}
          y={shapeProps.y}
          width={shapeProps.width}
          height={shapeProps.height}
          rotation={shapeProps.rotation || 0}
          opacity={opacity ?? 1}
          draggable
          onDblClick={isText ? () => onStartTextEdit?.(shapeProps.id) : undefined}
          onDblTap={isText ? () => onStartTextEdit?.(shapeProps.id) : undefined}
          onDragEnd={updatePosition}
          onTransformEnd={handleTransform}
          {...glowProps}
        />
      )}

      {/* once the user edits text, synthesize typography from the font estimation */}
      {isText && edited && (
        <Text
          ref={shapeRef}
          visible={!isEditing}
          text={shapeProps.text || "Edit me"}
          x={shapeProps.x}
          y={shapeProps.y}
          width={shapeProps.width}
          height={shapeProps.height}
          rotation={shapeProps.rotation || 0}
          opacity={opacity ?? 1}
          fontFamily={shapeProps.fontFamily || "Cinzel"}
          fontStyle="bold"
          fontSize={Math.max(14, shapeProps.height * 0.55)}
          fill={shapeProps.fontColor || "#d8b36a"}
          stroke={shapeProps.strokeColor || "#5a2e12"}
          strokeWidth={shapeProps.strokeWidth || 1}
          align="center"
          draggable
          onDblClick={() => onStartTextEdit?.(shapeProps.id)}
          onDblTap={() => onStartTextEdit?.(shapeProps.id)}
          onDragEnd={updatePosition}
          onTransformEnd={handleTransform}
          {...glowProps}
        />
      )}

      {isSelected && !isEditing && <Transformer ref={trRef} rotateEnabled />}
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
        background: "rgba(10,10,10,0.82)",
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
      if (eie.isAnimating(now)) requestAnimationFrame(step);
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
      if (cached && cached.cutout) {
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
        liftAssetsRef.current.set(id, {
          cutout: { url: p.cutout.png },
          fill: { url: p.fill.png, x: p.fill.x, y: p.fill.y, w: p.fill.w, h: p.fill.h },
          shadow: p.shadow,
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

  // ---- selection ----
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
      if (!picked) {
        update((s) => om.select(s, null));
        return;
      }
      const existing = session.entries[picked.id];
      if (existing && existing.state === "active") {
        update((s) => om.select(s, picked.id));
      } else {
        commit((s) => om.activate(s, picked.id).session);
        prefetchLift(picked.id);
      }
    },
    [om, isOpaqueAt, session.selectedId, session.entries, commit, update, prefetchLift]
  );

  const handleObjectChange = useCallback(
    (id, attrs, prevText) =>
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
      }),
    [om, commit]
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
      commit((s) => om.setText(s, id, v));
    },
    [editingId, om, commit]
  );

  // ---- toolbar actions ----
  const deleteSelected = useCallback(() => {
    const id = session.selectedId;
    if (id == null || !om) return;
    eie.delete(id, performance.now());
    kick();
    const ms = eie.config.deleteMs + 40;
    setTimeout(() => commit((s) => om.select(om.softDelete(s, id), null)), ms);
  }, [session.selectedId, om, eie, kick, commit]);

  const bringToFront = useCallback(() => {
    const id = session.selectedId;
    if (id != null && om) commit((s) => om.bringToFront(s, id));
  }, [session.selectedId, om, commit]);
  const sendToBack = useCallback(() => {
    const id = session.selectedId;
    if (id != null && om) commit((s) => om.sendToBack(s, id));
  }, [session.selectedId, om, commit]);

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
        {selected ? `selected: ${selected.category}#${selected.id}` : "click an object to lift it"}
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
      >
        <Layer>
          {/* BASE — the original image, the only visual source of truth */}
          <Image image={backgroundImage} x={0} y={0} width={imageSize.width} height={imageSize.height} listening={false} />

          {/* REPAIR FILLS — inpainted patches covering lifted/erased footprints */}
          {fills.map((f) => (
            <KImage key={`fill-${f.id}`} url={f.url} x={f.x} y={f.y} width={f.w} height={f.h} listening={false} />
          ))}

          {/* LIFTED SMART OBJECTS */}
          {scene.activeVisuals.map((v) => {
            const t = v.transform;
            const fr = eie.frame(v.objectId, now);
            const a = liftAssetsRef.current.get(v.objectId);
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
                edited={session.entries[v.objectId]?.text != null}
                glow={fr ? fr.selection.glow : session.selectedId === v.objectId ? 1 : 0}
                opacity={fr ? fr.cutout.opacity : 1}
                cutoutUrl={!v.isText && a && a.cutout ? a.cutout.url : null}
                onChange={(attrs) => handleObjectChange(v.objectId, attrs, v.text)}
                onStartTextEdit={startTextEdit}
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
