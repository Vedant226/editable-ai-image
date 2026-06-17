import React, {
  useEffect,
  useState,
  useRef,
  useMemo,
  useCallback,
} from "react";

import {
  Stage,
  Layer,
  Image,
  Transformer,
  Text,
} from "react-konva";

import useImage from "use-image";

import { createObjectManager } from "./objectManager";
import { createVisualResolver } from "./visualResolver";
import { useAlphaHitTester } from "./useAlphaHitTester";

/* ==========================
   EDITABLE LAYER
   Renders one lifted Smart Object (image or text). Selection is driven by the
   stage (Object Manager pickAt); this component handles drag, transform, and
   requesting inline text editing (double-click on text).
========================== */

function EditableLayer({ shapeProps, isSelected, isEditing, onChange, onStartTextEdit }) {
  const [image] = useImage(`/layers/${encodeURIComponent(shapeProps.file || "")}`);

  const shapeRef = useRef(null);
  const trRef = useRef(null);

  const isText = shapeProps.type?.toLowerCase().includes("text");

  useEffect(() => {
    if (isSelected && !isEditing && trRef.current && shapeRef.current) {
      trRef.current.nodes([shapeRef.current]);
      trRef.current.getLayer()?.batchDraw();
    }
  }, [isSelected, isEditing]);

  const updatePosition = (e) => {
    onChange({ ...shapeProps, x: e.target.x(), y: e.target.y() });
  };

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
      {!isText && image && (
        <Image
          ref={shapeRef}
          image={image}
          x={shapeProps.x}
          y={shapeProps.y}
          width={shapeProps.width}
          height={shapeProps.height}
          rotation={shapeProps.rotation || 0}
          draggable
          onDragEnd={updatePosition}
          onTransformEnd={handleTransform}
        />
      )}

      {isText && (
        <Text
          ref={shapeRef}
          visible={!isEditing}
          text={shapeProps.text || "Edit me"}
          x={shapeProps.x}
          y={shapeProps.y}
          width={shapeProps.width}
          height={shapeProps.height}
          rotation={shapeProps.rotation || 0}
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
        />
      )}

      {isSelected && !isEditing && <Transformer ref={trRef} rotateEnabled />}
    </>
  );
}

/* ==========================
   INLINE TEXT EDITOR
   An HTML textarea overlaid on the canvas at the text's on-screen position,
   matching font/scale. Commits on Enter or blur, cancels on Escape. Replaces
   the old window.prompt.
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
   REPAIR PATCH
   Inpainted fill (from the backend) for a lifted object's original footprint.
========================== */

function RepairPatch({ dataUrl, bbox }) {
  const [img] = useImage(dataUrl);
  if (!img || !bbox) return null;
  return (
    <Image
      image={img}
      x={bbox.x}
      y={bbox.y}
      width={bbox.w}
      height={bbox.h}
      listening={false}
    />
  );
}

/* ==========================
          APP
========================== */

const FALLBACK_IMAGE_SIZE = { width: 1408, height: 768 };

export default function App() {
  const [rawMetadata, setRawMetadata] = useState(null);
  const [session, setSession] = useState({ entries: {}, selectedId: null });
  const [viewport, setViewport] = useState({
    w: window.innerWidth,
    h: window.innerHeight,
  });
  const [inpaintEngine, setInpaintEngine] = useState(null);
  const [editingId, setEditingId] = useState(null);

  const [backgroundImage] = useImage("/layers/background.png");
  const stageRef = useRef(null);
  const repairCacheRef = useRef(new Map()); // objectId -> repair (cache across activations)
  const backendReadyRef = useRef(false);

  // ---- Object Manager + Visual Resolver (derived once from raw metadata) ----
  const om = useMemo(
    () => (rawMetadata ? createObjectManager(rawMetadata) : null),
    [rawMetadata]
  );
  const vr = useMemo(() => (om ? createVisualResolver(om) : null), [om]);
  const isOpaqueAt = useAlphaHitTester(om);

  useEffect(() => {
    fetch("/layers/metadata.json")
      .then((r) => r.json())
      .then(setRawMetadata)
      .catch((err) => {
        console.error("Failed to load metadata.json", err);
        setRawMetadata([]);
      });
  }, []);

  useEffect(() => {
    const onResize = () =>
      setViewport({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // probe the inpaint backend once; repairs are only requested if it's up
  useEffect(() => {
    let cancelled = false;
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((h) => {
        if (cancelled || !h) return;
        backendReadyRef.current = true;
        setInpaintEngine(h.engine);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // ---- single coordinate space: native image px, uniformly fit to viewport ----
  const imageSize = om ? om.getImageSize() : FALLBACK_IMAGE_SIZE;
  const scale = Math.min(
    viewport.w / imageSize.width,
    viewport.h / imageSize.height
  );
  const stageWidth = imageSize.width * scale;
  const stageHeight = imageSize.height * scale;

  // ---- on-demand inpaint: fill the hole behind a lifted object (cached) ----
  const requestRepair = useCallback(
    async (id) => {
      if (!om || !backendReadyRef.current) return;

      const cached = repairCacheRef.current.get(id);
      if (cached) {
        setSession((s) => om.attachRepair(s, id, cached));
        return;
      }

      repairCacheRef.current.set(id, { status: "pending" });
      setSession((s) => om.attachRepair(s, id, { status: "pending" }));

      try {
        const res = await fetch("/api/inpaint", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ objectId: id }),
        });
        if (!res.ok) throw new Error(`inpaint ${res.status}`);
        const p = await res.json();
        const repair = {
          status: "ready",
          dataUrl: p.png,
          bbox: { x: p.x, y: p.y, w: p.w, h: p.h },
        };
        repairCacheRef.current.set(id, repair);
        setSession((s) => om.attachRepair(s, id, repair));
      } catch (err) {
        console.warn("repair failed", err);
        const repair = { status: "failed" };
        repairCacheRef.current.set(id, repair);
        setSession((s) => om.attachRepair(s, id, repair));
      }
    },
    [om]
  );

  // ---- selection: a click resolves to the best object via the OM ----
  const handleStageClick = useCallback(
    (e) => {
      if (!om) return;
      const stage = e.target.getStage();
      if (!stage) return;

      // ignore clicks on transformer handles (resize/rotate anchors)
      const parent = e.target.getParent && e.target.getParent();
      if (parent && parent.className === "Transformer") return;

      const point = stage.getRelativePointerPosition();
      if (!point) return;

      const picked = om.pickAt(point, {
        currentSelectionId: session.selectedId,
        isOpaqueAt,
      });

      if (!picked) {
        setSession((s) => om.select(s, null)); // empty click → deselect
        return;
      }

      setSession((s) => om.activate(s, picked.id).session); // lift + select
      requestRepair(picked.id);
    },
    [om, isOpaqueAt, session.selectedId, requestRepair]
  );

  const handleObjectChange = useCallback(
    (id, attrs, prevText) =>
      setSession((s) => {
        let next = om.applyTransform(s, id, {
          x: attrs.x,
          y: attrs.y,
          width: attrs.width,
          height: attrs.height,
          rotation: attrs.rotation,
        });
        if (typeof attrs.text === "string" && attrs.text !== prevText) {
          next = om.setText(next, id, attrs.text);
        }
        return next;
      }),
    [om]
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
      if (!v) return; // empty → keep previous text
      setSession((s) => om.setText(s, id, v));
    },
    [editingId, om]
  );

  // ---- export at native resolution, regardless of on-screen fit-scale ----
  const handleExport = useCallback(() => {
    const stage = stageRef.current;
    if (!stage) return;
    setSession((s) => (om ? om.select(s, null) : s)); // hide transformer
    requestAnimationFrame(() => {
      const uri = stage.toDataURL({ pixelRatio: 1 / scale });
      const link = document.createElement("a");
      link.download = "edited.png";
      link.href = uri;
      link.click();
    });
  }, [scale, om]);

  const scene = vr ? vr.resolveScene(session) : { activeVisuals: [], repairs: [] };

  const selected =
    session.selectedId != null && om ? om.getObject(session.selectedId) : null;

  // screen-space rect for the inline text editor overlay
  let editor = null;
  if (editingId != null && om && stageRef.current) {
    const o = om.getObject(editingId);
    if (o) {
      const entry = session.entries[editingId];
      const t =
        entry?.transform || {
          x: o.bbox.x,
          y: o.bbox.y,
          width: o.bbox.w,
          height: o.bbox.h,
          rotation: o.rotation,
        };
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
      <div
        style={{
          position: "absolute",
          top: 12,
          left: 12,
          zIndex: 10,
          color: "#d8b36a",
          font: "13px/1.4 monospace",
          background: "rgba(0,0,0,0.45)",
          padding: "6px 10px",
          borderRadius: 6,
          pointerEvents: "none",
        }}
      >
        {selected
          ? `selected: ${selected.category}#${selected.id}`
          : "click an object to lift it"}
        {selected?.role === "text" ? "  ·  double-click to edit" : ""}
        {`  ·  inpaint: ${inpaintEngine || "offline"}`}
      </div>

      <button
        onClick={handleExport}
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          zIndex: 10,
          padding: "8px 14px",
          background: "#d8b36a",
          color: "#1a1a1a",
          border: "none",
          borderRadius: 6,
          fontWeight: 600,
          cursor: "pointer",
        }}
      >
        Export PNG
      </button>

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
          <Image
            image={backgroundImage}
            x={0}
            y={0}
            width={imageSize.width}
            height={imageSize.height}
            listening={false}
          />

          {/* REPAIRS — inpainted patches covering lifted objects' footprints */}
          {scene.repairs.map((r) => (
            <RepairPatch key={`repair-${r.objectId}`} dataUrl={r.dataUrl} bbox={r.bbox} />
          ))}

          {/* LIFTED SMART OBJECTS (resolved by the Visual Object Resolver) */}
          {scene.activeVisuals.map((v) => {
            const t = v.transform;
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
