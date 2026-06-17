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
   stage (Object Manager pickAt), so this component only handles drag, transform
   and (text) double-click editing.
========================== */

function EditableLayer({ shapeProps, isSelected, onChange }) {
  const [image] = useImage(`/layers/${encodeURIComponent(shapeProps.file || "")}`);

  const shapeRef = useRef(null);
  const trRef = useRef(null);

  const isText = shapeProps.type?.toLowerCase().includes("text");

  useEffect(() => {
    if (isSelected && trRef.current && shapeRef.current) {
      trRef.current.nodes([shapeRef.current]);
      trRef.current.getLayer()?.batchDraw();
    }
  }, [isSelected]);

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

  const editText = () => {
    if (!isText) return;
    const newText = window.prompt("Edit text:", shapeProps.text || "");
    if (newText === null || !newText.trim()) return;
    onChange({ ...shapeProps, text: newText });
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
          draggable
          onDblClick={editText}
          onDragEnd={updatePosition}
          onTransformEnd={handleTransform}
        />
      )}

      {isSelected && <Transformer ref={trRef} rotateEnabled />}
    </>
  );
}

/* ==========================
   REPAIR PATCH
   The inpainted fill (from the backend) for a lifted object's original
   footprint. Sits between the base image and the lifted object, so moving the
   object reveals clean background instead of the original.
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
                onChange={(attrs) => handleObjectChange(v.objectId, attrs, v.text)}
              />
            );
          })}
        </Layer>
      </Stage>
    </div>
  );
}
