import React, {
  useEffect,
  useState,
  useRef,
  useCallback,
} from "react";

import {
  Stage,
  Layer,
  Image,
  Transformer,
  Text,
  Rect,
} from "react-konva";

import useImage from "use-image";

import { getLayerData } from "./layerData";

/* ==========================
   EDITABLE LAYER
   (behavior unchanged — Phase 2 will route this through the Object Manager)
========================== */

function EditableLayer({ shapeProps, isSelected, onSelect, onChange }) {
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
      {/* IMAGE */}
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
          onClick={onSelect}
          onTap={onSelect}
          onDragEnd={updatePosition}
          onTransformEnd={handleTransform}
        />
      )}

      {/* TEXT */}
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
          onClick={onSelect}
          onTap={onSelect}
          onDblClick={editText}
          onDragEnd={updatePosition}
          onTransformEnd={handleTransform}
        />
      )}

      {/* TRANSFORMER */}
      {isSelected && <Transformer ref={trRef} rotateEnabled />}
    </>
  );
}

/* ==========================
          APP
========================== */

// Used only until the real background.png reports its natural size.
const FALLBACK_IMAGE_SIZE = { width: 1408, height: 768 };

export default function App() {
  const [layers, setLayers] = useState([]);
  const [selectedId, setSelectedId] = useState(null);

  const [backgroundImage] = useImage("/layers/background.png");

  // Viewport size drives the fit-scale; we recompute on resize.
  const [viewport, setViewport] = useState({
    w: window.innerWidth,
    h: window.innerHeight,
  });

  const stageRef = useRef(null);

  /* ----------------------------------------------------------
     SINGLE COORDINATE SPACE
     The canvas works entirely in NATIVE image pixels. The whole
     stage is then uniformly scaled to fit the viewport, so the
     background, click zones, editable objects and export all
     share one coordinate system (fixes click misalignment and
     lets us export at full resolution).
  ---------------------------------------------------------- */
  const imageSize =
    backgroundImage && backgroundImage.naturalWidth
      ? {
          width: backgroundImage.naturalWidth,
          height: backgroundImage.naturalHeight,
        }
      : FALLBACK_IMAGE_SIZE;

  const scale = Math.min(
    viewport.w / imageSize.width,
    viewport.h / imageSize.height
  );

  const stageWidth = imageSize.width * scale;
  const stageHeight = imageSize.height * scale;

  useEffect(() => {
    const onResize = () =>
      setViewport({ w: window.innerWidth, h: window.innerHeight });

    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    async function load() {
      const data = await getLayerData();

      // NOTE: temporary heuristic filter — Phase 2 replaces this with the
      // Object Manager. Kept verbatim so Phase 1 only changes coordinates.
      const filtered = data.filter((layer) => {
        const type = layer.type?.toLowerCase() || "";
        const area = layer.width * layer.height;
        const isText = type.includes("text");

        if (isText) return true;

        const badKeywords = [
          "historical_portrait",
          "portrait_frame",
          "royal_portrait",
          "background",
          "decorative_border",
        ];

        const isBad = badKeywords.some((keyword) => type.includes(keyword));

        if (isBad) return false;
        if (area > 35000) return false;
        if (layer.width > 220 || layer.height > 220) return false;

        return true;
      });

      filtered.sort((a, b) => a.zIndex - b.zIndex);

      setLayers(filtered);
    }

    load();
  }, []);

  /* ----------------------------------------------------------
     EXPORT at native resolution, independent of the on-screen
     fit-scale. The stage is drawn at `scale`, so pixelRatio
     1/scale renders it back out at full image resolution.
  ---------------------------------------------------------- */
  const handleExport = useCallback(() => {
    const stage = stageRef.current;
    if (!stage) return;

    setSelectedId(null); // keep the transformer handles out of the export

    requestAnimationFrame(() => {
      const uri = stage.toDataURL({ pixelRatio: 1 / scale });
      const link = document.createElement("a");
      link.download = "edited.png";
      link.href = uri;
      link.click();
    });
  }, [scale]);

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
        onMouseDown={(e) => {
          const clickedOnEmpty = e.target === e.target.getStage();
          if (clickedOnEmpty) setSelectedId(null);
        }}
      >
        <Layer>
          {/* ORIGINAL IMAGE — drawn at native size, the visual source of truth */}
          <Image
            image={backgroundImage}
            x={0}
            y={0}
            width={imageSize.width}
            height={imageSize.height}
            listening={false}
          />

          {/* CLICK ZONES (native coords now align with the background) */}
          {layers
            .filter((layer) => !layer.edited)
            .map((layer) => (
              <Rect
                key={`click-${layer.id}`}
                x={layer.x}
                y={layer.y}
                width={layer.width}
                height={layer.height}
                fill="transparent"
                listening
                onClick={() => {
                  setLayers((prev) =>
                    prev.map((l) =>
                      l.id === layer.id ? { ...l, edited: true } : l
                    )
                  );
                }}
              />
            ))}

          {/* EDITED OBJECTS */}
          {layers
            .filter((layer) => layer.edited === true)
            .map((layer) => (
              <EditableLayer
                key={layer.id}
                shapeProps={layer}
                isSelected={selectedId === layer.id}
                onSelect={() => setSelectedId(layer.id)}
                onChange={(newAttrs) => {
                  setLayers((prev) =>
                    prev.map((l) => (l.id === layer.id ? newAttrs : l))
                  );
                }}
              />
            ))}
        </Layer>
      </Stage>
    </div>
  );
}
