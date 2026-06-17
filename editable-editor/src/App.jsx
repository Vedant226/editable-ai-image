import React, {
  useEffect,
  useState,
  useRef,
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

import {
  getLayerData,
} from "./layerData";

/* ==========================
   EDITABLE LAYER
========================== */

function EditableLayer({
  shapeProps,
  isSelected,
  onSelect,
  onChange,
}) {
  const [image] = useImage(
    `/layers/${encodeURIComponent(
      shapeProps.file || ""
    )}`
  );

  const shapeRef =
    useRef(null);

  const trRef =
    useRef(null);

  // FIX TEXT DETECTION
  const isText =
    shapeProps.type
      ?.toLowerCase()
      .includes("text");

  useEffect(() => {
    if (
      isSelected &&
      trRef.current &&
      shapeRef.current
    ) {
      trRef.current.nodes([
        shapeRef.current,
      ]);

      trRef.current
        .getLayer()
        ?.batchDraw();
    }
  }, [isSelected]);

  const updatePosition =
    (e) => {
      onChange({
        ...shapeProps,
        x:
          e.target.x(),
        y:
          e.target.y(),
      });
    };

  const handleTransform =
    () => {
      const node =
        shapeRef.current;

      const scaleX =
        node.scaleX();

      const scaleY =
        node.scaleY();

      node.scaleX(1);
      node.scaleY(1);

      onChange({
        ...shapeProps,

        x:
          node.x(),

        y:
          node.y(),

        rotation:
          node.rotation(),

        width:
          Math.max(
            10,
            node.width() *
            scaleX
          ),

        height:
          Math.max(
            10,
            node.height() *
            scaleY
          ),
      });
    };

  const editText =
    () => {
      if (!isText)
        return;

      const newText =
        window.prompt(
          "Edit text:",
          shapeProps.text ||
          ""
        );

      if (
        newText ===
        null ||
        !newText.trim()
      )
        return;

      onChange({
        ...shapeProps,
        text: newText,
      });
    };

  return (
    <>
      {/* IMAGE */}
      {!isText &&
        image && (
          <Image
            ref={shapeRef}
            image={image}
            x={
              shapeProps.x
            }
            y={
              shapeProps.y
            }
            width={
              shapeProps.width
            }
            height={
              shapeProps.height
            }
            rotation={
              shapeProps.rotation ||
              0
            }
            draggable
            onClick={
              onSelect
            }
            onTap={
              onSelect
            }
            onDragEnd={
              updatePosition
            }
            onTransformEnd={
              handleTransform
            }
          />
        )}

      {/* TEXT */}
      {isText && (
        <Text
          ref={shapeRef}
          text={
            shapeProps.text ||
            "Edit me"
          }
          x={
            shapeProps.x
          }
          y={
            shapeProps.y
          }
          width={
            shapeProps.width
          }
          height={
            shapeProps.height
          }
          rotation={
            shapeProps.rotation ||
            0
          }
          fontFamily={
            shapeProps.fontFamily ||
            "Cinzel"
          }
          fontStyle="bold"
          fontSize={Math.max(
            14,
            shapeProps.height *
            0.55
          )}
          fill={
            shapeProps.fontColor ||
            "#d8b36a"
          }
          stroke={
            shapeProps.strokeColor ||
            "#5a2e12"
          }
          strokeWidth={
            shapeProps.strokeWidth ||
            1
          }
          draggable
          onClick={
            onSelect
          }
          onTap={
            onSelect
          }
          onDblClick={
            editText
          }
          onDragEnd={
            updatePosition
          }
          onTransformEnd={
            handleTransform
          }
        />
      )}

      {/* TRANSFORMER */}
      {isSelected && (
        <Transformer
          ref={trRef}
          rotateEnabled
        />
      )}
    </>
  );
}

/* ==========================
          APP
========================== */

export default function App() {
  const [layers,
    setLayers] =
    useState([]);

  const [selectedId,
    setSelectedId] =
    useState(null);

  const [
    backgroundImage,
  ] = useImage(
    "/layers/background.png"
  );

  useEffect(() => {
    async function load() {
      const data =
        await getLayerData();

      // BETTER FILTERING
      const filtered =
        data.filter(
          (layer) => {
            const type =
              layer.type
                ?.toLowerCase() ||
              "";

            const area =
              layer.width *
              layer.height;

            const isText =
              type.includes(
                "text"
              );

            if (isText)
              return true;

            // REMOVE BIG DUPLICATE CHUNKS
            const badKeywords =
              [
                "historical_portrait",
                "portrait_frame",
                "royal_portrait",
                "background",
                "decorative_border",
              ];

            const isBad =
              badKeywords.some(
                (
                  keyword
                ) =>
                  type.includes(
                    keyword
                  )
              );

            if (isBad)
              return false;

            if (
              area >
              35000
            )
              return false;

            if (
              layer.width >
              220 ||
              layer.height >
              220
            )
              return false;

            return true;
          }
        );

      filtered.sort(
        (a, b) =>
          a.zIndex -
          b.zIndex
      );

      setLayers(
        filtered
      );
    }

    load();
  }, []);

  return (
    <div
      style={{
        background:
          "#111",
        width:
          "100vw",
        height:
          "100vh",
        overflow:
          "hidden",
      }}
    >
      <Stage
        width={
          window.innerWidth
        }
        height={
          window.innerHeight
        }
        onMouseDown={(
          e
        ) => {
          const clickedOnEmpty =
            e.target ===
            e.target.getStage();

          if (
            clickedOnEmpty
          ) {
            setSelectedId(
              null
            );
          }
        }}
      >
        <Layer>

          {/* ORIGINAL IMAGE */}
          <Image
            image={
              backgroundImage
            }
            x={0}
            y={0}
            width={
              window.innerWidth
            }
            height={
              window.innerHeight
            }
            listening={
              false
            }
          />

          {/* CLICK ZONES */}
          {layers
            .filter(
              (
                layer
              ) =>
                !layer.edited
            )
            .map(
              (
                layer
              ) => (
                <Rect
                  key={`click-${layer.id}`}
                  x={
                    layer.x
                  }
                  y={
                    layer.y
                  }
                  width={
                    layer.width
                  }
                  height={
                    layer.height
                  }
                  fill="transparent"
                  listening
                  onClick={() => {
                    setLayers(
                      (
                        prev
                      ) =>
                        prev.map(
                          (
                            l
                          ) =>
                            l.id ===
                              layer.id
                              ? {
                                ...l,
                                edited: true,
                              }
                              : l
                        )
                    );
                  }}
                />
              )
            )}

          {/* EDITED OBJECTS */}
          {layers
            .filter(
              (
                layer
              ) =>
                layer.edited ===
                true
            )
            .map(
              (
                layer
              ) => (
                <EditableLayer
                  key={
                    layer.id
                  }
                  shapeProps={
                    layer
                  }
                  isSelected={
                    selectedId ===
                    layer.id
                  }
                  onSelect={() =>
                    setSelectedId(
                      layer.id
                    )
                  }
                  onChange={(
                    newAttrs
                  ) => {
                    setLayers(
                      (
                        prev
                      ) =>
                        prev.map(
                          (
                            l
                          ) =>
                            l.id ===
                              layer.id
                              ? newAttrs
                              : l
                        )
                    );
                  }}
                />
              )
            )}
        </Layer>
      </Stage>
    </div>
  );
}