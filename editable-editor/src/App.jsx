import React, {
  useEffect,
  useState,
  useRef,
  useMemo,
  useReducer,
  useCallback,
} from "react";

import { Stage, Layer, Image, Transformer, Text, TextPath } from "react-konva";
import useImage from "use-image";

import { createObjectManager } from "./objectManager";
import { createVisualResolver } from "./visualResolver";
import { useAlphaHitTester } from "./useAlphaHitTester";
import { createEditingIllusionEngine, PHASE } from "./editingIllusion";
import { estimateTypography } from "./typography";
import PropertyInspector from "./PropertyInspector";
import { categoryGroup, getActions } from "./propertyRegistry";

/* Decode an image URL/data-URL to a ready-to-paint HTMLImageElement (or null on
   failure). Used so lift bitmaps are decoded BEFORE a node depends on them — a
   node never has to mount against a not-yet-loaded image (which would blank a
   frame and, mid-drag, orphan the Konva node). */
function decodeImage(url) {
  return new Promise((resolve) => {
    if (!url) return resolve(null);
    const img = new window.Image();
    let done = false;
    const finish = (v) => {
      if (done) return;
      done = true;
      clearTimeout(t);
      resolve(v);
    };
    // safety timeout: a load/error that never fires must not wedge the object
    // in a permanent "pending" state (manipulable would never become true)
    const t = setTimeout(() => finish(null), 8000);
    img.onload = () => finish(img);
    img.onerror = () => finish(null);
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

function EditableLayer({ shapeProps, isSelected, isEditing, edited, textFade, manipulable, glow, opacity, cutoutImg, editImg, refText, styleOverride, onChange, onStartTextEdit, onLiftStart, onSelect }) {
  // The resting highlight draws the ORIGINAL /layers PNG (closest to the base);
  // the refined /lift cutout (decoded element, passed in) is used only once
  // lifted. Because both are READY HTMLImageElements, swapping between them
  // keeps the SAME Konva node mounted — a drag in progress is never orphaned.
  const [layerImage] = useImage(`/layers/${encodeURIComponent(shapeProps.file || "")}`);
  // an integrated AI edit (an object-sized canvas) BECOMES the object's
  // appearance — so it moves/scales/rotates with this node and truly replaces the
  // original pixels (no separate overlay that can drift or survive underneath)
  const image = editImg || cutoutImg || layerImage;

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

  // synthesize typography that matches the ORIGINAL bitmap (sampled colour /
  // gradient, font-family matched by width, size from cap height, outline +
  // shadow), so edited text reads as the artwork, not as HTML. Recomputed when
  // the string/footprint changes; falls back to metadata until the bitmap loads.
  const synthStyle = useMemo(
    () =>
      isText && edited && image
        ? estimateTypography(image, {
            text: shapeProps.text || "",
            refText: refText || shapeProps.text || "",
            width: shapeProps.width,
            height: shapeProps.height,
            fallbackFamily: shapeProps.fontFamily,
            fallbackColor: shapeProps.fontColor,
          })
        : null,
    [isText, edited, image, shapeProps.text, refText, shapeProps.width, shapeProps.height, shapeProps.fontFamily, shapeProps.fontColor]
  );

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
  // dupes select themselves on click (cancelBubble so the Stage's canvas
  // selection doesn't also fire); canonical objects keep selecting via the Stage.
  const selectClick = onSelect
    ? (e) => {
        e.cancelBubble = true;
        onSelect(shapeProps.id);
      }
    : undefined;
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
          onClick={selectClick}
          onTap={selectClick}
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
      {isText && edited && !styleOverride?.curve && (
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
          fontFamily={styleOverride?.fontFamily || synthStyle?.fontFamily || shapeProps.fontFamily || "Cinzel"}
          fontStyle={styleOverride?.fontStyle || synthStyle?.fontStyle || "bold"}
          fontSize={styleOverride?.fontSize || synthStyle?.fontSize || Math.max(14, shapeProps.height * 0.55)}
          letterSpacing={styleOverride?.letterSpacing ?? synthStyle?.letterSpacing ?? 0}
          lineHeight={styleOverride?.lineHeight ?? 1}
          // user gradient override > user colour override > sampled gradient > sampled solid
          {...(styleOverride?.gradient
            ? {
                fillLinearGradientStartPoint: { x: 0, y: 0 },
                fillLinearGradientEndPoint: { x: 0, y: shapeProps.height },
                fillLinearGradientColorStops: styleOverride.gradient,
              }
            : styleOverride?.fill
            ? { fill: styleOverride.fill }
            : synthStyle?.gradient
            ? {
                fillLinearGradientStartPoint: { x: 0, y: 0 },
                fillLinearGradientEndPoint: { x: 0, y: shapeProps.height },
                fillLinearGradientColorStops: synthStyle.gradient,
              }
            : { fill: synthStyle?.fill || shapeProps.fontColor || "#d8b36a" })}
          stroke={styleOverride?.stroke ?? synthStyle?.stroke ?? shapeProps.strokeColor ?? "#5a2e12"}
          strokeWidth={styleOverride?.strokeWidth ?? synthStyle?.strokeWidth ?? (shapeProps.strokeWidth || 1)}
          // user shadow (Shadow slider) overrides the sampled drop-shadow when set
          shadowColor={styleOverride?.shadowBlur != null ? (styleOverride.shadowColor || "#000000") : synthStyle?.shadowColor}
          shadowBlur={styleOverride?.shadowBlur ?? synthStyle?.shadowBlur}
          shadowOpacity={styleOverride?.shadowBlur != null ? (styleOverride.shadowBlur > 0 ? 0.55 : 0) : synthStyle?.shadowOpacity}
          shadowOffsetY={styleOverride?.shadowBlur != null ? 2 : synthStyle?.shadowOffsetY}
          align={styleOverride?.align || "center"}
          verticalAlign="middle"
          wrap="none"
          draggable={manipulable}
          onClick={selectClick}
          onTap={selectClick}
          onDblClick={() => onStartTextEdit?.(shapeProps.id)}
          onDblTap={() => onStartTextEdit?.(shapeProps.id)}
          onDragStart={liftStart}
          onDragEnd={updatePosition}
          onTransformStart={liftStart}
          onTransformEnd={handleTransform}
          {...glowProps}
        />
      )}

      {/* curved text: when the Curve slider is set, render along an arc path
          (TextPath). Solid fill only (gradient falls back to fill). Unset (0) =
          the plain <Text> above, so default behaviour is unchanged. */}
      {isText && edited && !!styleOverride?.curve && (
        <TextPath
          ref={textRef}
          visible={!isEditing}
          text={shapeProps.text || "Edit me"}
          x={shapeProps.x}
          y={shapeProps.y}
          rotation={shapeProps.rotation || 0}
          opacity={(opacity ?? 1) * tf}
          data={(() => {
            const W = shapeProps.width, H = shapeProps.height;
            const c = Math.max(-1, Math.min(1, (styleOverride.curve || 0) / 100));
            const my = H / 2;
            return `M 0 ${my} Q ${W / 2} ${my - c * H * 0.9} ${W} ${my}`;
          })()}
          align="center"
          fontFamily={styleOverride?.fontFamily || synthStyle?.fontFamily || shapeProps.fontFamily || "Cinzel"}
          fontStyle={styleOverride?.fontStyle || synthStyle?.fontStyle || "bold"}
          fontSize={styleOverride?.fontSize || synthStyle?.fontSize || Math.max(14, shapeProps.height * 0.55)}
          letterSpacing={styleOverride?.letterSpacing ?? synthStyle?.letterSpacing ?? 0}
          fill={styleOverride?.fill || synthStyle?.fill || shapeProps.fontColor || "#d8b36a"}
          stroke={styleOverride?.stroke ?? synthStyle?.stroke ?? shapeProps.strokeColor ?? "#5a2e12"}
          strokeWidth={styleOverride?.strokeWidth ?? synthStyle?.strokeWidth ?? (shapeProps.strokeWidth || 1)}
          shadowColor={styleOverride?.shadowBlur != null ? (styleOverride.shadowColor || "#000000") : synthStyle?.shadowColor}
          shadowBlur={styleOverride?.shadowBlur ?? synthStyle?.shadowBlur}
          shadowOpacity={styleOverride?.shadowBlur != null ? (styleOverride.shadowBlur > 0 ? 0.55 : 0) : synthStyle?.shadowOpacity}
          draggable={manipulable}
          onClick={selectClick}
          onTap={selectClick}
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

// Professional preset menus for the face AI tools (Phase 12). Each preset is
// [label, prompt, direction?]; the modal also exposes an intensity slider that
// maps to edit strength (denoise) on the backend. Icons label the dialog.
const FACE_PRESETS = {
  smile: { title: "Smile", icon: "😀", presets: [
    ["Subtle", "a subtle gentle closed-mouth smile"],
    ["Natural", "a warm natural smile"],
    ["Happy", "a happy smile showing teeth"],
    ["Laughing", "a big joyful laughing open smile"],
  ] },
  beard: { title: "Beard", icon: "🧔", presets: [
    ["Stubble", "light short stubble"],
    ["Short", "a short neatly trimmed beard"],
    ["Full", "a full thick beard"],
    ["Goatee", "a goatee and moustache"],
  ] },
  glasses: { title: "Glasses", icon: "👓", presets: [
    ["Round", "round metal eyeglasses"],
    ["Square", "square framed eyeglasses"],
    ["Rimless", "thin rimless eyeglasses"],
    ["Sunglasses", "dark sunglasses"],
  ] },
  age: { title: "Age", icon: "⏳", presets: [
    ["Younger", "a youthful younger version of the face", "younger"],
    ["Adult", "an adult version of the face"],
    ["Older", "an older aged face with natural wrinkles", "older"],
  ] },
  skin: { title: "Skin Tone", icon: "🎨", presets: [
    ["Smooth", "smooth clear retouched skin"],
    ["Fair", "fair light skin tone"],
    ["Tan", "warm sun-tanned skin tone"],
    ["Deep", "deep rich skin tone"],
  ] },
};

// Quick-fill style presets for the Change Clothes / Change Hair prompt dialogs.
const CLOTHES_PRESETS = [
  ["Suit", "a tailored business suit"], ["Casual", "casual everyday clothes"],
  ["Royal", "an ornate royal robe with gold trim"], ["Traditional", "traditional cultural attire"],
  ["Formal", "elegant formal evening wear"], ["Fantasy", "ornate fantasy armor"],
];
const HAIR_PRESETS = [
  ["Short", "short hair"], ["Long", "long flowing hair"], ["Curly", "curly hair"],
  ["Straight", "long straight hair"], ["Ponytail", "hair tied in a ponytail"],
  ["Buzz Cut", "a short buzz cut"], ["Bald", "bald head with no hair"],
];
const BG_GEN_PRESETS = [
  ["Studio", "a clean professional studio backdrop, soft even lighting"],
  ["Outdoor", "a natural outdoor landscape background, soft daylight, depth"],
  ["Sky", "a bright blue sky with soft clouds"],
  ["Gradient", "a smooth neutral gradient backdrop"],
  ["Bokeh", "a softly blurred bokeh background, shallow depth of field"],
];
const presetChip = {
  padding: "5px 10px", borderRadius: 999, border: "1px solid rgba(255,255,255,0.14)",
  background: "rgba(255,255,255,0.06)", color: "#f0ead9", cursor: "pointer",
  font: "12px -apple-system, Segoe UI, system-ui, sans-serif",
};

export default function App() {
  const [rawMetadata, setRawMetadata] = useState(null);
  const [history, setHistory] = useState({ past: [], present: { entries: {}, selectedId: null }, future: [] });
  const session = history.present;
  // latest session, readable from async callbacks without stale closures or
  // re-creating those callbacks on every edit
  const sessionRef = useRef(session);
  sessionRef.current = session;

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
  const dupCounterRef = useRef(0); // monotonic id source for duplicated instances
  const TEXT_FADE_MS = 260;

  // ---- Replace Object (AI) — overlay patches keyed by object id, plus the
  // upload trigger + in-flight/error UI. Additive: never mutates locked state.
  const replaceAssetsRef = useRef(new Map()); // id -> { img, x, y, w, h }  (face edits only — deferred)
  // Integrated AI edits (the 7 object-level features): each edit composites its
  // patch onto the object's CURRENT bitmap (original PNG, or a prior edit — they
  // accumulate) into ONE object-sized bitmap, rendered THROUGH the object's
  // EditableLayer at its live transform. The canvas lives here, versioned
  // (`${id}:${editId}`) so undo/redo pick the right one; the session entry
  // carries the `aiEdit` editId, so the edit is one undo step, moves/scales/
  // rotates with the object, exports, and clears on delete. This REPLACES the old
  // decoupled, statically-pinned top-most overlay that left the original beneath.
  const editAssetsRef = useRef(new Map()); // `${id}:${editId}` -> HTMLCanvasElement
  const editSeqRef = useRef(0);
  const fileInputRef = useRef(null);
  const pendingReplaceRef = useRef(null);
  const [replacingId, setReplacingId] = useState(null);
  const [replaceError, setReplaceError] = useState(null);

  // ---- Background Replacement (AI) — full-canvas result + its own picker.
  // Reuses the shared progress overlay + error toast; never touches the object-
  // replace flow above. Additive only.
  const bgFileInputRef = useRef(null);
  const bgColorInputRef = useRef(null);            // native picker for Background → Color
  const [bgBusy, setBgBusy] = useState(false);
  const [bgResult, setBgResult] = useState(null); // { img } full-canvas image
  const [bgGenModal, setBgGenModal] = useState(null); // Background → Generate prompt dialog
  const [bgGenText, setBgGenText] = useState("");
  const [lastEditInfo, setLastEditInfo] = useState(null); // transient "detected style · material · quality" readout

  // Surface the planner's analysis + self-eval after an AI edit (transparency).
  // No-op for responses without plan/quality, so it's safe to call everywhere.
  const noteEdit = useCallback((p) => {
    if (!p) return;
    const plan = p.plan || {}, q = p.quality || {};
    const parts = [];
    if (plan.style) parts.push(String(plan.style).replace(/_/g, " "));
    if (plan.material) parts.push(plan.material);
    if (q.score != null) parts.push(`quality ${Math.round(q.score * 100)}%`);
    if (!parts.length) return;
    setLastEditInfo(parts.join(" · "));
    setTimeout(() => setLastEditInfo(null), 6000);
  }, []);

  // ---- Remove Object (AI) — erases an object; the returned footprint patch is
  // stored in the SAME overlay store as Replace Object (reuse) and rendered by
  // the existing `replacements` pass. Reuses the shared progress/error UI too.
  const [removingId, setRemovingId] = useState(null);

  // ---- Recolor Object (AI) — color picker -> footprint patch (reuses overlay
  // store + shared progress/error UI). colorInputRef opens the native picker.
  const colorInputRef = useRef(null);
  const pendingRecolorRef = useRef(null);
  const [recoloringId, setRecoloringId] = useState(null);

  // ---- Change Clothes (AI) — prompt dialog -> clothing-only patch. Reuses the
  // overlay store + shared progress/error UI.
  const [clothesModal, setClothesModal] = useState(null); // { id }
  const [clothesText, setClothesText] = useState("");
  const [clothesBusy, setClothesBusy] = useState(false);
  const [aiIntensity, setAiIntensity] = useState(50); // shared edit strength for clothes/hair/person (50% == prior default)

  // ---- Change Hair (AI) — prompt dialog -> hair-only patch. Reuses the overlay
  // store + shared progress/error UI.
  const [hairModal, setHairModal] = useState(null); // { id }
  const [hairText, setHairText] = useState("");
  const [hairBusy, setHairBusy] = useState(false);

  // ---- Person Replace (AI) — dual-option dialog (upload image OR describe a
  // person) -> footprint patch. Reuses the overlay store + shared progress/error.
  const [personModal, setPersonModal] = useState(null); // { id }
  const [personText, setPersonText] = useState("");
  const [personBusy, setPersonBusy] = useState(false);
  const personFileInputRef = useRef(null);
  const pendingPersonRef = useRef(null);

  // ---- Identity Preservation (AI) — one optional toggle, OFF by default. When
  // ON, Hair / Clothes / Person Replace additionally call /comfyui/identity to
  // impose the reference identity (no behavior change when OFF). Additive only.
  const [preserveIdentity, setPreserveIdentity] = useState(true); // smart default: keep identity for face/hair/clothes/person

  // Crop a region of the original base image to a PNG data URL (the identity
  // reference for "preserve the original person").
  const cropRegion = useCallback((x, y, w, h) => {
    if (!backgroundImage) return null;
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    c.getContext("2d").drawImage(backgroundImage, x, y, w, h, 0, 0, w, h);
    return c.toDataURL("image/png");
  }, [backgroundImage]);

  // Run the identity engine over a feature patch; on any failure return the patch
  // unchanged so enabling identity never breaks a result.
  const applyIdentity = useCallback(async (patch, referenceDataUrl) => {
    if (!referenceDataUrl) return patch;
    try {
      const res = await fetch("/comfyui/identity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ referenceImage: referenceDataUrl, targetImage: patch.png, strength: 0.9 }),
      });
      if (!res.ok) return patch;
      const j = await res.json();
      return j && j.png ? { ...patch, png: j.png } : patch;
    } catch {
      return patch;
    }
  }, []);

  // ---- Object Manager + Visual Resolver ----
  const om = useMemo(() => (rawMetadata ? createObjectManager(rawMetadata) : null), [rawMetadata]);
  // live OM, readable from callbacks that may have been created before metadata
  // loaded (om starts null). composeEdit is reached THROUGH handlers that don't
  // list it as a dep, so without this it could close over a stale null om.
  const omRef = useRef(om);
  omRef.current = om;
  const vr = useMemo(() => (om ? createVisualResolver(om) : null), [om]);
  const isOpaqueAt = useAlphaHitTester(om);

  // ---- single coordinate space ----
  const imageSize = om ? om.getImageSize() : FALLBACK_IMAGE_SIZE;
  const scale = Math.min(viewport.w / imageSize.width, viewport.h / imageSize.height);
  const stageWidth = imageSize.width * scale;
  const stageHeight = imageSize.height * scale;

  // ---- Lighting & Shadow Harmonization (AI) — one optional toggle, OFF by
  // default. When ON, Replace / Hair / Clothes / Background / Person pass their
  // patch through /comfyui/lighting (a fast CPU post-process). On any failure it
  // returns the patch unchanged, so it never blocks editing. Additive only.
  const [autoLighting, setAutoLighting] = useState(true); // smart default: harmonize lighting in-engine

  const applyLighting = useCallback(async (patch, fullImage = false) => {
    try {
      const m = fullImage ? 0 : Math.round(Math.min(96, 0.4 * Math.max(patch.w, patch.h)));
      const rx = Math.max(0, patch.x - m), ry = Math.max(0, patch.y - m);
      const rx2 = Math.min(imageSize.width, patch.x + patch.w + m);
      const ry2 = Math.min(imageSize.height, patch.y + patch.h + m);
      const reference = cropRegion(rx, ry, rx2 - rx, ry2 - ry);
      if (!reference) return patch;
      const res = await fetch("/comfyui/lighting", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patch: patch.png, reference, patchLeft: patch.x - rx, patchTop: patch.y - ry }),
      });
      if (!res.ok) return patch;
      const j = await res.json();
      return j && j.png ? { ...patch, png: j.png } : patch;
    } catch {
      return patch;
    }
  }, [cropRegion, imageSize]);

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
      // only a successful, fill-bearing cache counts as done; a prior failure may retry
      if (cached && cached.readyAt && cached.fill) {
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
        const p = await res.json(); noteEdit(p);
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
        // the fill is what covers the footprint; if it didn't decode we cannot
        // lift without exposing a hole — mark failed (retryable) rather than
        // caching a permanently un-manipulable "ready" record.
        if (!fillImg) {
          liftAssetsRef.current.set(id, { failed: true });
          eie.setAssets(id, "failed", performance.now());
          return;
        }
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

  // ---- integrate an AI edit INTO the object (the 7 object-level features) ----
  // Composite the returned patch onto the object's current bitmap → ONE object-
  // sized bitmap that BECOMES the object's appearance (rendered through its
  // EditableLayer at the live transform). The object is activated so its lift
  // fill hides the original baked-in pixels, and the edit is recorded on the
  // session entry (`aiEdit` = versioned canvas id) so it is a single undo step
  // and clears on delete. The original is truly replaced — never an overlay
  // stacked on top of surviving pixels.
  const composeEdit = useCallback(
    async (id, patchImg, patch) => {
      const om = omRef.current; // always the live OM (never a stale-null closure)
      if (id == null || !om || !patchImg) return false;
      const o = om.getObject(id);
      if (!o) return false; // duplicates / non-OM ids: no integrated edit
      const bbox = o.bbox; // {x,y,w,h} native object box
      const w = Math.max(1, Math.round(bbox.w));
      const h = Math.max(1, Math.round(bbox.h));
      // Base = the object's CURRENT edited bitmap (so edits accumulate, e.g.
      // recolor then change-clothes) or, first time, its original layer PNG.
      const prevId = sessionRef.current.entries[id]?.aiEdit;
      let baseImg = prevId != null ? editAssetsRef.current.get(`${id}:${prevId}`) : null;
      if (!baseImg) baseImg = await decodeImage(`/layers/${encodeURIComponent(o.file || "")}`);
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      const ctx = c.getContext("2d");
      if (baseImg) ctx.drawImage(baseImg, 0, 0, w, h);
      // place the patch at its sub-position within the object box (full-footprint
      // patches fill it; sub-region patches — clothes/hair — only their region)
      const px = (patch.x ?? bbox.x) - bbox.x;
      const py = (patch.y ?? bbox.y) - bbox.y;
      ctx.drawImage(patchImg, px, py, patch.w ?? w, patch.h ?? h);
      const editId = (editSeqRef.current += 1);
      editAssetsRef.current.set(`${id}:${editId}`, c);
      // warm the footprint fill so a later drag never reveals a hole, then record
      // the edit (activate so the fill is in place + the edit actually renders)
      prefetchLift(id);
      commit((s) => {
        const base = s.entries[id]?.state === "active" ? s : om.activate(s, id).session;
        const e = base.entries[id];
        return { ...base, entries: { ...base.entries, [id]: { ...e, aiEdit: editId } } };
      });
      kick();
      return true;
    },
    [commit, prefetchLift, kick]
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
  // only canonical (OM-backed) objects have the inline editor + commit path; a
  // duplicate has no OM object, so editing it would be a dead-end — no-op it.
  const startTextEdit = useCallback((id) => { if (om?.getObject(id)) setEditingId(id); }, [om]);
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
    // a duplicate has no footprint to repair — fade it out and drop the entry
    if (!om.getObject(id) && session.entries[id]?.meta) {
      eie.delete(id, performance.now());
      kick();
      setTimeout(
        () =>
          commit((s) => {
            const entries = { ...s.entries };
            delete entries[id];
            return { ...s, entries, selectedId: s.selectedId === id ? null : s.selectedId };
          }),
        eie.config.deleteMs + 40
      );
      return;
    }
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
  }, [session.selectedId, session.entries, om, eie, kick, commit, prefetchLift]);

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

  // ---- duplicate: an additive COPY (its own synthetic session entry carrying
  // `meta`, so it needs no /lift fill — it sits on top of the scene and reveals
  // it when faded/moved). Reuses selection, drag, z-order, the inspector and the
  // EIE; never touches the OM or the canonical selection path. ----
  const selectDupe = useCallback((id) => update((s) => ({ ...s, selectedId: id })), [update]);
  const duplicateSelected = useCallback(() => {
    const id = session.selectedId;
    if (id == null || !om) return;
    const o = om.getObject(id);
    const srcEntry = session.entries[id];
    let meta, baseT;
    if (o) {
      meta = { category: o.category, role: o.role, file: o.file, bbox: { ...o.bbox }, style: o.style || null, text: srcEntry?.text ?? o.text };
      baseT = srcEntry?.transform || { x: o.bbox.x, y: o.bbox.y, width: o.bbox.w, height: o.bbox.h, rotation: o.rotation };
    } else if (srcEntry?.meta) {
      meta = { ...srcEntry.meta }; // duplicate of a duplicate
      baseT = srcEntry.transform;
    } else return;
    const dupId = `dup:${(dupCounterRef.current += 1)}`;
    const t = { x: baseT.x + 24, y: baseT.y + 24, width: baseT.width, height: baseT.height, rotation: baseT.rotation || 0 };
    // carry any integrated AI edit onto the copy: point a dupe-keyed canvas at the
    // SAME bitmap so the duplicate looks like the edited object, not the original.
    const aiEdit = srcEntry?.aiEdit;
    if (aiEdit != null) {
      const canv = editAssetsRef.current.get(`${id}:${aiEdit}`);
      if (canv) editAssetsRef.current.set(`${dupId}:${aiEdit}`, canv);
    }
    commit((s) => {
      const maxZ = Object.values(s.entries).reduce((m, e) => Math.max(m, e.z || 0), 0);
      return {
        ...s,
        selectedId: dupId,
        entries: { ...s.entries, [dupId]: { objectId: dupId, state: "active", deleted: false, z: maxZ + 1, transform: t, meta, aiEdit } },
      };
    });
    kick();
  }, [session.selectedId, session.entries, om, commit, kick]);

  // ---- Replace Object (AI): open a file picker for the selected object, then
  // POST the upload to the ComfyUI bridge and INTEGRATE the returned patch into
  // that object (composeEdit) so it becomes the object's own bitmap. The rest of
  // the canvas is never touched. ----
  const startReplace = useCallback((id) => {
    if (id == null) return;
    pendingReplaceRef.current = id;
    if (fileInputRef.current) {
      fileInputRef.current.value = ""; // allow re-picking the same file
      fileInputRef.current.click();
    }
  }, []);

  const onReplaceFile = useCallback(
    async (e) => {
      const file = e.target.files && e.target.files[0];
      const id = pendingReplaceRef.current;
      pendingReplaceRef.current = null;
      if (!file || id == null) return;
      const dataUrl = await new Promise((res, rej) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.onerror = rej;
        fr.readAsDataURL(file);
      });
      setReplaceError(null);
      setReplacingId(id);
      try {
        const res = await fetch("/comfyui/replace", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ objectId: id, replacement: dataUrl, harmonize: autoLighting, evaluator: true }),
        });
        if (!res.ok) {
          let msg = `replace failed (${res.status})`;
          try {
            const j = await res.json();
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {}
          throw new Error(msg);
        }
        const p = await res.json(); noteEdit(p);
        const patch = (autoLighting && !p.harmonized) ? await applyLighting(p) : p;
        const img = await decodeImage(patch.png);
        if (!img) throw new Error("could not decode the replacement image");
        await composeEdit(id, img, patch); // the edit BECOMES the object (no overlay)
      } catch (err) {
        console.error("replace failed", err);
        setReplaceError(String(err.message || err));
        setTimeout(() => setReplaceError(null), 6000);
      } finally {
        setReplacingId(null);
      }
    },
    [kick, autoLighting, applyLighting]
  );

  // ---- Background Replacement (AI): upload a new backdrop; the bridge keeps
  // every foreground object and regenerates only the background. The full-canvas
  // result is drawn as the new base. Reuses the shared progress/error UI. ----
  const startBackgroundReplace = useCallback(() => {
    if (bgFileInputRef.current) {
      bgFileInputRef.current.value = "";
      bgFileInputRef.current.click();
    }
  }, []);

  // Shared submit for every background mode (replace/generate/blur/color/remove).
  // The full-canvas result is drawn as the new base. CPU modes (engine "cpu") and
  // already-harmonized SDXL results skip the extra lighting pass.
  const submitBackground = useCallback(
    async (body) => {
      setReplaceError(null);
      setBgBusy(true);
      try {
        const res = await fetch("/comfyui/background", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          let msg = `background failed (${res.status})`;
          try {
            const j = await res.json();
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {}
          throw new Error(msg);
        }
        const p = await res.json(); noteEdit(p);
        const lit = (autoLighting && !p.harmonized && p.engine !== "cpu")
          ? await applyLighting({ x: 0, y: 0, w: imageSize.width, h: imageSize.height, png: p.png }, true)
          : { png: p.png };
        const img = await decodeImage(lit.png);
        if (!img) throw new Error("could not decode the background image");
        setBgResult({ img });
        kick();
      } catch (err) {
        console.error("background failed", err);
        setReplaceError(String(err.message || err));
        setTimeout(() => setReplaceError(null), 6000);
      } finally {
        setBgBusy(false);
      }
    },
    [kick, autoLighting, applyLighting, imageSize]
  );

  const onBackgroundFile = useCallback(
    async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const dataUrl = await new Promise((res, rej) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.onerror = rej;
        fr.readAsDataURL(file);
      });
      submitBackground({ replacement: dataUrl });
    },
    [submitBackground]
  );

  // Background mode shortcuts: blur/remove are one-click, color opens a picker,
  // generate opens a prompt dialog (studio/outdoor presets).
  const blurBackground = useCallback(() => submitBackground({ mode: "blur", blur: 0.6 }), [submitBackground]);
  const removeBackground = useCallback(() => submitBackground({ mode: "remove" }), [submitBackground]);
  const colorBackground = useCallback(() => { if (bgColorInputRef.current) bgColorInputRef.current.click(); }, []);
  const onBgColor = useCallback((e) => submitBackground({ mode: "color", targetColor: e.target.value }), [submitBackground]);
  const generateBackground = useCallback(() => { setBgGenText(""); setBgGenModal(true); }, []);
  const submitBgGenerate = useCallback(() => {
    const prompt = bgGenText.trim();
    setBgGenModal(null);
    submitBackground(prompt ? { mode: "generate", prompt } : { mode: "generate" });
  }, [bgGenText, submitBackground]);

  // ---- Remove Object (AI): erase the selected object and drop the generated
  // background fill onto its footprint (no upload). Reuses the overlay store and
  // the shared progress/error UI. ----
  const startRemove = useCallback(
    async (id) => {
      if (id == null) return;
      setReplaceError(null);
      setRemovingId(id);
      try {
        const res = await fetch("/comfyui/remove", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ objectId: id }),
        });
        if (!res.ok) {
          let msg = `remove failed (${res.status})`;
          try {
            const j = await res.json();
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {}
          throw new Error(msg);
        }
        const p = await res.json(); noteEdit(p);
        const img = await decodeImage(p.png);
        if (!img) throw new Error("could not decode the removal fill");
        await composeEdit(id, img, p); // the erase BECOMES the object (no overlay)
      } catch (err) {
        console.error("remove failed", err);
        setReplaceError(String(err.message || err));
        setTimeout(() => setReplaceError(null), 6000);
      } finally {
        setRemovingId(null);
      }
    },
    [kick]
  );

  // ---- Logo finishes (AI): metallic/glass (SDXL) + emboss/transparent (CPU).
  // Footprint patch dropped into the shared overlay, like Remove/Recolor. ----
  const [logoBusy, setLogoBusy] = useState(false);
  const startLogoEffect = useCallback(
    async (feature, id) => {
      if (id == null) return;
      setReplaceError(null);
      setLogoBusy(true);
      try {
        const res = await fetch("/comfyui/logo", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ objectId: id, feature }),
        });
        if (!res.ok) {
          let msg = `logo ${feature} failed (${res.status})`;
          try {
            const j = await res.json();
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {}
          throw new Error(msg);
        }
        const p = await res.json(); noteEdit(p);
        const img = await decodeImage(p.png);
        if (!img) throw new Error("could not decode the logo effect");
        await composeEdit(id, img, p); // the effect BECOMES the object (no overlay)
      } catch (err) {
        console.error("logo effect failed", err);
        setReplaceError(String(err.message || err));
        setTimeout(() => setReplaceError(null), 6000);
      } finally {
        setLogoBusy(false);
      }
    },
    [kick]
  );

  // ---- Recolor Object (AI): open the color picker for the selected object;
  // on a chosen color, recolor it (texture/lighting preserved) and drop the
  // patch on its footprint. Reuses the overlay store + shared progress/error. ----
  const startRecolor = useCallback((id) => {
    if (id == null || !colorInputRef.current) return;
    pendingRecolorRef.current = id;
    colorInputRef.current.click(); // opens the native color picker
  }, []);

  const onRecolorColor = useCallback(
    async (e) => {
      const hex = e.target.value;
      const id = pendingRecolorRef.current;
      pendingRecolorRef.current = null;
      if (!hex || id == null) return;
      setReplaceError(null);
      setRecoloringId(id);
      try {
        const res = await fetch("/comfyui/recolor", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ objectId: id, targetColor: hex }),
        });
        if (!res.ok) {
          let msg = `recolor failed (${res.status})`;
          try {
            const j = await res.json();
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {}
          throw new Error(msg);
        }
        const p = await res.json(); noteEdit(p);
        const img = await decodeImage(p.png);
        if (!img) throw new Error("could not decode the recolored object");
        await composeEdit(id, img, p); // the recolour BECOMES the object (no overlay)
      } catch (err) {
        console.error("recolor failed", err);
        setReplaceError(String(err.message || err));
        setTimeout(() => setReplaceError(null), 6000);
      } finally {
        setRecoloringId(null);
      }
    },
    [kick]
  );

  // ---- Change Clothes (AI): open a prompt dialog for the selected person; on
  // Generate, regenerate only the clothing region and drop the patch onto it.
  // Reuses the overlay store + shared progress/error UI. ----
  const startChangeClothes = useCallback((id) => {
    if (id == null) return;
    setClothesText("");
    setClothesModal({ id });
  }, []);

  const submitClothes = useCallback(async () => {
    const id = clothesModal?.id;
    const prompt = clothesText.trim();
    if (id == null || !prompt) return;
    setClothesModal(null);
    setReplaceError(null);
    setClothesBusy(true);
    try {
      const res = await fetch("/comfyui/clothes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objectId: id, prompt, intensity: aiIntensity / 100, harmonize: autoLighting, evaluator: true }),
      });
      if (!res.ok) {
        let msg = `change clothes failed (${res.status})`;
        try {
          const j = await res.json();
          msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        } catch {}
        throw new Error(msg);
      }
      const p = await res.json(); noteEdit(p);
      let patch = preserveIdentity ? await applyIdentity(p, cropRegion(p.x, p.y, p.w, p.h)) : p;
      if (autoLighting && !p.harmonized) patch = await applyLighting(patch);
      const img = await decodeImage(patch.png);
      if (!img) throw new Error("could not decode the new clothing");
      await composeEdit(id, img, patch); // new clothes composited onto the person (no overlay)
    } catch (err) {
      console.error("change clothes failed", err);
      setReplaceError(String(err.message || err));
      setTimeout(() => setReplaceError(null), 6000);
    } finally {
      setClothesBusy(false);
    }
  }, [clothesModal, clothesText, aiIntensity, kick, preserveIdentity, applyIdentity, cropRegion, autoLighting, applyLighting]);

  // ---- Change Hair (AI): open a prompt dialog for the selected person; on
  // Generate, regenerate only the hair region and drop the patch onto it.
  // Reuses the overlay store + shared progress/error UI. ----
  const startChangeHair = useCallback((id) => {
    if (id == null) return;
    setHairText("");
    setHairModal({ id });
  }, []);

  const submitHair = useCallback(async () => {
    const id = hairModal?.id;
    const prompt = hairText.trim();
    if (id == null || !prompt) return;
    setHairModal(null);
    setReplaceError(null);
    setHairBusy(true);
    try {
      const res = await fetch("/comfyui/hair", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objectId: id, prompt, intensity: aiIntensity / 100, harmonize: autoLighting, evaluator: true }),
      });
      if (!res.ok) {
        let msg = `change hair failed (${res.status})`;
        try {
          const j = await res.json();
          msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        } catch {}
        throw new Error(msg);
      }
      const p = await res.json(); noteEdit(p);
      let patch = preserveIdentity ? await applyIdentity(p, cropRegion(p.x, p.y, p.w, p.h)) : p;
      if (autoLighting && !p.harmonized) patch = await applyLighting(patch);
      const img = await decodeImage(patch.png);
      if (!img) throw new Error("could not decode the new hair");
      await composeEdit(id, img, patch); // new hair composited onto the person (no overlay)
    } catch (err) {
      console.error("change hair failed", err);
      setReplaceError(String(err.message || err));
      setTimeout(() => setReplaceError(null), 6000);
    } finally {
      setHairBusy(false);
    }
  }, [hairModal, hairText, aiIntensity, kick, preserveIdentity, applyIdentity, cropRegion, autoLighting, applyLighting]);

  // ---- Person Replace (AI): open a dialog offering "upload image" or "describe
  // a person"; either path replaces the whole person within its footprint.
  // Reuses the overlay store + shared progress/error UI. ----
  const startReplacePerson = useCallback((id) => {
    if (id == null) return;
    setPersonText("");
    setPersonModal({ id });
  }, []);

  const runPerson = useCallback(
    async (body) => {
      setPersonModal(null);
      setReplaceError(null);
      setPersonBusy(true);
      try {
        const res = await fetch("/comfyui/person", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...body, harmonize: autoLighting, evaluator: true }),
        });
        if (!res.ok) {
          let msg = `person replace failed (${res.status})`;
          try {
            const j = await res.json();
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {}
          throw new Error(msg);
        }
        const p = await res.json(); noteEdit(p);
        const ref = body.image || cropRegion(p.x, p.y, p.w, p.h);
        let patch = preserveIdentity ? await applyIdentity(p, ref) : p;
        if (autoLighting && !p.harmonized) patch = await applyLighting(patch);
        const img = await decodeImage(patch.png);
        if (!img) throw new Error("could not decode the replacement person");
        await composeEdit(body.objectId, img, patch); // the new person BECOMES the object (no overlay)
      } catch (err) {
        console.error("person replace failed", err);
        setReplaceError(String(err.message || err));
        setTimeout(() => setReplaceError(null), 6000);
      } finally {
        setPersonBusy(false);
      }
    },
    [kick, preserveIdentity, applyIdentity, cropRegion, autoLighting, applyLighting]
  );

  const submitPersonPrompt = useCallback(() => {
    const id = personModal?.id;
    const prompt = personText.trim();
    if (id == null || !prompt) return;
    runPerson({ objectId: id, prompt });
  }, [personModal, personText, runPerson]);

  const startPersonUpload = useCallback(() => {
    if (personModal?.id == null || !personFileInputRef.current) return;
    pendingPersonRef.current = personModal.id;
    personFileInputRef.current.value = "";
    personFileInputRef.current.click();
  }, [personModal]);

  const onPersonFile = useCallback(
    async (e) => {
      const file = e.target.files && e.target.files[0];
      const id = pendingPersonRef.current;
      pendingPersonRef.current = null;
      if (!file || id == null) return;
      const dataUrl = await new Promise((res, rej) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.onerror = rej;
        fr.readAsDataURL(file);
      });
      runPerson({ objectId: id, image: dataUrl });
    },
    [runPerson]
  );

  // ---- Face edits (Beard / Smile / Age / Glasses / Skin) — one-click, via the
  // shared engine. Like Remove/Recolor: no modal; the returned patch is stored in
  // the overlay store keyed by feature (so several face edits coexist) and reuses
  // the shared progress overlay + error toast + the lighting/identity hooks. ----
  const [faceBusy, setFaceBusy] = useState(false);
  const [faceModal, setFaceModal] = useState(null);   // { feature, id } — preset+intensity tool
  const [faceIntensity, setFaceIntensity] = useState(50);
  const startFaceEdit = useCallback(
    async (feature, id, opts = {}) => {
      if (id == null) return;
      setReplaceError(null);
      setFaceBusy(true);
      try {
        const res = await fetch("/comfyui/face", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ objectId: id, feature, ...opts, harmonize: autoLighting, evaluator: true }),
        });
        if (!res.ok) {
          let msg = `face edit failed (${res.status})`;
          try {
            const j = await res.json();
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {}
          throw new Error(msg);
        }
        const p = await res.json(); noteEdit(p);
        let patch = preserveIdentity ? await applyIdentity(p, cropRegion(p.x, p.y, p.w, p.h)) : p;
        if (autoLighting && !p.harmonized) patch = await applyLighting(patch);
        const img = await decodeImage(patch.png);
        if (!img) throw new Error("could not decode the face edit");
        replaceAssetsRef.current.set(`face_${feature}:${p.faceId ?? id}`, { img, x: patch.x, y: patch.y, w: patch.w, h: patch.h });
        kick();
      } catch (err) {
        console.error("face edit failed", err);
        setReplaceError(String(err.message || err));
        setTimeout(() => setReplaceError(null), 6000);
      } finally {
        setFaceBusy(false);
      }
    },
    [preserveIdentity, applyIdentity, cropRegion, autoLighting, applyLighting, kick]
  );

  // ---- property inspector: dispatch its REAL actions to existing handlers
  // (placeholder/AI actions are handled inside the panel itself) ----
  const handlePanelAction = useCallback(
    (id) => {
      const sel = session.selectedId;
      if (id === "replaceBg") return startBackgroundReplace(); // background needs no selection
      if (id === "blurBg") return blurBackground();
      if (id === "colorBg") return colorBackground();
      if (id === "generateBg") return generateBackground();
      if (id === "removeBg") return removeBackground();
      if (sel == null) return;
      if (id === "replaceText") startTextEdit(sel);
      else if (id === "front") bringToFront();
      else if (id === "back") sendToBack();
      else if (id === "delete") deleteSelected();
      else if (id === "duplicate") duplicateSelected();
      else if (id === "replace") startReplace(sel);
      else if (id === "remove") startRemove(sel);
      else if (id === "recolor") startRecolor(sel);
      else if (id === "clothes") startChangeClothes(sel);
      else if (id === "hair") startChangeHair(sel);
      else if (id === "replacePerson") startReplacePerson(sel);
      // Face panel features → open the preset + intensity tool (shared face engine).
      else if (["smile", "expression", "beard", "glasses", "skinTone", "faceAge", "age"].includes(id)) {
        const feature = id === "skinTone" ? "skin" : (id === "faceAge" || id === "age") ? "age" : id === "expression" ? "smile" : id;
        setFaceIntensity(50);
        setFaceModal({ feature, id: sel });
      }
      // "Hair" in the face panel edits the hair of the face's person (prompt modal).
      else if (id === "faceHair") { const par = om?.getParent?.(sel); startChangeHair(par ? par.id : sel); }
      // Logo panel reuses the object Replace (upload) + Recolor flows.
      else if (id === "replaceLogo" || id === "uploadLogo") startReplace(sel);
      else if (id === "logoColor") startRecolor(sel);
      else if (id === "logoMetallic") startLogoEffect("metallic", sel);
      else if (id === "logoGlass") startLogoEffect("glass", sel);
      else if (id === "logoEmboss") startLogoEffect("emboss", sel);
      else if (id === "logoTransparent") startLogoEffect("transparent", sel);
    },
    [session.selectedId, startTextEdit, bringToFront, sendToBack, deleteSelected, duplicateSelected, startReplace, startBackgroundReplace, blurBackground, colorBackground, generateBackground, removeBackground, startRemove, startRecolor, startChangeClothes, startChangeHair, startReplacePerson, startFaceEdit, startLogoEffect, om]
  );

  // ---- live property controls (opacity / rotation / text style overrides) ----
  // One history checkpoint at interaction start, then live `update`s (no per-tick
  // history). Overrides live on the session entry (no OM/typography change): the
  // object is activated so its fill is in place and the change actually renders.
  const controlCheckpoint = useCallback(() => {
    setHistory((h) => ({ past: [...h.past, h.present], present: h.present, future: [] }));
  }, []);
  const controlChange = useCallback(
    (patch) => {
      const id = session.selectedId;
      if (id == null || !om) return;
      if (om.getObject(id)) prefetchLift(id); // dupes need no fill
      update((s) => {
        const base = s.entries[id]?.state === "active" ? s : om.activate(s, id).session;
        if ("rotation" in patch) return om.applyTransform(base, id, { rotation: patch.rotation });
        const e = base.entries[id];
        if ("opacity" in patch) {
          const op = Math.max(0, Math.min(1, patch.opacity / 100));
          return { ...base, entries: { ...base.entries, [id]: { ...e, opacity: op } } };
        }
        const styled = { ...e, style: { ...(e.style || {}), ...patch } };
        // A typographic style change on an OCR text object must take effect
        // WITHOUT first pressing "Replace text". The synthesized-text renderer
        // only engages once the entry carries a string, so seed it with the
        // ORIGINAL OCR text the first time the user styles it. Appearance is
        // preserved (estimateTypography matches the bitmap; only the touched
        // property changes) and the content is unchanged — Replace text remains
        // the only way to alter the actual words.
        const o = om.getObject(id);
        if (o && styled.text == null && o.text && categoryGroup(o) === "text") {
          styled.text = o.text;
        }
        return { ...base, entries: { ...base.entries, [id]: styled } };
      });
      kick();
    },
    [session.selectedId, om, update, prefetchLift, kick]
  );

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
  let selected = session.selectedId != null && om ? om.getObject(session.selectedId) : null;
  // a selected DUPLICATE resolves from its session entry's carried metadata
  const selDupEntry = !selected && session.selectedId != null ? session.entries[session.selectedId] : null;
  const isDupeSel = !!selDupEntry?.meta;
  if (isDupeSel) {
    selected = { id: session.selectedId, ...selDupEntry.meta, rotation: selDupEntry.transform?.rotation ?? 0 };
  }
  const selectedLifted = selected ? (isDupeSel ? true : !!liftAssetsRef.current.get(selected.id)?.fill) : false;

  // Typography estimate for the SELECTED text → the inspector's default control
  // values (so Font/Size/Weight/Color start at the auto-matched values). Same
  // pure function the layer renders with, so the panel and canvas agree.
  const selIsText = !!(selected && (selected.role === "text" || categoryGroup(selected) === "text"));
  const selFile = selIsText ? `/layers/${encodeURIComponent(selected.file || "")}` : "";
  const [selBitmap] = useImage(selFile || undefined);
  const selText = selected ? (session.entries[selected.id]?.text ?? selected.text) : "";
  const selSynth = useMemo(
    () =>
      selIsText && selBitmap
        ? estimateTypography(selBitmap, {
            text: selText || "",
            refText: selected.text,
            width: selected.bbox.w,
            height: selected.bbox.h,
            fallbackFamily: selected.style?.fontFamily,
            fallbackColor: selected.style?.fontColor,
          })
        : null,
    [selIsText, selBitmap, selText, selected]
  );

  // current values for the inspector controls: user override → typography → metadata
  const selEntry = selected ? session.entries[selected.id] : null;
  const ov = selEntry?.style || {};
  const panelValues = selected
    ? {
        fontFamily: ov.fontFamily ?? selSynth?.fontFamily ?? selected.style?.fontFamily ?? "serif",
        fontSize: Math.round(ov.fontSize ?? selSynth?.fontSize ?? selected.bbox.h * 0.6),
        fontStyle: ov.fontStyle ?? selSynth?.fontStyle ?? "bold",
        fill: ov.fill ?? selSynth?.fill ?? selected.style?.fontColor ?? "#d8b36a",
        letterSpacing: Math.round(ov.letterSpacing ?? selSynth?.letterSpacing ?? 0),
        lineHeight: ov.lineHeight ?? 1,
        // text style extras (Outline / Shadow / Alignment / Gradient)
        stroke: ov.stroke ?? selSynth?.stroke ?? selected.style?.strokeColor ?? "#5a2e12",
        strokeWidth: ov.strokeWidth ?? selSynth?.strokeWidth ?? 1,
        shadowBlur: ov.shadowBlur ?? 0,
        align: ov.align ?? "center",
        gradient: ov.gradient ?? null,
        opacity: Math.round((selEntry?.opacity ?? 1) * 100),
        rotation: Math.round(selEntry?.transform?.rotation ?? selected.rotation ?? 0),
      }
    : null;

  // Render the SELECTED object even when it has not been lifted, so it can show
  // its silhouette glow and be grabbed — drawn from its ORIGINAL pixels in place
  // (identical to the base), with no fill behind it (the base still owns those
  // pixels). It joins the lifted objects from the scene.
  // duplicated instances — synthetic entries carrying `meta`; additive copies
  // rendered through the same layer (no /lift fill needed)
  const dupeVisuals = [];
  for (const key in session.entries) {
    const e = session.entries[key];
    if (e.meta && e.state === "active" && !e.deleted) {
      dupeVisuals.push({
        objectId: key,
        file: e.meta.file,
        isText: e.meta.role === "text",
        category: e.meta.category,
        text: e.text ?? e.meta.text,
        style: e.meta.style,
        transform: e.transform,
        z: e.z || 0,
        isDupe: true,
      });
    }
  }
  const renderList = [...scene.activeVisuals, ...dupeVisuals].sort((a, b) => (a.z || 0) - (b.z || 0));
  if (selected && vr && !isDupeSel) {
    const selEntry = session.entries[selected.id];
    if ((!selEntry || selEntry.state !== "active") && !selEntry?.deleted) {
      const sv = vr.resolve(selected.id, session);
      if (sv) renderList.push(sv); // resting-selected highlight on top
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
  // dissolving objects (DELETING, e.g. deleting a never-lifted selection): paint
  // the fill UNDER the fading cutout so the footprint is covered as the object
  // dissolves into the background — a clean fade-out, never a pop or a hole.
  for (const id of eie.ids()) {
    const fr = eie.frame(id, now);
    if (fr && fr.phase === PHASE.DELETING) {
      const a = liftAssetsRef.current.get(id);
      if (a && a.fill && !fills.some((f) => f.id === id)) fills.push({ id, ...a.fill });
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

  // AI face patches — the ONLY remaining top-most overlay (face sub-region edits
  // are the deferred follow-up). Every object-level edit now renders THROUGH its
  // object's EditableLayer (integrated, moves with the object), so it is NOT here.
  const replacements = [];
  for (const [id, r] of replaceAssetsRef.current) {
    if (r && r.img && String(id).startsWith("face_")) replacements.push({ id, ...r });
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

  // property inspector anchor — screen-space rect of the selected object so the
  // floating panel can sit beside it without covering it
  let inspectorAnchor = null;
  if (selected && !editingId && stageRef.current) {
    const entry = session.entries[selected.id];
    const t = entry?.transform || { x: selected.bbox.x, y: selected.bbox.y, width: selected.bbox.w, height: selected.bbox.h };
    const cont = stageRef.current.container().getBoundingClientRect();
    inspectorAnchor = {
      left: cont.left + t.x * scale,
      top: cont.top + t.y * scale,
      width: t.width * scale,
      height: t.height * scale,
    };
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
      {/* Hidden picker for Replace Object (AI) */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={onReplaceFile}
      />
      {/* Hidden picker for Background Replacement (AI) */}
      <input
        ref={bgFileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={onBackgroundFile}
      />
      {/* Native color picker for Recolor (AI) — visually hidden but clickable so
          .click() opens the OS picker. */}
      <input
        ref={colorInputRef}
        type="color"
        defaultValue="#d8b36a"
        onChange={onRecolorColor}
        style={{ position: "fixed", left: -9999, top: 0, width: 1, height: 1, opacity: 0, pointerEvents: "none" }}
        tabIndex={-1}
        aria-hidden="true"
      />
      {/* Native color picker for Background → Color */}
      <input
        ref={bgColorInputRef}
        type="color"
        defaultValue="#1565ff"
        onChange={onBgColor}
        style={{ position: "fixed", left: -9999, top: 0, width: 1, height: 1, opacity: 0, pointerEvents: "none" }}
        tabIndex={-1}
        aria-hidden="true"
      />
      {/* Hidden picker for Person Replace upload (AI) */}
      <input
        ref={personFileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={onPersonFile}
      />

      {/* Shared AI in-flight overlay (SDXL render takes a few seconds) */}
      {(replacingId != null || bgBusy || removingId != null || recoloringId != null || clothesBusy || hairBusy || personBusy || faceBusy || logoBusy) && (
        <div
          style={{
            position: "absolute", inset: 0, zIndex: 40, display: "flex",
            alignItems: "center", justifyContent: "center",
            background: "rgba(10,10,12,0.45)", backdropFilter: "blur(2px)",
          }}
        >
          <div
            style={{
              padding: "14px 20px", borderRadius: 12, background: "rgba(24,24,28,0.9)",
              border: "1px solid rgba(216,179,106,0.35)", color: "#f0ead9",
              font: "14px/1.4 -apple-system, Segoe UI, system-ui, sans-serif",
              boxShadow: "0 12px 40px rgba(0,0,0,0.5)",
            }}
          >
            <span style={{ color: "#d8b36a" }}>✦</span> {bgBusy ? "Replacing background with AI…" : removingId != null ? "Removing object with AI…" : recoloringId != null ? "Recoloring object with AI…" : clothesBusy ? "Changing clothes with AI…" : hairBusy ? "Changing hair with AI…" : personBusy ? "Replacing person with AI…" : faceBusy ? "Editing face with AI…" : logoBusy ? "Applying logo finish with AI…" : "Replacing object with AI…"} <span style={{ opacity: 0.6 }}>this can take a few seconds</span>
          </div>
        </div>
      )}

      {/* Replace error toast */}
      {replaceError && (
        <div
          style={{
            position: "absolute", bottom: 18, left: "50%", transform: "translateX(-50%)", zIndex: 41,
            maxWidth: "70vw", padding: "10px 14px", borderRadius: 10, background: "rgba(60,20,20,0.92)",
            border: "1px solid rgba(255,120,100,0.4)", color: "#ffd9d2",
            font: "13px/1.4 -apple-system, Segoe UI, system-ui, sans-serif",
          }}
        >
          Replace failed: {replaceError}
        </div>
      )}

      {/* Transient AI readout: what the planner detected + the self-eval score */}
      {lastEditInfo && !replaceError && (
        <div
          style={{
            position: "absolute", bottom: 18, left: "50%", transform: "translateX(-50%)", zIndex: 41,
            maxWidth: "70vw", padding: "8px 13px", borderRadius: 999, background: "rgba(20,24,20,0.9)",
            border: "1px solid rgba(216,179,106,0.35)", color: "#e9e1cc",
            font: "12px/1.3 -apple-system, Segoe UI, system-ui, sans-serif",
          }}
        >
          <span style={{ color: "#d8b36a" }}>✦ AI</span> {lastEditInfo}
        </div>
      )}

      {/* Face AI tool — presets + intensity (Phase 12) */}
      {faceModal && FACE_PRESETS[faceModal.feature] && (
        <div
          style={{ position: "absolute", inset: 0, zIndex: 42, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(10,10,12,0.5)", backdropFilter: "blur(3px)" }}
          onClick={() => setFaceModal(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{ width: 380, maxWidth: "86vw", padding: 18, borderRadius: 16, background: "rgba(24,24,28,0.92)", border: "1px solid rgba(255,255,255,0.12)", boxShadow: "0 16px 48px rgba(0,0,0,0.5)", color: "#f0ead9", font: "13px/1.4 -apple-system, Segoe UI, system-ui, sans-serif" }}
          >
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
              <span style={{ color: "#d8b36a" }}>{FACE_PRESETS[faceModal.feature].icon}</span> {FACE_PRESETS[faceModal.feature].title}
            </div>
            <div style={{ opacity: 0.6, marginBottom: 12 }}>
              Pick a style — identity, lighting and the rest of the face are preserved.
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, opacity: 0.92 }}>
              <span>Intensity</span><span style={{ opacity: 0.6, fontVariantNumeric: "tabular-nums" }}>{faceIntensity}%</span>
            </div>
            <input
              type="range" min={10} max={100} step={1} value={faceIntensity}
              onChange={(e) => setFaceIntensity(Number(e.target.value))}
              style={{ width: "100%", accentColor: "#d8b36a", cursor: "pointer", marginBottom: 12 }}
            />
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {FACE_PRESETS[faceModal.feature].presets.map(([label, prompt, direction]) => (
                <button
                  key={label}
                  onClick={() => { const m = faceModal; setFaceModal(null); startFaceEdit(m.feature, m.id, { prompt, intensity: faceIntensity / 100, ...(direction ? { direction } : {}) }); }}
                  style={{ flex: "1 1 44%", padding: "9px 10px", borderRadius: 9, border: "1px solid rgba(255,255,255,0.14)", background: "rgba(255,255,255,0.06)", color: "#f0ead9", cursor: "pointer", font: "inherit", fontWeight: 600 }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(216,179,106,0.18)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.06)")}
                >
                  {label}
                </button>
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 14 }}>
              <button onClick={() => setFaceModal(null)} style={{ padding: "7px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.05)", color: "#f0ead9", cursor: "pointer", font: "inherit" }}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Change Clothes (AI) — prompt dialog */}
      {clothesModal && (
        <div
          style={{
            position: "absolute", inset: 0, zIndex: 42, display: "flex",
            alignItems: "center", justifyContent: "center",
            background: "rgba(10,10,12,0.5)", backdropFilter: "blur(3px)",
          }}
          onClick={() => setClothesModal(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 380, maxWidth: "86vw", padding: 18, borderRadius: 16,
              background: "rgba(24,24,28,0.92)", border: "1px solid rgba(255,255,255,0.12)",
              boxShadow: "0 16px 48px rgba(0,0,0,0.5)", color: "#f0ead9",
              font: "13px/1.4 -apple-system, Segoe UI, system-ui, sans-serif",
            }}
          >
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
              <span style={{ color: "#d8b36a" }}>👕</span> Change clothes
            </div>
            <div style={{ opacity: 0.6, marginBottom: 10 }}>
              Describe the new clothing — the face, hair and pose are preserved.
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
              {CLOTHES_PRESETS.map(([label, p]) => (
                <button key={label} type="button" onClick={() => setClothesText(p)} style={presetChip}>{label}</button>
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, opacity: 0.92 }}>
              <span>Intensity</span><span style={{ opacity: 0.6, fontVariantNumeric: "tabular-nums" }}>{aiIntensity}%</span>
            </div>
            <input
              type="range" min={10} max={100} step={1} value={aiIntensity}
              onChange={(e) => setAiIntensity(Number(e.target.value))}
              style={{ width: "100%", accentColor: "#d8b36a", cursor: "pointer", marginBottom: 10 }}
            />
            <input
              autoFocus
              value={clothesText}
              onChange={(e) => setClothesText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submitClothes();
                else if (e.key === "Escape") setClothesModal(null);
              }}
              placeholder="e.g. a blue denim jacket"
              style={{
                width: "100%", padding: "9px 11px", borderRadius: 9, boxSizing: "border-box",
                border: "1px solid rgba(255,255,255,0.18)", background: "rgba(0,0,0,0.3)",
                color: "#f0ead9", font: "inherit", outline: "none",
              }}
            />
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 14 }}>
              <button
                onClick={() => setClothesModal(null)}
                style={{ padding: "7px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.05)", color: "#f0ead9", cursor: "pointer", font: "inherit" }}
              >
                Cancel
              </button>
              <button
                onClick={submitClothes}
                disabled={!clothesText.trim()}
                style={{ padding: "7px 16px", borderRadius: 8, border: "none", background: clothesText.trim() ? "#d8b36a" : "#5a5039", color: "#1a1a1a", fontWeight: 600, cursor: clothesText.trim() ? "pointer" : "default", font: "inherit", opacity: clothesText.trim() ? 1 : 0.6 }}
              >
                Generate
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Background → Generate (AI) — prompt dialog */}
      {bgGenModal && (
        <div
          style={{
            position: "absolute", inset: 0, zIndex: 42, display: "flex",
            alignItems: "center", justifyContent: "center",
            background: "rgba(10,10,12,0.5)", backdropFilter: "blur(3px)",
          }}
          onClick={() => setBgGenModal(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 380, maxWidth: "86vw", padding: 18, borderRadius: 16,
              background: "rgba(24,24,28,0.92)", border: "1px solid rgba(255,255,255,0.12)",
              boxShadow: "0 16px 48px rgba(0,0,0,0.5)", color: "#f0ead9",
              font: "13px/1.4 -apple-system, Segoe UI, system-ui, sans-serif",
            }}
          >
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
              <span style={{ color: "#d8b36a" }}>✨</span> Generate background
            </div>
            <div style={{ opacity: 0.6, marginBottom: 10 }}>
              Describe a new backdrop — every foreground object is preserved.
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
              {BG_GEN_PRESETS.map(([label, p]) => (
                <button key={label} type="button" onClick={() => setBgGenText(p)} style={presetChip}>{label}</button>
              ))}
            </div>
            <input
              autoFocus
              value={bgGenText}
              onChange={(e) => setBgGenText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submitBgGenerate();
                else if (e.key === "Escape") setBgGenModal(null);
              }}
              placeholder="e.g. a sunlit garden, soft depth of field"
              style={{
                width: "100%", padding: "9px 11px", borderRadius: 9, boxSizing: "border-box",
                border: "1px solid rgba(255,255,255,0.18)", background: "rgba(0,0,0,0.3)",
                color: "#f0ead9", font: "inherit", outline: "none",
              }}
            />
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 14 }}>
              <button
                onClick={() => setBgGenModal(null)}
                style={{ padding: "7px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.05)", color: "#f0ead9", cursor: "pointer", font: "inherit" }}
              >
                Cancel
              </button>
              <button
                onClick={submitBgGenerate}
                style={{ padding: "7px 16px", borderRadius: 8, border: "none", background: "#d8b36a", color: "#1a1a1a", fontWeight: 600, cursor: "pointer", font: "inherit" }}
              >
                Generate
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Change Hair (AI) — prompt dialog */}
      {hairModal && (
        <div
          style={{
            position: "absolute", inset: 0, zIndex: 42, display: "flex",
            alignItems: "center", justifyContent: "center",
            background: "rgba(10,10,12,0.5)", backdropFilter: "blur(3px)",
          }}
          onClick={() => setHairModal(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 380, maxWidth: "86vw", padding: 18, borderRadius: 16,
              background: "rgba(24,24,28,0.92)", border: "1px solid rgba(255,255,255,0.12)",
              boxShadow: "0 16px 48px rgba(0,0,0,0.5)", color: "#f0ead9",
              font: "13px/1.4 -apple-system, Segoe UI, system-ui, sans-serif",
            }}
          >
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
              <span style={{ color: "#d8b36a" }}>💇</span> Change hair
            </div>
            <div style={{ opacity: 0.6, marginBottom: 10 }}>
              Describe the new hairstyle or color — the face, eyes and ears are preserved.
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
              {HAIR_PRESETS.map(([label, p]) => (
                <button key={label} type="button" onClick={() => setHairText(p)} style={presetChip}>{label}</button>
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, opacity: 0.92 }}>
              <span>Intensity</span><span style={{ opacity: 0.6, fontVariantNumeric: "tabular-nums" }}>{aiIntensity}%</span>
            </div>
            <input
              type="range" min={10} max={100} step={1} value={aiIntensity}
              onChange={(e) => setAiIntensity(Number(e.target.value))}
              style={{ width: "100%", accentColor: "#d8b36a", cursor: "pointer", marginBottom: 10 }}
            />
            <input
              autoFocus
              value={hairText}
              onChange={(e) => setHairText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submitHair();
                else if (e.key === "Escape") setHairModal(null);
              }}
              placeholder="e.g. long blonde wavy hair"
              style={{
                width: "100%", padding: "9px 11px", borderRadius: 9, boxSizing: "border-box",
                border: "1px solid rgba(255,255,255,0.18)", background: "rgba(0,0,0,0.3)",
                color: "#f0ead9", font: "inherit", outline: "none",
              }}
            />
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 14 }}>
              <button
                onClick={() => setHairModal(null)}
                style={{ padding: "7px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.05)", color: "#f0ead9", cursor: "pointer", font: "inherit" }}
              >
                Cancel
              </button>
              <button
                onClick={submitHair}
                disabled={!hairText.trim()}
                style={{ padding: "7px 16px", borderRadius: 8, border: "none", background: hairText.trim() ? "#d8b36a" : "#5a5039", color: "#1a1a1a", fontWeight: 600, cursor: hairText.trim() ? "pointer" : "default", font: "inherit", opacity: hairText.trim() ? 1 : 0.6 }}
              >
                Generate
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Person Replace (AI) — dual-option dialog: upload an image OR describe a person */}
      {personModal && (
        <div
          style={{
            position: "absolute", inset: 0, zIndex: 42, display: "flex",
            alignItems: "center", justifyContent: "center",
            background: "rgba(10,10,12,0.5)", backdropFilter: "blur(3px)",
          }}
          onClick={() => setPersonModal(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 380, maxWidth: "86vw", padding: 18, borderRadius: 16,
              background: "rgba(24,24,28,0.92)", border: "1px solid rgba(255,255,255,0.12)",
              boxShadow: "0 16px 48px rgba(0,0,0,0.5)", color: "#f0ead9",
              font: "13px/1.4 -apple-system, Segoe UI, system-ui, sans-serif",
            }}
          >
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
              <span style={{ color: "#d8b36a" }}>🔁</span> Replace person
            </div>
            <div style={{ opacity: 0.6, marginBottom: 12 }}>
              Describe a new person, or upload an image. The background and pose are preserved.
            </div>
            <input
              autoFocus
              value={personText}
              onChange={(e) => setPersonText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submitPersonPrompt();
                else if (e.key === "Escape") setPersonModal(null);
              }}
              placeholder="e.g. a young queen in a green gown"
              style={{
                width: "100%", padding: "9px 11px", borderRadius: 9, boxSizing: "border-box",
                border: "1px solid rgba(255,255,255,0.18)", background: "rgba(0,0,0,0.3)",
                color: "#f0ead9", font: "inherit", outline: "none",
              }}
            />
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 10 }}>
              <button
                onClick={submitPersonPrompt}
                disabled={!personText.trim()}
                style={{ padding: "7px 16px", borderRadius: 8, border: "none", background: personText.trim() ? "#d8b36a" : "#5a5039", color: "#1a1a1a", fontWeight: 600, cursor: personText.trim() ? "pointer" : "default", font: "inherit", opacity: personText.trim() ? 1 : 0.6 }}
              >
                Generate
              </button>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "14px 0 10px", opacity: 0.5 }}>
              <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.15)" }} />
              <span style={{ fontSize: 11 }}>OR</span>
              <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.15)" }} />
            </div>
            <button
              onClick={startPersonUpload}
              style={{ width: "100%", padding: "9px 12px", borderRadius: 9, border: "1px solid rgba(255,255,255,0.18)", background: "rgba(255,255,255,0.06)", color: "#f0ead9", cursor: "pointer", font: "inherit" }}
            >
              ⤴ Upload person image
            </button>
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 12 }}>
              <button
                onClick={() => setPersonModal(null)}
                style={{ padding: "7px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.05)", color: "#f0ead9", cursor: "pointer", font: "inherit" }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div style={{ ...chip, position: "absolute", top: 12, left: 12, zIndex: 10, pointerEvents: "none" }}>
        {selected ? `selected: ${selected.category}#${selected.id}` : "click an object to select it"}
        {selected ? (selectedLifted ? "  ·  drag to move" : "  ·  preparing…") : ""}
        {selected?.role === "text" ? "  ·  double-click to edit" : ""}
        {`  ·  lift: ${liftEngine || "offline"}`}
      </div>

      <div style={{ position: "absolute", top: 12, right: 12, zIndex: 10, display: "flex", gap: 8 }}>
        {/* Identity Preservation (AI) — one optional toggle, OFF by default. */}
        <button
          onClick={() => setPreserveIdentity((v) => !v)}
          title="Preserve facial identity for Change Hair, Change Clothes and Person Replace"
          style={{
            padding: "7px 12px", borderRadius: 6, border: "1px solid " + (preserveIdentity ? "#d8b36a" : "rgba(255,255,255,0.15)"),
            background: preserveIdentity ? "rgba(216,179,106,0.18)" : "rgba(255,255,255,0.06)",
            color: preserveIdentity ? "#e6cf9c" : "#cfc7b5", fontWeight: 600, cursor: "pointer",
            font: "13px -apple-system, Segoe UI, system-ui, sans-serif",
          }}
        >
          {preserveIdentity ? "◉" : "◯"} Preserve Identity
        </button>
        {/* Lighting & Shadow Harmonization (AI) — one optional toggle, OFF by default. */}
        <button
          onClick={() => setAutoLighting((v) => !v)}
          title="Harmonize lighting & shadows after Replace, Hair, Clothes, Background and Person edits"
          style={{
            padding: "7px 12px", borderRadius: 6, border: "1px solid " + (autoLighting ? "#d8b36a" : "rgba(255,255,255,0.15)"),
            background: autoLighting ? "rgba(216,179,106,0.18)" : "rgba(255,255,255,0.06)",
            color: autoLighting ? "#e6cf9c" : "#cfc7b5", fontWeight: 600, cursor: "pointer",
            font: "13px -apple-system, Segoe UI, system-ui, sans-serif",
          }}
        >
          {autoLighting ? "◉" : "◯"} Auto Lighting
        </button>
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

      {/* Context-aware property inspector — floats beside the selected object,
          contents driven by its semantic category (auto-hides when nothing is
          selected or while inline-editing text). */}
      {selected && !editingId && inspectorAnchor && (
        <PropertyInspector
          key={selected.id}
          object={selected}
          panel={getActions(categoryGroup(selected))}
          values={panelValues}
          anchorRect={inspectorAnchor}
          viewport={{ w: viewport.w, h: viewport.h }}
          onAction={handlePanelAction}
          onControlStart={controlCheckpoint}
          onControlChange={controlChange}
        />
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

          {/* AI BACKGROUND REPLACEMENT — full canvas (new backdrop + preserved
              foreground), drawn over the base so the new background shows; fills
              and objects still render on top. */}
          {bgResult && (
            <Image image={bgResult.img} x={0} y={0} width={imageSize.width} height={imageSize.height} listening={false} />
          )}

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
            // integrated AI edit: the versioned object-sized canvas this object
            // currently resolves to (null when unedited or a text object)
            const editImg =
              !v.isText && entry?.aiEdit != null
                ? editAssetsRef.current.get(`${v.objectId}:${entry.aiEdit}`) || null
                : null;
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
            // user property-inspector overrides (live, stored on the session entry)
            const userOpacity = entry?.opacity ?? 1;
            const styleOverride = entry?.style || null;
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
                manipulable={v.isDupe ? true : fillReady}
                glow={fr ? fr.selection.glow : session.selectedId === v.objectId ? 1 : 0}
                opacity={opacity * userOpacity}
                cutoutImg={!v.isText && isActive && a && a.cutout ? a.cutout.img : null}
                editImg={editImg}
                refText={v.isText ? om.getObject(v.objectId)?.text : null}
                styleOverride={styleOverride}
                onChange={(attrs) => handleObjectChange(v.objectId, attrs, v.text)}
                onStartTextEdit={startTextEdit}
                onLiftStart={liftStart}
                onSelect={v.isDupe ? selectDupe : undefined}
              />
            );
          })}

          {/* AI REPLACEMENTS — top-most patches that swap an object's pixels for
              the uploaded replacement, footprint-shaped so only that layer
              changes. listening=false so the underlying object stays selectable. */}
          {replacements.map((r) => (
            <Image key={`replace-${r.id}`} image={r.img} x={r.x} y={r.y} width={r.w} height={r.h} listening={false} />
          ))}
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
