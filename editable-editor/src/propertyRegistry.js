/**
 * Property registry — maps an object's SEMANTIC CATEGORY (from the extraction
 * metadata, never an id) to the set of actions/controls its context panel
 * offers.
 *
 *   selection → categoryGroup(object) → getActions(group) → context panel
 *
 *   kind: "control"  a live editor control (slider / dropdown / colour / weight)
 *                    wired to a real style/transform field.
 *   kind: "real"     a button wired to an existing editor handler.
 *   kind: "soon"     a genuine planned capability shown with a clean placeholder
 *                    callback — never fake functionality.
 *
 * Generalised: any object whose category doesn't match a specific group falls
 * through to OBJECT, so every editable object gets a sensible panel.
 */

/** Map a logical object → a panel group, purely from its category/role. */
export function categoryGroup(obj) {
  if (!obj) return "object";
  const cat = String(obj.category || obj.type || "").toLowerCase();
  const role = String(obj.role || "").toLowerCase();
  const has = (...ks) => ks.some((k) => cat.includes(k));

  if (role === "text" || has("text", "title", "heading", "word", "caption", "label", "letter")) return "text";
  if (has("background", "backdrop", "scene", "sky")) return "background";
  if (has("face", "head", "portrait")) return "face";
  if (role === "person" || has("person", "people", "man", "woman", "human", "figure", "child", "boy", "girl", "king", "queen")) return "person";
  if (has("logo", "emblem", "crest", "brand", "wordmark", "monogram", "insignia")) return "logo";
  return "object";
}

// Families offered in the (searchable) font dropdown. UI concern only — the
// typography engine still auto-matches the original until the user overrides.
export const FONT_FAMILIES = [
  "Cinzel", "Playfair Display", "Trajan Pro", "Cormorant Garamond", "EB Garamond",
  "Georgia", "Garamond", "Times New Roman", "Baskerville", "Palatino", "Bodoni MT",
  "serif", "Arial", "Helvetica", "Verdana", "Trebuchet MS", "Courier New", "monospace",
];

const slider = (field, min, max, step, unit) => ({ type: "slider", field, min, max, step, unit });

// Universal arrange/delete footer — real, available on every object.
const FOOTER = [
  { id: "front", label: "Bring to front", icon: "⤒", kind: "real" },
  { id: "back", label: "Send to back", icon: "⤓", kind: "real" },
  { id: "delete", label: "Delete", icon: "🗑", kind: "real", danger: true },
];

const GROUPS = {
  text: {
    title: "Text",
    note: "Type is auto-matched to the original until you override it.",
    actions: [
      { id: "replaceText", label: "Replace text", icon: "✎", kind: "real" },
      { id: "font", label: "Font", icon: "𝐀", kind: "control", control: { type: "fontFamily", field: "fontFamily" } },
      { id: "size", label: "Size", icon: "↕", kind: "control", control: slider("fontSize", 8, 220, 1, "px") },
      { id: "weight", label: "Weight", icon: "𝐁", kind: "control", control: { type: "weight", field: "fontStyle" } },
      { id: "color", label: "Color", icon: "🎨", kind: "control", control: { type: "color", field: "fill" } },
      { id: "spacing", label: "Letter spacing", icon: "⇿", kind: "control", control: slider("letterSpacing", -20, 80, 1, "px") },
      { id: "lineHeight", label: "Line height", icon: "↥", kind: "control", control: slider("lineHeight", 0.8, 2.5, 0.05, "") },
      { id: "opacity", label: "Opacity", icon: "◐", kind: "control", control: slider("opacity", 0, 100, 1, "%") },
      { id: "rotation", label: "Rotation", icon: "⟳", kind: "control", control: slider("rotation", -180, 180, 1, "°") },
      { id: "gradient", label: "Gradient", icon: "▥", kind: "soon" },
      { id: "outline", label: "Outline", icon: "◌", kind: "soon" },
      { id: "shadow", label: "Shadow", icon: "☁", kind: "soon" },
      { id: "align", label: "Alignment", icon: "≡", kind: "soon" },
    ],
  },
  person: {
    title: "Person",
    actions: [
      { id: "clothes", label: "Change clothes", icon: "👕", kind: "soon" },
      { id: "hair", label: "Change hair", icon: "💇", kind: "soon" },
      { id: "expression", label: "Change expression", icon: "😊", kind: "soon" },
      { id: "age", label: "Change age", icon: "⏳", kind: "soon" },
      { id: "replacePerson", label: "Replace person", icon: "🔁", kind: "soon" },
      { id: "opacity", label: "Opacity", icon: "◐", kind: "control", control: slider("opacity", 0, 100, 1, "%") },
      { id: "rotation", label: "Rotation", icon: "⟳", kind: "control", control: slider("rotation", -180, 180, 1, "°") },
      { id: "duplicate", label: "Duplicate", icon: "⧉", kind: "real" },
    ],
  },
  face: {
    title: "Face",
    actions: [
      { id: "smile", label: "Smile", icon: "😀", kind: "soon" },
      { id: "beard", label: "Beard", icon: "🧔", kind: "soon" },
      { id: "glasses", label: "Glasses", icon: "👓", kind: "soon" },
      { id: "faceHair", label: "Hair", icon: "💇", kind: "soon" },
      { id: "faceAge", label: "Age", icon: "⏳", kind: "soon" },
      { id: "skinTone", label: "Skin tone", icon: "🎨", kind: "soon" },
      { id: "opacity", label: "Opacity", icon: "◐", kind: "control", control: slider("opacity", 0, 100, 1, "%") },
      { id: "rotation", label: "Rotation", icon: "⟳", kind: "control", control: slider("rotation", -180, 180, 1, "°") },
      { id: "duplicate", label: "Duplicate", icon: "⧉", kind: "real" },
    ],
  },
  background: {
    title: "Background",
    actions: [
      { id: "replaceBg", label: "Replace background", icon: "🖼", kind: "soon" },
      { id: "blurBg", label: "Blur", icon: "🌫", kind: "soon" },
      { id: "colorBg", label: "Color", icon: "🎨", kind: "soon" },
      { id: "generateBg", label: "Generate new background", icon: "✨", kind: "soon" },
      { id: "removeBg", label: "Remove", icon: "⌫", kind: "soon" },
    ],
  },
  logo: {
    title: "Logo",
    actions: [
      { id: "replaceLogo", label: "Replace logo", icon: "🔁", kind: "soon" },
      { id: "uploadLogo", label: "Upload logo", icon: "⤴", kind: "soon" },
      { id: "logoColor", label: "Change color", icon: "🎨", kind: "soon" },
      { id: "opacity", label: "Opacity", icon: "◐", kind: "control", control: slider("opacity", 0, 100, 1, "%") },
      { id: "rotation", label: "Rotation", icon: "⟳", kind: "control", control: slider("rotation", -180, 180, 1, "°") },
      { id: "duplicate", label: "Duplicate", icon: "⧉", kind: "real" },
    ],
  },
  object: {
    title: "Object",
    actions: [
      { id: "opacity", label: "Opacity", icon: "◐", kind: "control", control: slider("opacity", 0, 100, 1, "%") },
      { id: "rotation", label: "Rotation", icon: "⟳", kind: "control", control: slider("rotation", -180, 180, 1, "°") },
      { id: "duplicate", label: "Duplicate", icon: "⧉", kind: "real" },
      { id: "replace", label: "Replace", icon: "🔁", kind: "soon" },
      { id: "recolor", label: "Recolor", icon: "🎨", kind: "soon" },
      { id: "shadow", label: "Shadow", icon: "☁", kind: "soon" },
    ],
  },
};

/** Actions for a group: its category-specific set + the universal footer. */
export function getActions(group) {
  const g = GROUPS[group] || GROUPS.object;
  return { title: g.title, note: g.note || null, actions: [...g.actions, ...FOOTER] };
}

/** Action ids wired to real editor handlers (the rest are controls or placeholders). */
export const REAL_ACTIONS = new Set(["replaceText", "duplicate", "front", "back", "delete"]);
