/**
 * Semantic vocabulary for the Object Manager.
 *
 * GroundingDINO labels arrive as underscore-joined token bags, e.g.
 *   "person_royal_portrait_frame"  ->  ["person","royal","portrait","frame"]
 * We classify an object by the SINGLE highest-specificity token it matches.
 * Specificity (higher = more specific / preferred when detections overlap) is
 * the primary signal for both classification and click-resolution — never size.
 *
 * `editable: false` means "ignored by default" — the object is RETAINED and
 * promotable, it simply does not participate in editing. Nothing is discarded.
 */

export const ROLE = Object.freeze({
  TEXT: "text", // editable text (OCR-backed)
  PART: "part", // leaf of a person: face / crown / clothes / sword / head
  DECOR: "decor", // free-standing decoration: flower / ornament / emblem
  PARENT: "parent", // a person / figure that owns parts
  CONTAINER: "container", // portrait / frame / painting — scene structure, not an object
  SCENE: "scene", // background / sky — the backdrop itself
});

/**
 * token -> { category, role, specificity, editable }
 */
export const VOCABULARY = Object.freeze({
  // ---- TEXT (OCR-backed text wins over everything when clicked) ----
  text: { category: "text", role: ROLE.TEXT, specificity: 100, editable: true },
  title: { category: "text", role: ROLE.TEXT, specificity: 96, editable: true },
  heading: { category: "text", role: ROLE.TEXT, specificity: 95, editable: true },
  subtitle: { category: "text", role: ROLE.TEXT, specificity: 94, editable: true },
  typography: { category: "text", role: ROLE.TEXT, specificity: 92, editable: true },
  author: { category: "text", role: ROLE.TEXT, specificity: 88, editable: true },
  name: { category: "text", role: ROLE.TEXT, specificity: 80, editable: true },
  words: { category: "text", role: ROLE.TEXT, specificity: 70, editable: true },
  letters: { category: "text", role: ROLE.TEXT, specificity: 68, editable: true },
  sentence: { category: "text", role: ROLE.TEXT, specificity: 66, editable: true },

  // ---- PARTS (leaves of a person) ----
  face: { category: "face", role: ROLE.PART, specificity: 90, editable: true },
  crown: { category: "crown", role: ROLE.PART, specificity: 88, editable: true },
  sword: { category: "sword", role: ROLE.PART, specificity: 86, editable: true },
  head: { category: "head", role: ROLE.PART, specificity: 84, editable: true },
  hair: { category: "hair", role: ROLE.PART, specificity: 74, editable: true },
  clothing: { category: "clothing", role: ROLE.PART, specificity: 70, editable: true },

  // ---- DECOR (free-standing) ----
  ornament: { category: "ornament", role: ROLE.DECOR, specificity: 62, editable: true },
  flower: { category: "flower", role: ROLE.DECOR, specificity: 60, editable: true },
  emblem: { category: "emblem", role: ROLE.DECOR, specificity: 58, editable: true },
  animal: { category: "animal", role: ROLE.DECOR, specificity: 56, editable: true },
  decoration: { category: "decoration", role: ROLE.DECOR, specificity: 50, editable: true },
  symbol: { category: "symbol", role: ROLE.DECOR, specificity: 48, editable: true },

  // ---- PARENTS (a large person is still editable — semantics, not size) ----
  person: { category: "person", role: ROLE.PARENT, specificity: 40, editable: true },
  human: { category: "person", role: ROLE.PARENT, specificity: 40, editable: true },
  figure: { category: "person", role: ROLE.PARENT, specificity: 38, editable: true },

  // ---- CONTAINERS (retained, not editable by default) ----
  logo: { category: "logo", role: ROLE.CONTAINER, specificity: 16, editable: false },
  portrait: { category: "portrait", role: ROLE.CONTAINER, specificity: 15, editable: false },
  painting: { category: "painting", role: ROLE.CONTAINER, specificity: 14, editable: false },
  illustration: { category: "illustration", role: ROLE.CONTAINER, specificity: 13, editable: false },
  frame: { category: "frame", role: ROLE.CONTAINER, specificity: 12, editable: false },
  border: { category: "border", role: ROLE.CONTAINER, specificity: 12, editable: false },

  // ---- SCENE (retained, not editable) ----
  building: { category: "building", role: ROLE.SCENE, specificity: 8, editable: false },
  texture: { category: "texture", role: ROLE.SCENE, specificity: 6, editable: false },
  sky: { category: "sky", role: ROLE.SCENE, specificity: 5, editable: false },
  clouds: { category: "clouds", role: ROLE.SCENE, specificity: 5, editable: false },
  background: { category: "background", role: ROLE.SCENE, specificity: 2, editable: false },
});

/** Pure modifier tokens that carry no category of their own. */
export const MODIFIERS = new Set([
  "historical",
  "royal",
  "illustrated",
  "decorative",
  "book",
  "paper",
  "object",
]);

export const DEFAULT_CONFIG = Object.freeze({
  duplicateIoU: 0.8, // same-category boxes above this IoU collapse to one logical object
  childContainment: 0.6, // a part is a child if >60% of its area sits inside a parent
  // a non-text detection with >= this fraction of its area inside an OCR text
  // line is a mislabelled fragment of that text (SAM word/letter mask that CLIP
  // tagged emblem/symbol/ornament) -> ceded to the text object, so OCR text owns
  // every text region. Honours "OCR-backed text wins" (see VOCABULARY header).
  textDominanceContainment: 0.7,
});
