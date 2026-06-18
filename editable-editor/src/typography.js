/**
 * Runtime typography matching.
 *
 * The ORIGINAL text bitmap is the source of truth. When the user edits the
 * string we must synthesize live typography that stays visually close to the
 * baked artwork — so we read the bitmap's own pixels and metrics rather than
 * trusting a single estimated colour/family:
 *
 *   - COLOUR  : sample the glyph pixels (overall, top band, bottom band, the
 *               brightest and darkest) → a vertical gradient (e.g. gold) + an
 *               outline colour, so the fill matches the original exactly.
 *   - SIZE    : measure the opaque glyph band → font size from real cap height.
 *   - FAMILY  : measureText across available serif families, pick the one whose
 *               rendered width is closest to the bitmap's, then add tracking so
 *               it fits the original footprint exactly.
 *   - DEPTH   : a subtle stroke + shadow sampled from the darkest glyph pixels,
 *               approximating the original's outline / emboss.
 *
 * Pure/agnostic: no per-image or per-category logic. Everything is derived from
 * the bitmap + the object's footprint. Degrades to the caller's fallbacks if the
 * bitmap can't be read (e.g. not yet decoded).
 */

// Serif display families worth trying; only those actually available are used.
const CANDIDATES = [
  "Cinzel",
  "Playfair Display",
  "Trajan Pro",
  "Cormorant Garamond",
  "EB Garamond",
  "Georgia",
  "Garamond",
  "Times New Roman",
  "serif",
];

let _measureCtx = null;
let _analyzeCanvas = null;

function rgb(c) {
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Mean / extreme glyph colours + vertical bands + opaque row coverage. */
export function analyzeBitmap(img, maxDim = 120) {
  const w = (img && (img.naturalWidth || img.width)) || 0;
  const h = (img && (img.naturalHeight || img.height)) || 0;
  if (!w || !h) return null;
  const scale = Math.min(1, maxDim / Math.max(w, h));
  const cw = Math.max(1, Math.round(w * scale));
  const ch = Math.max(1, Math.round(h * scale));
  if (!_analyzeCanvas) _analyzeCanvas = document.createElement("canvas");
  _analyzeCanvas.width = cw;
  _analyzeCanvas.height = ch;
  const ctx = _analyzeCanvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(img, 0, 0, cw, ch);
  let data;
  try {
    data = ctx.getImageData(0, 0, cw, ch).data;
  } catch {
    return null; // tainted canvas (shouldn't happen for same-origin / data URLs)
  }

  const BANDS = 3;
  const band = Array.from({ length: BANDS }, () => ({ r: 0, g: 0, b: 0, n: 0 }));
  let sumR = 0, sumG = 0, sumB = 0, n = 0;
  let minY = ch, maxY = -1;
  let bright = null, brightLum = -1, dark = null, darkLum = 1e9;

  for (let y = 0; y < ch; y++) {
    const bi = Math.min(BANDS - 1, Math.floor((y / ch) * BANDS));
    for (let x = 0; x < cw; x++) {
      const i = (y * cw + x) * 4;
      if (data[i + 3] < 128) continue; // glyph pixels only
      const r = data[i], g = data[i + 1], b = data[i + 2];
      band[bi].r += r; band[bi].g += g; band[bi].b += b; band[bi].n++;
      sumR += r; sumG += g; sumB += b; n++;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      if (lum > brightLum) { brightLum = lum; bright = [r, g, b]; }
      if (lum < darkLum) { darkLum = lum; dark = [r, g, b]; }
    }
  }
  if (!n) return null;
  const mid = [Math.round(sumR / n), Math.round(sumG / n), Math.round(sumB / n)];
  const bandMean = (b) => (b.n ? [Math.round(b.r / b.n), Math.round(b.g / b.n), Math.round(b.b / b.n)] : mid);
  return {
    top: bandMean(band[0]),
    bottom: bandMean(band[BANDS - 1]),
    mid,
    bright: bright || mid,
    dark: dark || mid,
    coverTop: minY / ch,
    coverBottom: (maxY + 1) / ch,
  };
}

function availableFamilies() {
  const out = [];
  for (const f of CANDIDATES) {
    if (f === "serif") { out.push(f); continue; }
    try {
      if (document.fonts && document.fonts.check(`40px "${f}"`)) out.push(f);
    } catch {
      /* ignore */
    }
  }
  if (!out.includes("serif")) out.push("serif");
  return out;
}

function measureWidth(text, family, size, weight) {
  if (!_measureCtx) _measureCtx = document.createElement("canvas").getContext("2d");
  _measureCtx.font = `${weight} ${size}px "${family}"`;
  return _measureCtx.measureText(text).width;
}

/** Pick the available family whose rendered width best matches the original. */
export function matchFont({ text, targetW, fontSize, weight = "bold", preferred }) {
  const fams = availableFamilies();
  if (preferred && preferred !== "serif" && !fams.includes(preferred)) fams.unshift(preferred);
  const str = text && text.length ? text : "Ag";
  let best = preferred || fams[0];
  let bestErr = Infinity;
  for (const f of fams) {
    const wpx = measureWidth(str, f, fontSize, weight);
    if (!wpx) continue;
    const err = targetW > 0 ? Math.abs(wpx - targetW) / targetW : 0;
    if (err < bestErr) { bestErr = err; best = f; }
  }
  // tracking so the chosen family fills the original footprint width exactly
  const wpx = measureWidth(str, best, fontSize, weight) || targetW;
  const gaps = Math.max(1, str.length - 1);
  let letterSpacing = targetW > 0 ? (targetW - wpx) / gaps : 0;
  letterSpacing = Math.max(-fontSize * 0.12, Math.min(fontSize * 0.6, letterSpacing));
  return { fontFamily: best, letterSpacing, fitError: bestErr };
}

/**
 * Konva <Text> style props synthesized to match the original bitmap.
 * `img` is the decoded bitmap element; falls back to the caller's metadata
 * (fallbackFamily / fallbackColor) when the bitmap can't be analyzed.
 */
export function estimateTypography(img, { text, width, height, weight = "bold", fallbackFamily, fallbackColor }) {
  const a = analyzeBitmap(img);
  if (!a) {
    return {
      fontFamily: fallbackFamily || "serif",
      fontSize: Math.max(12, Math.round(height * 0.6)),
      letterSpacing: 0,
      fill: fallbackColor || "#d8b36a",
    };
  }
  const cover = Math.max(0.25, a.coverBottom - a.coverTop); // glyph band fraction
  const glyphH = cover * height;
  const fontSize = Math.max(10, Math.round(glyphH / 0.72)); // cap-height → em
  const fm = matchFont({
    text,
    targetW: width,
    fontSize,
    weight,
    preferred: fallbackFamily,
  });

  const props = {
    fontFamily: fm.fontFamily,
    fontSize,
    letterSpacing: fm.letterSpacing,
    fill: rgb(a.bright),
  };

  // vertical gradient when the top/bottom of the glyphs differ (gold, gradients)
  const vdiff =
    Math.abs(a.top[0] - a.bottom[0]) + Math.abs(a.top[1] - a.bottom[1]) + Math.abs(a.top[2] - a.bottom[2]);
  if (vdiff > 36) {
    props.gradient = [0, rgb(a.top), 0.5, rgb(a.bright), 1, rgb(a.bottom)];
  }

  // outline + shadow approximating the original's stroke / emboss
  props.stroke = rgb(a.dark);
  props.strokeWidth = Math.max(0.4, fontSize * 0.018);
  props.shadowColor = rgb(a.dark);
  props.shadowBlur = Math.max(1, fontSize * 0.05);
  props.shadowOpacity = 0.45;
  props.shadowOffsetY = Math.max(1, Math.round(fontSize * 0.03));
  return props;
}
