/**
 * Runtime typography matching.
 *
 * The ORIGINAL text bitmap is the source of truth. When the user edits the
 * string we synthesize live typography that stays visually close to the baked
 * artwork by reading the bitmap's own pixels and metrics:
 *
 *   - COLOUR : sample the glyph pixels (overall, top/bottom band, brightest,
 *              darkest) → a vertical gradient (gold) + an outline colour, so the
 *              fill matches the original exactly.
 *   - SIZE   : measure the opaque glyph band → font size from real cap height.
 *   - FAMILY : RENDER each available serif family (× normal/bold weight), fit it
 *              to the original footprint, rasterise it, and compare its glyph
 *              ink mask to the bitmap's (IoU). Pick the closest rendering — i.e.
 *              choose the family + weight by pixel similarity, not just width.
 *   - DEPTH  : stroke + shadow sampled from the darkest glyph pixels.
 *
 * Pure/agnostic: no per-image or per-category constants — everything is derived
 * from the bitmap + footprint. Degrades to the caller's fallbacks whenever the
 * bitmap can't be read (e.g. not yet decoded, tainted canvas).
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
  "Bodoni MT",
  "serif",
];

let _measureCtx = null;
let _aCanvas = null;
let _bCanvas = null;

function rgb(c) {
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function ctxFor(which, w, h) {
  const ref = which === "a" ? (_aCanvas ||= document.createElement("canvas")) : (_bCanvas ||= document.createElement("canvas"));
  ref.width = w;
  ref.height = h;
  return ref.getContext("2d", { willReadFrequently: true });
}

/** Mean / extreme glyph colours + vertical bands + opaque row coverage. */
export function analyzeBitmap(img, maxDim = 120) {
  const w = (img && (img.naturalWidth || img.width)) || 0;
  const h = (img && (img.naturalHeight || img.height)) || 0;
  if (!w || !h) return null;
  const scale = Math.min(1, maxDim / Math.max(w, h));
  const cw = Math.max(1, Math.round(w * scale));
  const ch = Math.max(1, Math.round(h * scale));
  const ctx = ctxFor("a", cw, ch);
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
    coverage: n / (cw * ch), // ink fraction → a weight cue
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

// ---- pixel-similarity font matching ----------------------------------------

// A small normalised grid both the bitmap and each candidate are rasterised to.
const GRID_W = 160;
const GRID_H = 40;

/** Binary ink mask of the original bitmap (alpha if present, else luminance contrast). */
function bitmapInk(img) {
  const ctx = ctxFor("a", GRID_W, GRID_H);
  ctx.clearRect(0, 0, GRID_W, GRID_H);
  ctx.drawImage(img, 0, 0, GRID_W, GRID_H);
  let d;
  try {
    d = ctx.getImageData(0, 0, GRID_W, GRID_H).data;
  } catch {
    return null;
  }
  const ink = new Uint8Array(GRID_W * GRID_H);
  // does alpha carry the glyphs?
  let amin = 255, amax = 0;
  for (let i = 0; i < GRID_W * GRID_H; i++) {
    const a = d[i * 4 + 3];
    if (a < amin) amin = a;
    if (a > amax) amax = a;
  }
  if (amax - amin > 40) {
    for (let i = 0; i < ink.length; i++) ink[i] = d[i * 4 + 3] > 128 ? 1 : 0;
  } else {
    // opaque rect: ink = pixels whose luminance deviates from the median background
    const lum = new Float32Array(GRID_W * GRID_H);
    for (let i = 0; i < ink.length; i++) lum[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
    const sorted = Float32Array.from(lum).sort();
    const med = sorted[sorted.length >> 1];
    let dev = 0;
    for (let i = 0; i < lum.length; i++) dev += Math.abs(lum[i] - med);
    dev /= lum.length;
    const t = Math.max(18, dev * 1.5);
    for (let i = 0; i < ink.length; i++) ink[i] = Math.abs(lum[i] - med) > t ? 1 : 0;
  }
  return ink;
}

/** Ink mask of a candidate rendering, fitted to the same grid. */
function renderInk(text, family, size, weight, letterSpacing) {
  const ctx = ctxFor("b", GRID_W, GRID_H);
  ctx.clearRect(0, 0, GRID_W, GRID_H);
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  try {
    ctx.letterSpacing = `${letterSpacing || 0}px`;
  } catch {
    /* older browsers ignore */
  }
  ctx.font = `${weight} ${size}px "${family}"`;
  ctx.fillText(text || "Ag", GRID_W / 2, GRID_H / 2 + size * 0.02);
  let d;
  try {
    d = ctx.getImageData(0, 0, GRID_W, GRID_H).data;
  } catch {
    return null;
  }
  const ink = new Uint8Array(GRID_W * GRID_H);
  for (let i = 0; i < ink.length; i++) ink[i] = d[i * 4 + 3] > 64 ? 1 : 0;
  return ink;
}

function iou(a, b) {
  let inter = 0, uni = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (x | y) uni++;
    if (x & y) inter++;
  }
  return uni ? inter / uni : 0;
}

/**
 * Choose family + weight by PIXEL SIMILARITY to the original bitmap, then fit
 * tracking so it fills the original width exactly. `refImg` is the original
 * bitmap element; `refText` is the original string (best glyphs to compare).
 */
export function matchFont({ refImg, refText, text, targetW, fontSize, fallback }) {
  const fams = availableFamilies();
  if (fallback && fallback !== "serif" && !fams.includes(fallback)) fams.unshift(fallback);
  const displayStr = text && text.length ? text : refText || "Ag";

  // tracking that fills the footprint width for a given family/weight
  const trackFor = (family, weight) => {
    const wpx = measureWidth(displayStr, family, fontSize, weight) || targetW;
    const gaps = Math.max(1, displayStr.length - 1);
    let ls = targetW > 0 ? (targetW - wpx) / gaps : 0;
    return Math.max(-fontSize * 0.12, Math.min(fontSize * 0.6, ls));
  };

  const refInk = refImg ? bitmapInk(refImg) : null;
  const cmpStr = refText && refText.length ? refText : displayStr; // compare with original glyphs
  let best = { fontFamily: fallback || fams[0], fontStyle: "bold", letterSpacing: trackFor(fallback || fams[0], "bold"), score: -1 };

  if (refInk) {
    for (const f of fams) {
      for (const weight of ["normal", "bold"]) {
        const ls = trackFor(f, weight);
        const cand = renderInk(cmpStr, f, fontSize, weight, ls);
        if (!cand) continue;
        const s = iou(refInk, cand);
        if (s > best.score) best = { fontFamily: f, fontStyle: weight, letterSpacing: trackFor(f, weight), score: s };
      }
    }
  } else {
    // no bitmap: width-only heuristic across families (bold)
    let err = Infinity;
    for (const f of fams) {
      const wpx = measureWidth(displayStr, f, fontSize, "bold");
      const e = targetW > 0 ? Math.abs(wpx - targetW) / targetW : 0;
      if (e < err) { err = e; best = { fontFamily: f, fontStyle: "bold", letterSpacing: trackFor(f, "bold"), score: 1 - e }; }
    }
  }
  return best;
}

/**
 * Konva <Text> style props synthesized to match the original bitmap. `img` is
 * the decoded bitmap element; `refText` is the original string (for matching).
 * Falls back to metadata (fallbackFamily / fallbackColor) when unreadable.
 */
export function estimateTypography(img, { text, refText, width, height, fallbackFamily, fallbackColor }) {
  const a = analyzeBitmap(img);
  if (!a) {
    return {
      fontFamily: fallbackFamily || "serif",
      fontStyle: "bold",
      fontSize: Math.max(12, Math.round(height * 0.6)),
      letterSpacing: 0,
      fill: fallbackColor || "#d8b36a",
    };
  }
  const cover = Math.max(0.25, a.coverBottom - a.coverTop); // glyph band fraction
  const glyphH = cover * height;
  const fontSize = Math.max(10, Math.round(glyphH / 0.72)); // cap-height → em
  const fm = matchFont({ refImg: img, refText, text, targetW: width, fontSize, fallback: fallbackFamily });

  const props = {
    fontFamily: fm.fontFamily,
    fontStyle: fm.fontStyle || "bold",
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
