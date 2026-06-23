import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { FONT_FAMILIES } from "./propertyRegistry";

/**
 * Context-aware property inspector — a floating glass card beside the selected
 * object. Contents come from the registry (driven by semantic category).
 *
 *  - kind "control" → a live widget (slider / font dropdown / weight / colour)
 *    bound to `values`, emitting onControlStart() on interaction begin (history
 *    checkpoint) and onControlChange(patch) live.
 *  - kind "real"    → a button → onAction(id).
 *  - kind "soon"    → a button with a SOON badge → clean inline "coming soon".
 *
 * Purely presentational + self-positioning. Keyed by object id in the parent so
 * selecting a new object remounts it (re-runs the enter animation).
 */

const CARD_W = 268;

const card = {
  position: "fixed",
  width: CARD_W,
  maxHeight: "82vh",
  overflowY: "auto",
  padding: 12,
  borderRadius: 16,
  background: "rgba(24,24,28,0.62)",
  backdropFilter: "blur(18px) saturate(140%)",
  WebkitBackdropFilter: "blur(18px) saturate(140%)",
  border: "1px solid rgba(255,255,255,0.12)",
  boxShadow: "0 12px 40px rgba(0,0,0,0.45)",
  color: "#f0ead9",
  font: "13px/1.35 -apple-system, Segoe UI, system-ui, sans-serif",
  zIndex: 30,
  transition: "opacity 180ms ease, transform 180ms cubic-bezier(.2,.8,.2,1)",
};

const rowBtn = (danger) => ({
  display: "flex",
  alignItems: "center",
  gap: 10,
  width: "100%",
  padding: "8px 10px",
  marginTop: 4,
  borderRadius: 10,
  border: "1px solid transparent",
  background: "rgba(255,255,255,0.05)",
  color: danger ? "#ff9e87" : "#f0ead9",
  cursor: "pointer",
  textAlign: "left",
  font: "inherit",
  transition: "background 120ms ease",
});

const ctrlRow = { padding: "6px 8px", marginTop: 4, borderRadius: 10, background: "rgba(255,255,255,0.05)" };
const ctrlHead = { display: "flex", alignItems: "center", gap: 8, marginBottom: 4, opacity: 0.92 };

function rgbToHex(c) {
  if (!c) return "#d8b36a";
  if (c[0] === "#") return c.slice(0, 7);
  const m = String(c).match(/(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  if (!m) return "#d8b36a";
  const h = (n) => Math.max(0, Math.min(255, +n)).toString(16).padStart(2, "0");
  return `#${h(m[1])}${h(m[2])}${h(m[3])}`;
}

/** Choose a screen position beside the anchor that doesn't cover it. */
function place(anchor, vp, h) {
  const GAP = 14, M = 10;
  let x = anchor.left + anchor.width + GAP;
  if (x + CARD_W > vp.w - M) x = anchor.left - CARD_W - GAP;
  if (x < M) x = Math.min(Math.max(M, anchor.left), vp.w - CARD_W - M);
  let y = anchor.top + anchor.height / 2 - h / 2;
  y = Math.max(M, Math.min(y, vp.h - h - M));
  return { x, y };
}

function Slider({ c, value, onStart, onChange }) {
  const v = Number.isFinite(value) ? value : c.control.min;
  const disp = c.control.field === "opacity" ? `${Math.round(v)}${c.control.unit}` : `${c.control.step < 1 ? v.toFixed(2) : Math.round(v)}${c.control.unit}`;
  return (
    <div style={ctrlRow}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, opacity: 0.92 }}>
        <span>{c.label}</span>
        <span style={{ opacity: 0.6, fontVariantNumeric: "tabular-nums" }}>{disp}</span>
      </div>
      <input
        type="range"
        min={c.control.min}
        max={c.control.max}
        step={c.control.step}
        value={v}
        onPointerDown={onStart}
        onChange={(e) => onChange({ [c.control.field]: Number(e.target.value) })}
        style={{ width: "100%", accentColor: "#d8b36a", cursor: "pointer" }}
      />
    </div>
  );
}

export default function PropertyInspector({ object, panel, values, anchorRect, viewport, onAction, onControlStart, onControlChange }) {
  const ref = useRef(null);
  const [shown, setShown] = useState(false);
  const [pos, setPos] = useState(null);
  const [note, setNote] = useState(null);
  const noteTimer = useRef(0);

  useLayoutEffect(() => {
    const h = ref.current ? ref.current.offsetHeight : 360;
    setPos(place(anchorRect, viewport, h));
  }, [anchorRect, viewport, panel, values]);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);
  useEffect(() => () => clearTimeout(noteTimer.current), []);

  const soonNote = (label) => {
    setNote(`${label} — coming soon`);
    clearTimeout(noteTimer.current);
    noteTimer.current = setTimeout(() => setNote(null), 1800);
  };

  const hidden = !pos;
  const style = {
    ...card,
    left: pos ? pos.x : -9999,
    top: pos ? pos.y : 0,
    opacity: shown && !hidden ? 1 : 0,
    transform: shown && !hidden ? "translateX(0) scale(1)" : "translateX(6px) scale(0.98)",
  };

  const renderControl = (a) => {
    const c = a.control;
    if (c.type === "slider") {
      return <Slider key={a.id} c={a} value={values[c.field]} onStart={onControlStart} onChange={onControlChange} />;
    }
    if (c.type === "fontFamily") {
      return (
        <div key={a.id} style={ctrlRow}>
          <div style={ctrlHead}>
            <span style={{ width: 16, textAlign: "center" }}>{a.icon}</span>
            <span>Font</span>
          </div>
          <input
            list="pi-fonts"
            defaultValue={values.fontFamily || ""}
            placeholder="Search fonts…"
            onFocus={onControlStart}
            onChange={(e) => e.target.value && onControlChange({ fontFamily: e.target.value })}
            style={{
              width: "100%", padding: "6px 8px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.15)",
              background: "rgba(0,0,0,0.25)", color: "#f0ead9", font: "inherit", boxSizing: "border-box",
            }}
          />
          <datalist id="pi-fonts">
            {FONT_FAMILIES.map((f) => (
              <option key={f} value={f} />
            ))}
          </datalist>
        </div>
      );
    }
    if (c.type === "weight") {
      const cur = values.fontStyle === "normal" ? "normal" : "bold";
      const opt = (w, label) => (
        <button
          key={w}
          onClick={() => {
            onControlStart();
            onControlChange({ fontStyle: w });
          }}
          style={{
            flex: 1, padding: "6px 0", borderRadius: 8, border: "1px solid rgba(255,255,255,0.12)",
            background: cur === w ? "#d8b36a" : "rgba(255,255,255,0.05)",
            color: cur === w ? "#1a1a1a" : "#f0ead9", cursor: "pointer", font: "inherit",
            fontWeight: w === "bold" ? 700 : 400,
          }}
        >
          {label}
        </button>
      );
      return (
        <div key={a.id} style={ctrlRow}>
          <div style={ctrlHead}>
            <span style={{ width: 16, textAlign: "center" }}>{a.icon}</span>
            <span>Weight</span>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {opt("normal", "Regular")}
            {opt("bold", "Bold")}
          </div>
        </div>
      );
    }
    if (c.type === "color") {
      const field = c.field || "fill";
      const hex = rgbToHex(values[field]);
      return (
        <div key={a.id} style={{ ...ctrlRow, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={ctrlHead}>
            <span style={{ width: 16, textAlign: "center" }}>{a.icon}</span>
            <span>{a.label}</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ opacity: 0.6, fontVariantNumeric: "tabular-nums" }}>{hex}</span>
            <input
              type="color"
              defaultValue={hex}
              onPointerDown={onControlStart}
              onInput={(e) => onControlChange({ [field]: e.target.value })}
              style={{ width: 30, height: 26, padding: 0, border: "none", background: "none", cursor: "pointer" }}
            />
          </div>
        </div>
      );
    }
    if (c.type === "align") {
      const cur = values.align || "center";
      const opt = (val, label) => (
        <button
          key={val}
          onClick={() => { onControlStart(); onControlChange({ align: val }); }}
          style={{
            flex: 1, padding: "6px 0", borderRadius: 8, border: "1px solid rgba(255,255,255,0.12)",
            background: cur === val ? "#d8b36a" : "rgba(255,255,255,0.05)",
            color: cur === val ? "#1a1a1a" : "#f0ead9", cursor: "pointer", font: "inherit",
          }}
        >
          {label}
        </button>
      );
      return (
        <div key={a.id} style={ctrlRow}>
          <div style={ctrlHead}>
            <span style={{ width: 16, textAlign: "center" }}>{a.icon}</span>
            <span>Alignment</span>
          </div>
          <div style={{ display: "flex", gap: 6 }}>{opt("left", "Left")}{opt("center", "Center")}{opt("right", "Right")}</div>
        </div>
      );
    }
    if (c.type === "gradient") {
      const g = Array.isArray(values.gradient) ? values.gradient : null;
      const start = g ? g[1] : rgbToHex(values.fill);
      const end = g ? g[3] : "#5a2e12";
      const sw = { width: 30, height: 26, padding: 0, border: "none", background: "none", cursor: "pointer" };
      return (
        <div key={a.id} style={ctrlRow}>
          <div style={ctrlHead}>
            <span style={{ width: 16, textAlign: "center" }}>{a.icon}</span>
            <span>Gradient</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input type="color" defaultValue={start} title="Top colour" onPointerDown={onControlStart}
              onInput={(e) => onControlChange({ gradient: [0, e.target.value, 1, end] })} style={sw} />
            <span style={{ opacity: 0.5 }}>→</span>
            <input type="color" defaultValue={end} title="Bottom colour" onPointerDown={onControlStart}
              onInput={(e) => onControlChange({ gradient: [0, start, 1, e.target.value] })} style={sw} />
            <button
              onClick={() => { onControlStart(); onControlChange({ gradient: null }); }}
              style={{ marginLeft: "auto", padding: "4px 8px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.05)", color: "#f0ead9", cursor: "pointer", font: "inherit", fontSize: 11 }}
            >
              Off
            </button>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div ref={ref} style={style} onClick={(e) => e.stopPropagation()}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 2 }}>
        <span style={{ fontWeight: 700, letterSpacing: 0.3 }}>{panel.title}</span>
        <span style={{ fontSize: 11, opacity: 0.55 }}>
          {object.category || object.type}#{object.id}
        </span>
      </div>
      {panel.note && <div style={{ fontSize: 11, opacity: 0.6, margin: "2px 0 6px" }}>{panel.note}</div>}

      {panel.actions.map((a, i) => {
        if (a.kind === "control") return renderControl(a);

        const prevReal = i > 0 ? panel.actions[i - 1].kind === "real" : a.kind === "real";
        const divider = a.kind === "real" && !prevReal;
        return (
          <React.Fragment key={a.id}>
            {divider && <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "8px 2px 2px" }} />}
            <button
              style={rowBtn(a.danger)}
              onClick={() => (a.kind === "real" ? onAction(a.id) : soonNote(a.label))}
              onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.12)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.05)")}
            >
              <span style={{ width: 18, textAlign: "center", opacity: 0.9 }}>{a.icon}</span>
              <span style={{ flex: 1 }}>{a.label}</span>
              {a.kind === "soon" && (
                <span
                  style={{
                    fontSize: 9, fontWeight: 700, letterSpacing: 0.4, padding: "2px 6px", borderRadius: 6,
                    background: "rgba(216,179,106,0.18)", color: "#d8b36a",
                  }}
                >
                  SOON
                </span>
              )}
            </button>
          </React.Fragment>
        );
      })}

      {note && (
        <div style={{ marginTop: 8, padding: "7px 10px", borderRadius: 9, background: "rgba(216,179,106,0.14)", color: "#e6cf9c", fontSize: 12 }}>
          {note}
        </div>
      )}
    </div>
  );
}
