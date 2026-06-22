import React, { useEffect, useLayoutEffect, useRef, useState } from "react";

/**
 * Context-aware property inspector — a floating glass card that appears beside
 * the selected object. Its contents come from the property registry (driven by
 * the object's semantic category). Real actions call `onAction`; not-yet-built
 * AI actions show a clean inline "coming soon" note (never fake behaviour).
 *
 * Purely presentational + self-positioning. It is keyed by object id in the
 * parent, so selecting a new object remounts it (re-runs the enter animation).
 */

const CARD_W = 250;

const card = {
  position: "fixed",
  width: CARD_W,
  maxHeight: "78vh",
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

const row = (danger) => ({
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

/** Choose a screen position beside the anchor that doesn't cover it. */
function place(anchor, vp, h) {
  const GAP = 14, M = 10;
  let x = anchor.left + anchor.width + GAP; // prefer right of the object
  if (x + CARD_W > vp.w - M) x = anchor.left - CARD_W - GAP; // flip to the left
  if (x < M) x = Math.min(Math.max(M, anchor.left), vp.w - CARD_W - M); // last resort: clamp over a side
  let y = anchor.top + anchor.height / 2 - h / 2; // vertically centred on the object
  y = Math.max(M, Math.min(y, vp.h - h - M));
  return { x, y };
}

export default function PropertyInspector({ object, panel, anchorRect, viewport, onAction }) {
  const ref = useRef(null);
  const [shown, setShown] = useState(false);
  const [pos, setPos] = useState(null);
  const [note, setNote] = useState(null);
  const noteTimer = useRef(0);

  // measure height → position; re-measure when content/anchor/viewport change
  useLayoutEffect(() => {
    const h = ref.current ? ref.current.offsetHeight : 320;
    setPos(place(anchorRect, viewport, h));
  }, [anchorRect, viewport, panel]);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => () => clearTimeout(noteTimer.current), []);

  const click = (a) => {
    if (a.kind === "real") {
      onAction(a.id);
    } else {
      setNote(`${a.label} — coming soon`);
      clearTimeout(noteTimer.current);
      noteTimer.current = setTimeout(() => setNote(null), 1800);
    }
  };

  const hidden = !pos;
  const style = {
    ...card,
    left: pos ? pos.x : -9999,
    top: pos ? pos.y : 0,
    opacity: shown && !hidden ? 1 : 0,
    transform: shown && !hidden ? "translateX(0) scale(1)" : "translateX(6px) scale(0.98)",
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
        const prevReal = i > 0 ? panel.actions[i - 1].kind === "real" : a.kind === "real";
        const divider = a.kind === "real" && !prevReal; // separate the universal real footer
        return (
          <React.Fragment key={a.id}>
            {divider && <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "8px 2px 2px" }} />}
            <button
              style={row(a.danger)}
              onClick={() => click(a)}
              onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.12)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.05)")}
            >
              <span style={{ width: 18, textAlign: "center", opacity: 0.9 }}>{a.icon}</span>
              <span style={{ flex: 1 }}>{a.label}</span>
              {a.kind === "soon" && (
                <span
                  style={{
                    fontSize: 9,
                    fontWeight: 700,
                    letterSpacing: 0.4,
                    padding: "2px 6px",
                    borderRadius: 6,
                    background: "rgba(216,179,106,0.18)",
                    color: "#d8b36a",
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
        <div
          style={{
            marginTop: 8,
            padding: "7px 10px",
            borderRadius: 9,
            background: "rgba(216,179,106,0.14)",
            color: "#e6cf9c",
            fontSize: 12,
          }}
        >
          {note}
        </div>
      )}
    </div>
  );
}
