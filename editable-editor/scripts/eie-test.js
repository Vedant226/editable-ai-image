/**
 * Headless test for the Editing Illusion Engine core (deterministic clock).
 * Run:  npm run eie:test
 */

import { createEditingIllusionEngine, PHASE } from "../src/editingIllusion/index.js";

let pass = 0;
let fail = 0;
const check = (name, cond) => {
  if (cond) {
    pass++;
    console.log(`  ✓ ${name}`);
  } else {
    fail++;
    console.log(`  ✗ ${name}`);
  }
};

console.log("\nRESTING / SELECTION (I2 identity-at-rest)");
const e = createEditingIllusionEngine();
e.select(1, 0);
let f = e.frame(1, 0);
check("RESTING shows the base", f.showBaseAtFootprint === true);
check("RESTING hides the cutout (I2)", f.cutout.visible === false);
check("selection glow rises by glowMs", e.frame(1, 150).selection.glow > 0.9);

console.log("\nLIFT with temp fill (assets not yet ready)");
check("canLift via temp fill", e.canLift(1) === true);
check("beginLift succeeds", e.beginLift(1, 200) === true);
f = e.frame(1, 200);
check("LIFTING occludes the base footprint (I3/I4)", f.showBaseAtFootprint === false);
check("LIFTING covers footprint with temp fill", f.fill.visible === true && f.fill.source === "temp");
check("LIFTING shows the cutout", f.cutout.visible === true);
f = e.frame(1, 290);
check("lift rises (dy < 0) and scales up", f.cutout.dy < 0 && f.cutout.scale > 1);
e.tick(380);
check("auto-advances to FLOATING", e.getState(1).phase === PHASE.FLOATING);
f = e.frame(1, 380);
check(
  "FLOATING holds full elevation",
  Math.abs(f.cutout.dy + e.config.liftPx) < 1e-6 && Math.abs(f.cutout.scale - e.config.liftScale) < 1e-6
);
check("FLOATING shadow fully grown", f.shadow.opacity > 0 && f.shadow.growth === 1);

console.log("\nTEMP → LAMA cross-fade (I6 masked swap)");
e.setAssets(1, "ready", 400);
f = e.frame(1, 400);
check("fill source becomes lama", f.fill.source === "lama");
check("cross-fade from temp underway", f.fill.crossfadeFrom === "temp" && f.fill.crossfade < 1);
f = e.frame(1, 600);
check("cross-fade completes", f.fill.crossfade >= 1 && f.fill.crossfadeFrom === null);

console.log("\nDROP → SETTLE → PLACED");
e.drop(1, 700);
check("drop → SETTLING", e.getState(1).phase === PHASE.SETTLING);
e.tick(860);
check("settle → PLACED", e.getState(1).phase === PHASE.PLACED);
f = e.frame(1, 860);
check("PLACED grounded (dy 0, scale 1)", f.cutout.dy === 0 && f.cutout.scale === 1);
check("PLACED still occludes the base (I4)", f.showBaseAtFootprint === false && f.fill.visible === true);

console.log("\nDEGRADATION (I3 — never displace over a hole)");
const e2 = createEditingIllusionEngine({ tempFillAvailable: false });
e2.select(2, 0);
e2.setAssets(2, "failed", 0);
check("cannot lift without any cover", e2.beginLift(2, 10) === false);
check("stays RESTING (no displacement)", e2.getState(2).phase === PHASE.RESTING);
check("RESTING still shows the base", e2.frame(2, 10).showBaseAtFootprint === true);
e2.setAssets(2, "ready", 20);
check("can lift once assets ready", e2.beginLift(2, 30) === true);
check("uses lama fill (no temp needed)", e2.frame(2, 30).fill.source === "lama");

console.log("\nDELETE (cutout fades, fill remains = erased)");
const e3 = createEditingIllusionEngine();
e3.select(3, 0);
e3.beginLift(3, 0);
e3.tick(180);
e3.delete(3, 500);
f = e3.frame(3, 590);
check("DELETING fades the cutout out", f.cutout.opacity < 1 && f.cutout.opacity > 0);
check("DELETING keeps the fill (erased look)", f.fill.visible === true);
f = e3.frame(3, 680);
check("DELETING completes", f.done === true && f.cutout.opacity <= 0.001);

console.log("\nSINGLE SELECTION");
const e4 = createEditingIllusionEngine();
e4.select(10, 0);
e4.select(11, 5);
check(
  "selecting another deselects the first",
  e4.getState(10).selected === false && e4.getState(11).selected === true
);

console.log(`\n${fail === 0 ? "ALL PASS" : "FAILURES"} — ${pass} passed, ${fail} failed\n`);
process.exit(fail === 0 ? 0 : 1);
