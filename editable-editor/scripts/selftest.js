/**
 * Headless regression test for the Object Manager + Visual Object Resolver
 * logic (selection, escalation, activation mutual-exclusion, transforms, text).
 *
 * Pixel-perfect hit testing is stubbed with `allTrue` (bbox-level) since real
 * alpha sampling needs a browser canvas. Run:  npm run om:test
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createObjectManager } from "../src/objectManager/index.js";
import { createVisualResolver } from "../src/visualResolver/index.js";
import { center } from "../src/objectManager/geometry.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const raw = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, "../public/layers/metadata.json"), "utf8")
);

const om = createObjectManager(raw);
const vr = createVisualResolver(om);
const allTrue = () => true;

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

// A nested leaf that is the top-ranked object at its own center (deterministic).
const leaf = om
  .getEditableObjects()
  .filter((o) => o.role === "part" && o.parentId != null)
  .find((l) => om.pickAt(center(l.bbox), { isOpaqueAt: allTrue })?.id === l.id);
const parent = om.getObject(leaf.parentId);
const p = center(leaf.bbox);

console.log("\nSELECTION");
check("click leaf center selects the leaf", om.pickAt(p, { isOpaqueAt: allTrue })?.id === leaf.id);
check(
  "re-click escalates leaf → parent",
  om.pickAt(p, { currentSelectionId: leaf.id, isOpaqueAt: allTrue })?.id === parent.id
);
check("click far outside → null", om.pickAt({ x: -50, y: -50 }, { isOpaqueAt: allTrue }) === null);

console.log("\nACTIVATION (hierarchy mutual-exclusion)");
let s = om.activate(om.createSession(), leaf.id).session;
check("leaf is active", s.entries[leaf.id]?.state === "active");
s = om.activate(s, parent.id).session;
check("activating parent deactivates its child", !s.entries[leaf.id]);
check("parent is active", s.entries[parent.id]?.state === "active");
s = om.activate(s, leaf.id).session;
check("activating child deactivates the parent", !s.entries[parent.id]);

console.log("\nVISUAL RESOLVER");
const scene = vr.resolveScene(s);
check("scene has exactly one active visual", scene.activeVisuals.length === 1);
check("active visual is the leaf", scene.activeVisuals[0]?.objectId === leaf.id);
check(
  "lift sits exactly on the native footprint",
  scene.activeVisuals[0]?.transform.x === leaf.bbox.x &&
    scene.activeVisuals[0]?.transform.y === leaf.bbox.y
);
check("base is the original image", scene.base.file === "background.png");

console.log("\nEDITS");
let s2 = om.applyTransform(om.activate(om.createSession(), leaf.id).session, leaf.id, { x: 999 });
check("applyTransform updates geometry", s2.entries[leaf.id].transform.x === 999);

const textObj = om.getEditableObjects().find((o) => o.role === "text");
let s3 = om.setText(om.activate(om.createSession(), textObj.id).session, textObj.id, "HELLO");
const tv = vr.resolve(textObj.id, s3);
check("text object resolves as text", tv.isText === true);
check("setText is reflected in the visual", tv.text === "HELLO");

console.log("\nDELETE + Z-ORDER");
let sd = om.activate(om.createSession(), leaf.id).session;
sd = om.attachRepair(sd, leaf.id, {
  status: "ready",
  dataUrl: "data:image/png;base64,AA",
  bbox: { x: leaf.bbox.x, y: leaf.bbox.y, w: leaf.bbox.w, h: leaf.bbox.h },
});
sd = om.softDelete(sd, leaf.id);
const sceneD = vr.resolveScene(sd);
check("deleted object renders no active visual", !sceneD.activeVisuals.some((v) => v.objectId === leaf.id));
check("deleted object still shows its repair patch (erased)", sceneD.repairs.some((r) => r.objectId === leaf.id));

const flower = om.getEditableObjects().find((o) => o.category === "flower");
let sz = om.activate(om.activate(om.createSession(), flower.id).session, leaf.id).session;
const orderA = vr.resolveScene(sz).activeVisuals.map((v) => v.objectId);
check("newest activation renders on top", orderA[orderA.length - 1] === leaf.id);
sz = om.sendToBack(sz, leaf.id);
check("sendToBack moves object to the back", vr.resolveScene(sz).activeVisuals[0].objectId === leaf.id);
sz = om.bringToFront(sz, leaf.id);
const orderC = vr.resolveScene(sz).activeVisuals.map((v) => v.objectId);
check("bringToFront moves object to the front", orderC[orderC.length - 1] === leaf.id);

console.log(`\n${fail === 0 ? "ALL PASS" : "FAILURES"} — ${pass} passed, ${fail} failed\n`);
process.exit(fail === 0 ? 0 : 1);
