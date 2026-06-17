/**
 * Phase 0 dev report — runs the Object Manager over the REAL metadata.json and
 * prints what it decided, so the semantics can be reviewed before any UI work.
 *
 * Usage:  npm run om:report        (from editable-editor/)
 *
 * Read-only: loads metadata.json, mutates nothing.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createObjectManager } from "../src/objectManager/index.js";
import { center } from "../src/objectManager/geometry.js";
import { isContainerDominated } from "../src/objectManager/classify.js";
import { ROLE } from "../src/objectManager/vocabulary.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const METADATA_PATH = path.resolve(__dirname, "../public/layers/metadata.json");

const raw = JSON.parse(fs.readFileSync(METADATA_PATH, "utf8"));
const om = createObjectManager(raw);
const stats = om.getStats();
const { width, height } = om.getImageSize();

const pad = (s, n) => String(s).padEnd(n).slice(0, n);
const padL = (s, n) => String(s).padStart(n);
const pct = (f) => `${(f * 100).toFixed(1)}%`;
const trunc = (s, n) => (String(s).length > n ? String(s).slice(0, n - 1) + "…" : String(s));
const rule = (c = "─") => console.log(c.repeat(78));

console.log("\n" + "═".repeat(78));
console.log(`  OBJECT MANAGER REPORT   image ${width}×${height}   raw objects: ${stats.rawCount}`);
console.log("═".repeat(78));

// ---- summary ----
console.log("\nSUMMARY");
rule();
console.log(`  editable (clickable):  ${stats.editableCount}`);
console.log(`  ignored (retained):    ${stats.ignoredCount}`);
for (const [reason, n] of Object.entries(stats.ignoredByReason).sort((a, b) => b[1] - a[1])) {
  console.log(`      ${pad(reason, 26)} ${padL(n, 3)}`);
}
console.log(`  duplicate collapses:   ${stats.collapsed.length}`);
console.log(`  soft demotions:        ${stats.demotions.length}`);
console.log(`  multi-member groups:   ${stats.groups.length}`);
console.log(`\n  editable by category:`);
for (const [cat, n] of Object.entries(stats.categoryCounts).sort((a, b) => b[1] - a[1])) {
  console.log(`      ${pad(cat, 14)} ${padL(n, 3)}`);
}

// ---- editable objects ----
console.log("\nEDITABLE OBJECTS  (most-specific first)");
rule();
console.log(
  `  ${pad("id", 6)}${pad("category", 12)}${pad("role", 9)}${pad("spec", 5)}${pad("area", 7)}${pad("group", 9)}${pad("parent", 8)}type`
);
const editable = om.getEditableObjects().slice().sort((a, b) => b.specificity - a.specificity || a.id - b.id);
for (const o of editable) {
  console.log(
    `  ${pad(o.id, 6)}${pad(o.category, 12)}${pad(o.role, 9)}${pad(o.specificity, 5)}${pad(pct(o.areaFraction), 7)}${pad(o.groupId || "-", 9)}${pad(o.parentId ?? "-", 8)}${trunc(o.rawType, 30)}`
  );
}

// ---- groups ----
console.log("\nGROUPS  (person ⊃ parts)");
rule();
if (!stats.groups.length) console.log("  (none)");
for (const g of stats.groups) {
  const parent = om.getObject(g.memberIds[0]);
  const kids = g.memberIds
    .slice(1)
    .map((id) => `${om.getObject(id).category}#${id}`)
    .join(", ");
  console.log(`  ${pad(g.groupId, 8)} ${parent.category}#${parent.id}  ⊃  [ ${kids} ]`);
}

// ---- duplicate collapses ----
console.log("\nDUPLICATES COLLAPSED  (canonical ← retained aliases)");
rule();
if (!stats.collapsed.length) console.log("  (none)");
for (const c of stats.collapsed) {
  console.log(`  ${pad(c.category, 12)} #${pad(c.canonicalId, 6)} ← [ ${c.aliasIds.map((i) => "#" + i).join(", ")} ]`);
}

// ---- demotions ----
console.log("\nDEMOTED PARENTS  (label container-dominated → treated as framed-portrait panel)");
rule();
if (!stats.demotions.length) console.log("  (none)");
for (const d of stats.demotions) {
  console.log(`  #${pad(d.id, 6)} ${pad(pct(d.areaFraction), 8)} ${trunc(d.type, 50)}`);
}

// ---- parent container-dominance audit ----
console.log("\nPARENT AUDIT  (struct = container/scene tokens, edit = editable tokens)");
rule();
console.log(`  ${pad("id", 6)}${pad("area", 8)}${pad("struct", 8)}${pad("edit", 6)}${pad("dominated", 11)}type`);
const parents = om.getEditableObjects().filter((o) => o.role === ROLE.PARENT);
for (const o of parents.sort((a, b) => b.areaFraction - a.areaFraction)) {
  const struct = o.matched.filter((m) => m.role === ROLE.CONTAINER || m.role === ROLE.SCENE).length;
  const edit = o.matched.filter((m) => m.editable).length;
  const dom = isContainerDominated(o.matched);
  console.log(
    `  ${pad(o.id, 6)}${pad(pct(o.areaFraction), 8)}${pad(struct, 8)}${pad(edit, 6)}${pad(dom ? "YES ←" : "no", 11)}${trunc(o.rawType, 28)}`
  );
}

// ---- selection sanity checks ----
console.log("\nSELECTION SANITY  (click center of a leaf → re-click escalates to parent)");
rule();
const leaves = om.getEditableObjects().filter((o) => o.role === "part" && o.parentId != null).slice(0, 5);
if (!leaves.length) console.log("  (no nested leaves found)");
for (const leaf of leaves) {
  const p = center(leaf.bbox);
  const first = om.pickAt(p);
  const second = om.pickAt(p, { currentSelectionId: first?.id });
  console.log(
    `  click ${leaf.category}#${leaf.id} center → picks ${first ? first.category + "#" + first.id : "null"}` +
      `  →  re-click escalates to ${second ? second.category + "#" + second.id : "null"}`
  );
}

console.log("\n" + "═".repeat(78) + "\n");
