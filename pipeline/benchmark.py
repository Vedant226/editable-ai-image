"""
Category-agnostic benchmark harness.

Runs the full pipeline over every image in pipeline/benchmark_images/ and reports
NO-REFERENCE quality metrics (no ground-truth labels are required), so it works
on arbitrary AI-generated images of any category. Drop images into that folder
and re-run; the harness is identical regardless of content.

Metrics per image:
  proposals, canonical, dedup_ratio       (duplicate suppression)
  editable, editable_coverage             (editable coverage)
  avg_confidence, pct_uncertain           (semantic accuracy proxy)
  avg_solidity                            (mask quality proxy)
  bitmap_text                             (text preservation)
  timings                                 (performance)

  python -m pipeline.benchmark
"""

import json
import os

import numpy as np

from . import config as C
from . import engine

BENCH_DIR = os.path.join(C.ROOT, "pipeline", "benchmark_images")
OUT = os.path.join(C.ROOT, "pipeline", "_work", "benchmark")
EXTS = (".png", ".jpg", ".jpeg", ".webp")

METRICS = [
    "proposals", "canonical", "dedup_ratio", "editable", "editable_coverage",
    "avg_confidence", "pct_uncertain", "avg_solidity", "bitmap_text",
]


def collect():
    imgs = []
    if os.path.isdir(BENCH_DIR):
        imgs = [os.path.join(BENCH_DIR, f) for f in sorted(os.listdir(BENCH_DIR)) if f.lower().endswith(EXTS)]
    return imgs or [C.IMAGE_PATH]


def main():
    os.makedirs(OUT, exist_ok=True)
    images = collect()
    print(f"benchmark: {len(images)} image(s); loading models once ...")
    engine.load_all()

    rows = []
    for img in images:
        name = os.path.splitext(os.path.basename(img))[0]
        try:
            s = engine.process_image(img, os.path.join(OUT, name))
        except Exception as exc:  # noqa: BLE001 — a failed image must not abort the suite
            print(f"  [FAIL] {name}: {exc}")
            s = {"image": os.path.basename(img), "error": str(exc)}
        rows.append(s)
        if "error" not in s:
            print(f"  {name:22} prop={s['proposals']:4} canon={s['canonical']:4} edit={s['editable']:4} "
                  f"cov={s['editable_coverage']:.2f} conf={s['avg_confidence']:.2f} "
                  f"unc={s['pct_uncertain']:.2f} sol={s['avg_solidity']:.2f} txt={s['bitmap_text']:2} "
                  f"t={s['timings']['total']:.0f}s")

    ok = [r for r in rows if "error" not in r]
    avg = {m: round(float(np.mean([r[m] for r in ok])), 3) for m in METRICS} if ok else {}
    report = {"n_images": len(images), "n_ok": len(ok), "average": avg, "images": rows}
    json.dump(report, open(os.path.join(OUT, "benchmark_report.json"), "w"), indent=1)
    _html(report)

    print("\n" + "=" * 64)
    print("BENCHMARK AVERAGE")
    print("=" * 64)
    for m in METRICS:
        if m in avg:
            print(f"  {m:20} {avg[m]}")
    print(f"\n  report -> pipeline/_work/benchmark/benchmark_report.json + .html")
    print("=" * 64 + "\n")


def _html(report):
    rows = report["images"]
    cols = ["image"] + METRICS + ["total_s"]
    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{background:#15110d;color:#e8d9bf;font:13px system-ui;margin:18px}",
        "table{border-collapse:collapse}td,th{border:1px solid #3a2f1c;padding:5px 9px;text-align:right}",
        "th{color:#d8b36a}td:first-child,th:first-child{text-align:left}tr.avg{color:#d8b36a;font-weight:700}",
        "</style></head><body><h1 style='color:#d8b36a'>Pipeline Benchmark</h1><table><tr>",
        "".join(f"<th>{c}</th>" for c in cols), "</tr>",
    ]
    for r in rows:
        if "error" in r:
            parts.append(f"<tr><td>{r['image']}</td><td colspan='{len(cols)-1}'>ERROR: {r['error']}</td></tr>")
            continue
        cells = [r["image"]] + [r.get(m, "") for m in METRICS] + [r["timings"]["total"]]
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    if report.get("average"):
        cells = ["AVERAGE"] + [report["average"].get(m, "") for m in METRICS] + [""]
        parts.append("<tr class='avg'>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    parts.append("</table></body></html>")
    open(os.path.join(OUT, "benchmark_report.html"), "w").write("".join(parts))


if __name__ == "__main__":
    main()
