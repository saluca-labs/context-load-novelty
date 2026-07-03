#!/usr/bin/env python3
"""Analyze the frontier dose-response: dial vs switch vs peak.

Per load level (aggregated across domains; drift computed per-domain vs that
domain's load-0 baseline centroid):
  grounded   = mean judge score
  drift      = mean embedding distance from the load-0 conventional answer
  divergence = mean intra-load pairwise spread
  ratio      = mean input_tokens / window

Shape verdict:
  DIAL   = drift rises across loads (more load -> more off-mainstream)
  SWITCH = drift jumps early then flat
  PEAK   = drift rises then falls (a productive band / optimal ratio)
  + watch grounded: if it falls at high load, that's the coherence ceiling.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, requests

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data" / "raw_logs_saluca" / "creativity" / "frontier"


def embed(texts):
    r = requests.post("http://localhost:11434/api/embed",
                      json={"model": "nomic-embed-text", "input": texts}, timeout=180)
    r.raise_for_status()
    return np.array(r.json()["embeddings"], dtype=float)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def analyze(path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("idea")]
    if not rows:
        print(f"  {path.name}: empty"); return
    for r, e in zip(rows, embed([r["idea"] for r in rows])):
        r["_e"] = e
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    for dom, rs in by_dom.items():
        base_load = min(x["load_target"] for x in rs)
        base = np.mean([x["_e"] for x in rs if x["load_target"] == base_load], axis=0)
        for x in rs:
            x["_drift"] = 1.0 - cos(x["_e"], base)
    by_load = defaultdict(list)
    for r in rows:
        by_load[r["load_target"]].append(r)

    print(f"\n=== {path.stem.replace('ideas_frontier_','')} ===")
    print(f"{'load':>6} {'ratio':>6} {'n':>3} {'grounded':>9} {'drift':>7} {'diverge':>8}")
    print("-" * 46)
    stats = []
    for load in sorted(by_load):
        rs = by_load[load]
        g = np.mean([x["grounded"] for x in rs if x["grounded"] is not None])
        drift = float(np.mean([x["_drift"] for x in rs]))
        es = [x["_e"] for x in rs]
        div = float(np.mean([1 - cos(es[i], es[j]) for i in range(len(es)) for j in range(i+1, len(es))])) if len(es) > 1 else float("nan")
        ratio = float(np.mean([x["ratio"] for x in rs]))
        stats.append((load, ratio, g, drift, div))
        print(f"{load:6d} {ratio:6.2f} {len(rs):3d} {g:9.3f} {drift:7.3f} {div:8.3f}")

    drifts = [s[3] for s in stats]
    peak_i = int(np.argmax(drifts))
    last = len(stats) - 1
    verdict = ("PEAK (productive band)" if 0 < peak_i < last and drifts[peak_i] > drifts[last] * 1.15
               else "DIAL (rises to max load)" if peak_i == last and drifts[last] > drifts[0] * 1.3
               else "SWITCH (jumps then flat)" if drifts[-1] > drifts[0] * 1.2 and peak_i <= 1
               else "flat / weak")
    gtrend = "grounded HOLDS" if stats[-1][2] >= stats[0][2] - 0.15 else "grounded FALLS at high load"
    print(f"  -> {verdict}; peak drift at load {stats[peak_i][0]} (ratio {stats[peak_i][1]:.2f}); {gtrend}")


def main():
    files = sorted(LOG_DIR.glob("ideas_frontier_*.jsonl"))
    if not files:
        print(f"No frontier logs in {LOG_DIR}. Run run_frontier.py first."); return 1
    for f in files:
        analyze(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
