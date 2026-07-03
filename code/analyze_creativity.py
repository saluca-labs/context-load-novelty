#!/usr/bin/env python3
"""Analyze the creativity-ratio sweep: find the productive band (if any).

Per (domain, load): groundedness = mean judge score; novelty = mean embedding
drift from the domain's LOW-LOAD baseline centroid (how far ideas wander from the
conventional answer). Aggregates across domains, then reports the load/ratio where
novelty is elevated AND groundedness still holds.

Verdict:
  - PRODUCTIVE BAND EXISTS if some mid/high load has novelty clearly above baseline
    while groundedness stays high -> that ratio is the "creativity ratio."
  - NO BAND (hypothesis dead) if novelty and groundedness fall together (novelty only
    rises where groundedness collapses).

Usage:
    .\.venv\Scripts\python.exe saluca\analyze_creativity.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data" / "raw_logs_saluca" / "creativity"
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def embed(texts):
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    return np.array(r.json()["embeddings"], dtype=float)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def analyze_file(path: Path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("idea")]
    if not rows:
        print(f"  {path.name}: no ideas"); return
    embs = embed([r["idea"] for r in rows])
    for r, e in zip(rows, embs):
        r["_e"] = e

    # per-domain baseline centroid = mean embedding at that domain's smallest load
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
    for dom, rs in by_domain.items():
        base_load = min(x["load_target"] for x in rs)
        base = np.mean([x["_e"] for x in rs if x["load_target"] == base_load], axis=0)
        for x in rs:
            x["_novelty"] = 1.0 - cos(x["_e"], base)  # drift from conventional answer

    # aggregate across domains per load (novelty already per-domain relative)
    by_load = defaultdict(list)
    for r in rows:
        by_load[r["load_target"]].append(r)

    def divergence(rs):
        """Mean pairwise cosine distance among a load's ideas (non-tautological
        novelty: how spread-out / non-clustered the ideas are). Needs n>=2."""
        es = [x["_e"] for x in rs]
        if len(es) < 2:
            return float("nan")
        ds = [1.0 - cos(es[i], es[j]) for i in range(len(es)) for j in range(i + 1, len(es))]
        return float(np.mean(ds))

    print(f"\n=== {path.stem.replace('ideas_','')} ===")
    print(f"{'load':>6} {'ratio':>6} {'n':>3} {'grounded':>9} {'diverge':>8} {'drift':>7} {'prod':>7}")
    print("-" * 52)
    stats = []
    for load in sorted(by_load):
        rs = by_load[load]
        g = np.mean([x["grounded"] for x in rs if x["grounded"] is not None]) if rs else float("nan")
        div = divergence(rs)                       # primary novelty (intra-load spread)
        drift = float(np.mean([x["_novelty"] for x in rs]))  # secondary (drift from baseline)
        ratio = float(np.mean([x["ratio"] for x in rs]))
        prod = div * g if not (np.isnan(g) or np.isnan(div)) else float("nan")
        stats.append((load, ratio, len(rs), g, div, drift, prod))
        print(f"{load:6d} {ratio:6.2f} {len(rs):3d} {g:9.3f} {div:8.3f} {drift:7.3f} {prod:7.3f}")

    # band = peak divergence at a NON-baseline load, meaningfully above baseline
    # spread, with groundedness still high. Requires n>=4/cell to mean anything.
    base_div = stats[0][4]
    valid = [s for s in stats if not np.isnan(s[6])]
    if not valid or min(s[2] for s in stats) < 4:
        print("  -> INCONCLUSIVE (need n>=4 per load; this is a plumbing check only)")
        return
    peak = max(valid, key=lambda s: s[6])
    if peak[0] != stats[0][0] and peak[3] >= 0.6 and peak[4] > base_div * 1.3:
        print(f"  -> PRODUCTIVE BAND near load~{peak[0]} (ratio {peak[1]:.2f}): "
              f"diverge {peak[4]:.3f} vs baseline {base_div:.3f}, grounded {peak[3]:.2f}")
    else:
        print(f"  -> no clean band (peak diverge at load {peak[0]}, grounded {peak[3]:.2f}, "
              f"diverge {peak[4]:.3f} vs base {base_div:.3f})")


def main() -> int:
    files = sorted(LOG_DIR.glob("ideas_*.jsonl"))
    if not files:
        print(f"No idea logs in {LOG_DIR}. Run creativity_ratio.py first.")
        return 1
    for f in files:
        analyze_file(f)
    print("\nRead: novelty = drift from the low-load conventional answer; productive =")
    print("novelty x groundedness. A peak at mid/high load with grounded>=0.6 = the")
    print("creativity ratio. Novelty rising only as grounded falls = no band (falsified).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
