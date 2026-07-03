#!/usr/bin/env python3
"""Isolation analysis: separate LOAD from INSTRUCTION.

Per condition (plain / content / instruction / functional), measured against the
PLAIN baseline centroid (per domain):
  drift      = mean distance of the condition's ideas from the plain centroid
               (how far off the mainstream answer the ideas moved)
  divergence = mean pairwise spread within the condition
  grounded   = mean judge score

Verdict logic:
  - content drifts ~ plain      -> LOAD alone does nothing; ratio hypothesis DEAD.
  - content drifts >> plain      -> relevant LOAD moves ideas even without instruction.
  - instruction ~ functional > content -> the effect is the INSTRUCTION, not the load.

Usage: python saluca\analyze_isolation.py
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
COND_ORDER = ["plain", "content", "instruction", "functional"]


def embed(texts):
    r = requests.post(EMBED_URL, json={"model": "nomic-embed-text", "input": texts}, timeout=180)
    r.raise_for_status()
    return np.array(r.json()["embeddings"], dtype=float)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def analyze(path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("idea") and r.get("condition")]
    if not rows:
        print(f"  {path.name}: no isolation ideas"); return
    for r, e in zip(rows, embed([r["idea"] for r in rows])):
        r["_e"] = e

    # per-domain plain centroid
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    plain_c = {}
    for dom, rs in by_dom.items():
        p = [x["_e"] for x in rs if x["condition"] == "plain"]
        plain_c[dom] = np.mean(p, axis=0) if p else None
    for r in rows:
        c = plain_c[r["domain"]]
        r["_drift"] = (1.0 - cos(r["_e"], c)) if c is not None else float("nan")

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    print(f"\n=== {path.stem.replace('ideas_','')} ===")
    print(f"{'condition':12s} {'n':>3} {'grounded':>9} {'drift':>7} {'diverge':>8} {'in_tok':>7}")
    print("-" * 50)
    res = {}
    for c in COND_ORDER:
        rs = by_cond.get(c, [])
        if not rs:
            continue
        g = np.mean([x["grounded"] for x in rs if x["grounded"] is not None])
        drift = float(np.mean([x["_drift"] for x in rs]))
        es = [x["_e"] for x in rs]
        div = float(np.mean([1 - cos(es[i], es[j]) for i in range(len(es)) for j in range(i+1, len(es))])) if len(es) > 1 else float("nan")
        tok = float(np.mean([x["input_tokens"] for x in rs]))
        res[c] = (g, drift, div)
        print(f"{c:12s} {len(rs):3d} {g:9.3f} {drift:7.3f} {div:8.3f} {tok:7.0f}")

    if all(k in res for k in ("plain", "content", "instruction", "functional")):
        cd, idr, fd = res["content"][1], res["instruction"][1], res["functional"][1]
        base = res["plain"][1]  # ~0 by construction; use content-vs-instruction contrast
        print("  verdict:")
        load_effect = cd > 0.12 and cd > 0.6 * fd          # content moved a lot on its own
        instr_effect = idr > 0.12
        if load_effect:
            print(f"    -> LOAD ITSELF MOVES IDEAS: content drift {cd:.3f} (vs functional {fd:.3f}, "
                  f"instruction {idr:.3f}) — ratio hypothesis ALIVE, worth the cloud frontier test.")
        elif instr_effect and cd < 0.6 * idr:
            print(f"    -> EFFECT IS THE INSTRUCTION: content {cd:.3f} << instruction {idr:.3f} "
                  f"~ functional {fd:.3f}. Load-without-instruction does little — ratio hypothesis WEAK.")
        else:
            print(f"    -> mixed: content {cd:.3f}, instruction {idr:.3f}, functional {fd:.3f} "
                  f"(threshold 0.12). Inspect ideas.")


def main() -> int:
    files = sorted(LOG_DIR.glob("ideas_*_isolation.jsonl"))
    if not files:
        print(f"No isolation logs in {LOG_DIR}. Run creativity_ratio.py --load-type isolation first.")
        return 1
    for f in files:
        analyze(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
