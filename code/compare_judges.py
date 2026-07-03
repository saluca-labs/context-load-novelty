#!/usr/bin/env python3
"""Compare the original gemma2 judge vs the stronger opus judge on groundedness.

The novelty/drift results are judge-independent; only groundedness depends on the
judge. This checks whether the paper's claim -- groundedness HOLDS across
conditions/loads with no coherence cliff -- survives a stronger, more
discriminating evaluator (which we expect to score lower and with more range).
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "data" / "raw_logs_saluca" / "creativity"
ISO = {"gemma2:9b": "ideas_ollama_gemma2-9b_isolation.jsonl",
       "haiku": "ideas_claude_cli_haiku_isolation.jsonl",
       "opus": "ideas_claude_cli_opus_isolation.jsonl",
       "Fable": "ideas_claude_cli_claude-fable-5_isolation.jsonl"}
FRO = {"gemma2:9b": CDIR / "frontier" / "ideas_frontier_gemma2-9b.jsonl",
       "Fable": CDIR / "frontier" / "ideas_frontier_claude-fable-5.jsonl"}
CONDS = ["plain", "content", "instruction", "functional"]


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("nan")


print("=" * 62)
print("ISOLATION — groundedness by condition:  gemma2-judge / opus-judge")
print("=" * 62)
for model, fn in ISO.items():
    p = CDIR / fn
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    by = defaultdict(list)
    for r in rows:
        if r.get("idea"):
            by[r["condition"]].append(r)
    cells = "  ".join(
        f"{c[:4]}:{mean([r.get('grounded') for r in by[c]]):.2f}/{mean([r.get('grounded_strong') for r in by[c]]):.2f}"
        for c in CONDS if by.get(c))
    print(f"  {model:11s} {cells}")

print()
print("=" * 62)
print("FRONTIER — groundedness by load (tokens):  gemma2 / opus")
print("=" * 62)
for model, p in FRO.items():
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    by = defaultdict(list)
    for r in rows:
        if r.get("idea"):
            by[r["load_target"]].append(r)
    cells = "  ".join(
        f"{L}:{mean([r.get('grounded') for r in by[L]]):.2f}/{mean([r.get('grounded_strong') for r in by[L]]):.2f}"
        for L in sorted(by))
    print(f"  {model:11s} {cells}")

print()
# overall correlation + level shift
allr = []
for fn in list(ISO.values()):
    p = CDIR / fn
    if p.exists():
        allr += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
for p in FRO.values():
    if p.exists():
        allr += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
pairs = [(r["grounded"], r["grounded_strong"]) for r in allr
         if r.get("grounded") is not None and r.get("grounded_strong") is not None]
if pairs:
    g, o = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    corr = float(np.corrcoef(g, o)[0, 1])
    print(f"overall: gemma2 mean {g.mean():.3f}  opus mean {o.mean():.3f}  "
          f"(opus range {o.min():.2f}-{o.max():.2f}, sd {o.std():.2f} vs gemma2 sd {g.std():.2f})  "
          f"corr={corr:.2f}  n={len(pairs)}")
