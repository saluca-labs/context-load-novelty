#!/usr/bin/env python3
"""Merge the filled human-rating CSV back and re-analyze groundedness.

Reads rate_me.csv (with your 1-5 ratings) + _key.json, writes grounded_human into
the source logs, and prints the decisive comparison: for the Fable frontier, does
HUMAN-rated groundedness decline with load (corroborating opus's 0.94->0.72) or
stay flat (corroborating gemma2)? Also reports human-vs-opus and human-vs-gemma2
correlation.

Run: python saluca\\merge_human_rating.py
"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw_logs_saluca" / "creativity" / "human_rating"


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("nan")


def main():
    key = json.loads((OUT / "_key.json").read_text(encoding="utf-8"))
    ratings = {}
    with (OUT / "rate_me.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("groundedness_1to5") or "").strip()
            if v and v[0] in "12345":
                ratings[row["id"]] = (int(v[0]) - 1) / 4.0
    print(f"parsed {len(ratings)}/{len(key)} human ratings")
    if not ratings:
        print("No ratings found. Fill the groundedness_1to5 column in rate_me.csv first.")
        return

    # write grounded_human back into source logs (match by domain+load+seed+condition)
    by_file = defaultdict(dict)
    for rid, meta in key.items():
        if rid in ratings:
            k = (meta["domain"], meta["load_target"], meta["seed"], meta["condition"])
            by_file[meta["file"]][k] = ratings[rid]
    for fpath, kmap in by_file.items():
        p = Path(fpath)
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        for r in rows:
            k = (r.get("domain"), r.get("load_target"), r.get("seed"), r.get("condition"))
            if k in kmap:
                r["grounded_human"] = kmap[k]
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # decisive readout: Fable frontier groundedness by load, three judges
    rated = [(rid, key[rid], ratings[rid]) for rid in ratings if key[rid]["source"] == "fable_frontier"]
    if rated:
        print("\nFABLE FRONTIER groundedness by load:  human / opus / gemma2")
        by_load = defaultdict(list)
        for rid, meta, h in rated:
            by_load[meta["load_target"]].append((h, meta["grounded_opus"], meta["grounded_gemma2"]))
        for L in sorted(by_load):
            hs = by_load[L]
            print(f"  load {L:5d} (n={len(hs)}):  "
                  f"{mean([x[0] for x in hs]):.2f} / {mean([x[1] for x in hs]):.2f} / {mean([x[2] for x in hs]):.2f}")

    # correlations across all rated
    trip = [(ratings[rid], key[rid]["grounded_opus"], key[rid]["grounded_gemma2"]) for rid in ratings]
    trip = [(h, o, g) for h, o, g in trip if o is not None and g is not None]
    if len(trip) > 2:
        h = np.array([t[0] for t in trip]); o = np.array([t[1] for t in trip]); g = np.array([t[2] for t in trip])
        print(f"\ncorr human-opus = {np.corrcoef(h,o)[0,1]:.2f}   "
              f"human-gemma2 = {np.corrcoef(h,g)[0,1]:.2f}   (n={len(trip)})")
        print(f"means: human {h.mean():.3f}  opus {o.mean():.3f}  gemma2 {g.mean():.3f}")


if __name__ == "__main__":
    main()
