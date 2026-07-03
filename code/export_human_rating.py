#!/usr/bin/env python3
"""Export a BLINDED human-rating sheet for groundedness.

The disputed claim is whether groundedness erodes with load on the frontier model
(opus says yes 0.94->0.72; gemma2 says flat). A human rating blind to load settles
it. This writes a shuffled CSV (id, domain, idea, groundedness) with load/condition
HIDDEN, plus a private key for merge-back. Rate in Excel, then run
merge_human_rating.py.

Default target: Fable frontier (the decisive set). Use --all for everything.

Run: python saluca\\export_human_rating.py            # Fable frontier (54)
     python saluca\\export_human_rating.py --all       # all 390
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "data" / "raw_logs_saluca" / "creativity"
OUT = CDIR / "human_rating"
FILES_KEY = {
    "fable_frontier": CDIR / "frontier" / "ideas_frontier_claude-fable-5.jsonl",
    "gemma2_frontier": CDIR / "frontier" / "ideas_frontier_gemma2-9b.jsonl",
    "gemma2_iso": CDIR / "ideas_ollama_gemma2-9b_isolation.jsonl",
    "haiku_iso": CDIR / "ideas_claude_cli_haiku_isolation.jsonl",
    "opus_iso": CDIR / "ideas_claude_cli_opus_isolation.jsonl",
    "fable_iso": CDIR / "ideas_claude_cli_claude-fable-5_isolation.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    which = list(FILES_KEY) if args.all else ["fable_frontier"]

    random.seed(42)
    rows_out, key = [], {}
    n = 0
    for tag in which:
        path = FILES_KEY[tag]
        if not path.exists():
            continue
        for r in (json.loads(l) for l in path.open(encoding="utf-8") if l.strip()):
            if not r.get("idea"):
                continue
            rid = f"R{n:04d}"; n += 1
            rows_out.append({"id": rid, "domain": r["domain"], "idea": r["idea"], "groundedness_1to5": ""})
            key[rid] = {"file": str(path), "source": tag, "domain": r["domain"],
                        "load_target": r.get("load_target"), "condition": r.get("condition"),
                        "seed": r.get("seed"), "grounded_gemma2": r.get("grounded"),
                        "grounded_opus": r.get("grounded_strong")}
    random.shuffle(rows_out)

    csv_path = OUT / "rate_me.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "domain", "idea", "groundedness_1to5"])
        w.writeheader(); w.writerows(rows_out)
    (OUT / "_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"wrote {len(rows_out)} blinded ideas -> {csv_path}")
    print(f"private key -> {OUT / '_key.json'} (do NOT peek before rating)")
    print("Rate the 'groundedness_1to5' column (1-5), save, then run merge_human_rating.py")


if __name__ == "__main__":
    main()
