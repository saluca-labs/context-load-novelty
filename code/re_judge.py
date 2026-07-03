#!/usr/bin/env python3
"""Strengthening pass: re-score groundedness of every logged idea with a STRONGER
judge (opus via claude -p, isolated), batched to stay fast. Adds `grounded_strong`
alongside the original gemma2 `grounded`, and prints a comparison so we can see
whether the paper's "groundedness holds" claim survives a better evaluator.

Note: opus is also a generator for one condition-set; opus-judging-opus rows are
mildly self-referential (1 of 4 models) and flagged in the paper.

Run:
    python saluca\\re_judge.py --judge opus
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "data" / "raw_logs_saluca" / "creativity"
CLAUDE = r"C:\Users\crist\bin\claude.cmd"
_ISO = os.path.join(tempfile.gettempdir(), "claude_isolated")
_SCRATCH = os.path.join(tempfile.gettempdir(), "gen_scratch")

FILES = [CDIR / "ideas_ollama_gemma2-9b_isolation.jsonl",
         CDIR / "ideas_claude_cli_haiku_isolation.jsonl",
         CDIR / "ideas_claude_cli_opus_isolation.jsonl",
         CDIR / "ideas_claude_cli_claude-fable-5_isolation.jsonl",
         CDIR / "frontier" / "ideas_frontier_gemma2-9b.jsonl",
         CDIR / "frontier" / "ideas_frontier_claude-fable-5.jsonl"]


def setup_iso():
    os.makedirs(_ISO, exist_ok=True); os.makedirs(_SCRATCH, exist_ok=True)
    shutil.copyfile(os.path.expanduser(r"~\.claude\.credentials.json"), os.path.join(_ISO, ".credentials.json"))
    with open(os.path.join(_ISO, "settings.json"), "w", encoding="utf-8") as f:
        f.write("{}")


def judge_batch(ideas, model):
    numbered = "\n".join(f"{i+1}. {idea}" for i, idea in enumerate(ideas))
    prompt = ("Rate each idea below for GROUNDEDNESS: 5 = plausible, concrete, could be real; "
              "1 = far-fetched, incoherent, or nonsensical. Consider only plausibility/coherence, "
              "not novelty. Return ONLY a JSON array of integers (1-5), one per idea, in order, "
              f"length {len(ideas)}. No other text.\n\nIDEAS:\n{numbered}")
    env = dict(os.environ, CLAUDE_CONFIG_DIR=_ISO)
    r = subprocess.run([CLAUDE, "-p", "--model", model], input=prompt, cwd=_SCRATCH,
                       env=env, capture_output=True, text=True, timeout=180)
    m = re.search(r"\[[\s\d,]+\]", r.stdout or "")
    if not m:
        return [None] * len(ideas)
    try:
        arr = json.loads(m.group(0))
        arr = [(int(x) - 1) / 4.0 if isinstance(x, (int, float)) and 1 <= x <= 5 else None for x in arr]
        return (arr + [None] * len(ideas))[:len(ideas)]
    except Exception:
        return [None] * len(ideas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="opus")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    setup_iso()
    t0 = time.time()
    for path in FILES:
        if not path.exists():
            print(f"skip (missing): {path.name}"); continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        ideas_idx = [(i, r) for i, r in enumerate(rows) if r.get("idea")]
        scores = {}
        for b in range(0, len(ideas_idx), args.batch):
            chunk = ideas_idx[b:b + args.batch]
            res = judge_batch([r["idea"] for _, r in chunk], args.judge)
            for (i, _), s in zip(chunk, res):
                scores[i] = s
        for i, r in enumerate(rows):
            r["grounded_strong"] = scores.get(i)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        got = [v for v in scores.values() if v is not None]
        print(f"{path.name}: re-judged {len(got)}/{len(ideas_idx)} "
              f"({(time.time()-t0)/60:.1f} min elapsed)")
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
