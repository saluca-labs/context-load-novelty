#!/usr/bin/env python3
"""Frontier dose-response: does relevant context-LOAD move ideation more as it
rises toward the model's window (a DIAL), or step-and-plateau (a SWITCH), or
peak-then-collapse? Injects an increasing amount of RELEVANT reference material
(neutral, NO instruction -- the condition proven to move ideas) and measures
drift / groundedness / divergence vs load (as a fraction of the window).

Local gemma2 first (free): its 4096 window lets us reach ~78% load ratio cheaply.

Run:
    $env:OPENAI_BASE_URL="http://localhost:11434/v1"; $env:OPENAI_API_KEY="ollama"
    python saluca\\run_frontier.py --model gemma2:9b --seeds 6
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {
    "science":  "a novel, testable hypothesis about why biological sleep is necessary",
    "product":  "a genuinely new feature for a note-taking app",
    "strategy": "a growth strategy for a small specialty coffee roaster",
}
SYS = ("You generate ONE original, specific, plausible idea. Concrete, not far-fetched. "
       "Output only the idea in one or two sentences.")


def build(ask, ref_slice):
    if ref_slice:
        return ("REFERENCE MATERIAL (context only):\n" + ref_slice +
                f"\n\nTASK: Propose ONE original, specific idea for {ask}. One or two "
                "sentences. Output only the idea.")
    return f"TASK: Propose ONE original, specific idea for {ask}. One or two sentences. Output only the idea."


def judge(c, model, ask, idea):
    p = (f"Evaluate this idea for GROUNDEDNESS (plausible & concrete vs far-fetched). Domain: {ask}\n"
         f"Idea: \"{idea}\"\nReply with ONLY a single integer 1-5 (5=solid, 1=nonsense).")
    r = c.chat.completions.create(model=model, temperature=0, max_tokens=12, seed=7,
                                  messages=[{"role": "user", "content": p}])
    for ch in (r.choices[0].message.content or ""):
        if ch in "12345":
            return (int(ch) - 1) / 4.0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma2:9b")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--loads", default="0,300,800,1600,2400,3200", help="approx ref tokens to inject")
    ap.add_argument("--window", type=int, default=4096)
    args = ap.parse_args()

    refs = json.load(open(ROOT / "saluca" / "references.json", encoding="utf-8"))
    loads = [int(x) for x in args.loads.split(",")]
    c = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1", timeout=180)
    out = ROOT / "data" / "raw_logs_saluca" / "creativity" / "frontier" / f"ideas_frontier_{args.model.replace(':','-')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")

    print(f"[frontier] model={args.model} loads={loads} seeds={args.seeds} window={args.window}")
    t0 = time.time(); n = 0
    for dom, ask in DOMAINS.items():
        for load in loads:
            ref_slice = refs[dom][:load * 4] if load else ""   # ~4 chars/token
            for s in range(args.seeds):
                try:
                    r = c.chat.completions.create(model=args.model, temperature=0.7, max_tokens=220,
                        seed=2000 + load + s,
                        messages=[{"role": "system", "content": SYS},
                                  {"role": "user", "content": build(ask, ref_slice)}])
                    idea = (r.choices[0].message.content or "").strip()
                    g = judge(c, args.model, ask, idea) if idea else None
                    it = r.usage.prompt_tokens
                except Exception as e:
                    print(f"  ERR {dom} load={load} s{s}: {type(e).__name__} {str(e)[:70]}"); continue
                fh.write(json.dumps({"domain": dom, "load_target": load, "seed": s,
                    "input_tokens": it, "ratio": round(it / args.window, 3),
                    "grounded": g, "idea": idea}, ensure_ascii=False) + "\n"); fh.flush()
                n += 1
            print(f"  {dom:9s} load~{load:5d} done ({args.seeds})")
    fh.close()
    print(f"[frontier] {n} ideas in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
