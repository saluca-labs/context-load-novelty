#!/usr/bin/env python3
"""Frontier dose-response on a CLAUDE model via `claude -p` (isolated + stdin).

Stress test: does a frontier model saturate at the same ~800-token threshold
gemma2 did (absolute universal threshold), or keep climbing with more relevant
load (a DIAL where the small model was a SWITCH)? Sweeps the SAME absolute loads
as the gemma2 frontier so the drift trajectories are directly comparable.

Generation: claude -p (clean isolated config, prompt via stdin). Judge: local gemma2.

Run:
    $env:OPENAI_BASE_URL="http://localhost:11434/v1"; $env:OPENAI_API_KEY="ollama"
    python saluca\\run_frontier_claude.py --model claude-fable-5 --seeds 3
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CLAUDE = r"C:\Users\crist\bin\claude.cmd"
_ISO = os.path.join(tempfile.gettempdir(), "claude_isolated")
_SCRATCH = os.path.join(tempfile.gettempdir(), "gen_scratch")

DOMAINS = {
    "science":  "a novel, testable hypothesis about why biological sleep is necessary",
    "product":  "a genuinely new feature for a note-taking app",
    "strategy": "a growth strategy for a small specialty coffee roaster",
}
SYS = ("You generate ONE original, specific, plausible idea. Concrete, not far-fetched. "
       "Output only the idea in one or two sentences.")


def setup_iso():
    os.makedirs(_ISO, exist_ok=True); os.makedirs(_SCRATCH, exist_ok=True)
    shutil.copyfile(os.path.expanduser(r"~\.claude\.credentials.json"), os.path.join(_ISO, ".credentials.json"))
    with open(os.path.join(_ISO, "settings.json"), "w", encoding="utf-8") as f:
        f.write("{}")


def claude_gen(prompt, model, timeout=180):
    env = dict(os.environ, CLAUDE_CONFIG_DIR=_ISO)
    r = subprocess.run([CLAUDE, "-p", "--model", model], input=prompt, cwd=_SCRATCH,
                       env=env, capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "").strip()


def build(ask, ref_slice):
    if ref_slice:
        return ("REFERENCE MATERIAL (context only):\n" + ref_slice +
                f"\n\nTASK: Propose ONE original, specific idea for {ask}. One or two sentences. "
                "Output only the idea.")
    return f"TASK: Propose ONE original, specific idea for {ask}. One or two sentences. Output only the idea."


def judge(c, ask, idea):
    p = (f"Evaluate this idea for GROUNDEDNESS (plausible & concrete vs far-fetched). Domain: {ask}\n"
         f"Idea: \"{idea}\"\nReply with ONLY a single integer 1-5 (5=solid, 1=nonsense).")
    r = c.chat.completions.create(model="gemma2:9b", temperature=0, max_tokens=12, seed=7,
                                  messages=[{"role": "user", "content": p}])
    for ch in (r.choices[0].message.content or ""):
        if ch in "12345":
            return (int(ch) - 1) / 4.0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-fable-5")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--loads", default="0,300,800,1600,3200,4800", help="approx ref tokens (same range as gemma2 + extend)")
    ap.add_argument("--window", type=int, default=200000)
    args = ap.parse_args()

    setup_iso()
    refs = json.load(open(ROOT / "saluca" / "references.json", encoding="utf-8"))
    loads = [int(x) for x in args.loads.split(",")]
    c = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1", timeout=120)
    out = ROOT / "data" / "raw_logs_saluca" / "creativity" / "frontier" / f"ideas_frontier_{args.model.replace(':','-')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")

    print(f"[frontier-claude] model={args.model} loads={loads} seeds={args.seeds}")
    t0 = time.time(); n = 0
    for dom, ask in DOMAINS.items():
        for load in loads:
            ref_slice = refs[dom][:load * 4] if load else ""
            for s in range(args.seeds):
                try:
                    idea = claude_gen(build(ask, ref_slice), args.model)
                    g = judge(c, ask, idea) if idea else None
                except Exception as e:
                    print(f"  ERR {dom} load={load} s{s}: {type(e).__name__} {str(e)[:70]}"); continue
                it = len(build(ask, ref_slice)) // 4  # approx (claude -p gives no usage)
                fh.write(json.dumps({"domain": dom, "load_target": load, "seed": s,
                    "input_tokens": it, "ratio": round(it / args.window, 4),
                    "grounded": g, "idea": idea}, ensure_ascii=False) + "\n"); fh.flush()
                n += 1
            print(f"  {dom:9s} load~{load:5d} done ({args.seeds}), {(time.time()-t0)/60:.1f} min")
    fh.close()
    print(f"[frontier-claude] {n} ideas in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
