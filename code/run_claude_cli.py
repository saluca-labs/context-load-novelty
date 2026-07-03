#!/usr/bin/env python3
"""Run the creativity isolation test on Claude models via `claude -p` (headless).

Uses the Claude Code SUBSCRIPTION (no API key, sidesteps the failing vault keys),
and judges each idea with LOCAL gemma2 via Ollama (fast, decoupled = no self-judge).
Logs the same JSONL schema as creativity_ratio.py so analyze_isolation.py just works.

Run:
    $env:OPENAI_BASE_URL="http://localhost:11434/v1"; $env:OPENAI_API_KEY="ollama"
    python saluca\run_claude_cli.py --model haiku --seeds 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CLAUDE = r"C:\Users\crist\bin\claude.cmd"

# Isolation: run `claude -p` against a CLEAN config dir (credentials only, empty
# settings) from a scratch cwd, so our SessionStart soul-hook + MCP + project
# config don't bleed into the generation. Prompt goes via STDIN (arg-passing
# truncates multi-line prompts at the first newline). Both proven necessary.
_ISO = os.path.join(tempfile.gettempdir(), "claude_isolated")
_SCRATCH = os.path.join(tempfile.gettempdir(), "gen_scratch")


def _setup_isolation():
    os.makedirs(_ISO, exist_ok=True)
    os.makedirs(_SCRATCH, exist_ok=True)
    shutil.copyfile(os.path.expanduser(r"~\.claude\.credentials.json"),
                    os.path.join(_ISO, ".credentials.json"))
    with open(os.path.join(_ISO, "settings.json"), "w", encoding="utf-8") as f:
        f.write("{}")

# reuse the exact prompts/pools/judge from the main harness
_spec = importlib.util.spec_from_file_location("cr", str(ROOT / "saluca" / "creativity_ratio.py"))
cr = importlib.util.module_from_spec(_spec); sys.modules["cr"] = cr; _spec.loader.exec_module(cr)


def claude_gen(prompt, model, timeout=150):
    env = dict(os.environ, CLAUDE_CONFIG_DIR=_ISO)
    r = subprocess.run([CLAUDE, "-p", "--model", model], input=prompt, cwd=_SCRATCH,
                       env=env, capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku", help="claude -p --model (haiku, opus, ...)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--inject", type=int, default=10)
    ap.add_argument("--judge-model", default="gemma2:9b")
    ap.add_argument("--window", type=int, default=200000, help="Claude context window (for ratio)")
    args = ap.parse_args()
    _setup_isolation()

    judge = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1", timeout=120)
    out = ROOT / "data" / "raw_logs_saluca" / "creativity" / f"ideas_claude_cli_{args.model}_isolation.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")

    print(f"[claude-cli] model={args.model} judge={args.judge_model} seeds={args.seeds}")
    t0 = time.perf_counter(); n = 0
    for domain, ask in cr.DOMAINS.items():
        conds = [("plain", 0, dict(mode="plain")),
                 ("content", 1, dict(mode="content", avoid=cr.POOLS[domain][:args.inject])),
                 ("instruction", 2, dict(mode="instruction")),
                 ("functional", 3, dict(mode="functional", avoid=cr.POOLS[domain][:args.inject]))]
        for s in range(args.seeds):
            for label, code, kw in conds:
                user_p = cr.build_user(ask, **kw)
                try:
                    idea = claude_gen(user_p, args.model)
                    grounded = cr.judge_grounded(judge, args.judge_model, ask, idea, 12, 7) if idea else None
                except Exception as e:
                    print(f"  ERR {domain}/{label} s{s}: {type(e).__name__} {str(e)[:80]}"); continue
                # approx input tokens (claude -p text mode gives no usage): chars/4
                in_tok = len(user_p) // 4
                fh.write(json.dumps({"domain": domain, "load_type": "isolation", "condition": label,
                    "load_target": code, "seed": s, "input_tokens": in_tok,
                    "ratio": round(in_tok / args.window, 4), "grounded": grounded,
                    "idea": idea}, ensure_ascii=False) + "\n"); fh.flush()
                n += 1
        print(f"  {domain:9s} done ({args.seeds*4} calls, {(time.perf_counter()-t0)/60:.1f} min elapsed)")
    fh.close()
    print(f"[claude-cli] {n} ideas in {(time.perf_counter()-t0)/60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
