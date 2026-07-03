#!/usr/bin/env python3
"""Feasibility smoke for a cloud model with a LARGE relevant-reference prompt.
Tests configs to find one that returns a real idea cheaply (reasoning models
empty-out at low max_tokens and burn tokens on hidden thinking).

Usage: python saluca\\_smoke_cloud.py [model] [domain]
"""
import json, sys, time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:cloud"
domain = sys.argv[2] if len(sys.argv) > 2 else "science"

refs = json.load(open(ROOT / "saluca" / "references.json", encoding="utf-8"))
ref = refs[domain]
base_user = ("REFERENCE MATERIAL (context only):\n" + ref +
             "\n\nTASK: Propose ONE original, specific, plausible idea for a novel testable "
             "hypothesis about why biological sleep is necessary. One or two sentences. Output only the idea.")
SYS = "You generate ONE original, specific, plausible idea. Output only the idea."
c = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1", timeout=300)


def run(label, **kw):
    user = kw.pop("user", base_user)
    t = time.time()
    try:
        r = c.chat.completions.create(model=model,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}], **kw)
        dt = time.time() - t
        txt = (r.choices[0].message.content or "").strip()
        print(f"[{label}] {dt:.1f}s  ptok={r.usage.prompt_tokens} ctok={r.usage.completion_tokens} "
              f"idea_len={len(txt)}")
        print(f"   -> {txt[:220]}")
    except Exception as e:
        print(f"[{label}] ERR {type(e).__name__}: {str(e)[:160]}")


print(f"model={model} ref~{len(ref)//4} tok")
# A: disable thinking via Ollama extra_body -> cheap, clean, no reasoning confound
run("think=false mt=400", temperature=0.7, max_tokens=400, extra_body={"think": False})
# B: Qwen /no_think convention in the prompt
run("/no_think mt=400", temperature=0.7, max_tokens=400, user=base_user + " /no_think")
# C: brute force high tokens (reasoning on) -- fallback
run("reasoning mt=8000", temperature=0.7, max_tokens=8000)
