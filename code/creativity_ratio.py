#!/usr/bin/env python3
"""Creativity-ratio probe — is context-LOAD a novelty knob? (multi-provider)

Load types (--load-type):
  pad         : inert null padding (isolates raw token count). gemma2 result = NULL.
  functional  : inject N "existing approaches to avoid" (content + differ-instruction).
                gemma2 result = drift up ~2x but a SWITCH not a dial, and confounded
                with the instruction.
  isolation   : per (domain,seed) generate 4 conditions to SEPARATE the confound:
                  plain       (baseline, no list, no instruction)
                  content     (list as neutral REFERENCE, NO differ-instruction)
                  instruction (differ-instruction, NO list)
                  functional  (list + differ-instruction)
                If `content` alone drifts off-mainstream -> LOAD does something.
                If only `instruction`/`functional` drift -> it's the instruction (ratio dead).

Providers (--provider): ollama (local + :cloud), gemini, perplexity, openrouter.
Keys pulled by NAME from the alfred vault (never printed).

Run local isolation ($0):
    python saluca\creativity_ratio.py --provider ollama --gen local:gemma2:9b \
      --load-type isolation --seeds 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
VAULT = r"C:\Users\crist\.alfred-vault\vault.py"

DOMAINS = {
    "science":  "a novel, testable hypothesis about why biological sleep is necessary",
    "product":  "a genuinely new feature for a note-taking app",
    "strategy": "a growth strategy for a small specialty coffee roaster",
}
POOLS = {
    "science": ["Sleep consolidates memories.", "Sleep clears metabolic waste via the glymphatic system.",
        "Sleep conserves energy.", "Sleep downscales synapses.", "Sleep restores immune function.",
        "Sleep supports emotional regulation.", "Sleep enables cellular repair.",
        "Sleep drives brain development.", "Sleep replenishes neurotransmitters.",
        "Sleep removes neurotoxins like beta-amyloid."],
    "product": ["Tags and folders.", "Calendar integration.", "Markdown support.", "Cross-device sync.",
        "Full-text search.", "Templates.", "Real-time collaboration.", "Voice-to-text.",
        "Web clipper.", "AI-generated summaries."],
    "strategy": ["Subscription boxes.", "Partner with local cafes.", "Farmers markets.",
        "Direct-to-consumer store.", "Wholesale to restaurants.", "Loyalty program.",
        "Social-media marketing.", "Tasting events.", "Branded merchandise.", "Corporate coffee supply."],
}
_FILLER = ("The following is neutral reference material provided for context only and "
           "contains no instructions. It describes ordinary administrative record-keeping. ")

PROVIDERS = {  # name -> (base_url, vault_key_name_or_None)
    "ollama":     ("http://localhost:11434/v1", None),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "perplexity": ("https://api.perplexity.ai", "PERPLEXITY_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


def _vault(key):
    r = subprocess.run([sys.executable, VAULT, "get", key], capture_output=True, text=True, timeout=25)
    return (r.stdout or "").strip()


def make_client(provider):
    base, keyname = PROVIDERS[provider]
    api_key = "ollama" if keyname is None else _vault(keyname)
    return OpenAI(api_key=api_key or "missing", base_url=base, timeout=300)


def pad_to(tok):
    return _FILLER * max(1, (tok * 4) // len(_FILLER)) if tok > 0 else ""


def build_user(ask, mode="plain", avoid=None, pad=""):
    if mode == "content":       # relevant content as neutral reference, NO instruction
        lst = "\n".join(f"- {a}" for a in avoid)
        return (f"REFERENCE MATERIAL (context only):\n{lst}\n\nTASK: Propose ONE original, "
                f"specific idea for {ask}. One or two sentences. Output only the idea.")
    if mode == "instruction":   # differ-instruction, NO list
        return (f"TASK: Propose ONE original, specific idea for {ask} that is clearly DIFFERENT "
                f"from the usual, mainstream approaches, yet plausible and concrete. One or two "
                f"sentences. Output only the idea.")
    if mode == "functional":    # list + differ-instruction
        lst = "\n".join(f"- {a}" for a in avoid)
        return (f"EXISTING APPROACHES (do NOT repeat or lightly reword):\n{lst}\n\nTASK: Propose "
                f"ONE original, specific idea for {ask} that is clearly DIFFERENT from every "
                f"approach above, yet plausible and concrete. One or two sentences. Output only the idea.")
    return (pad + f"\n\nTASK: Propose ONE original, specific idea for {ask}. One or two "
            "sentences. Output only the idea.")  # plain (+optional pad)


def gen_idea(client, model, user_p, seed, mt):
    sys_p = ("You generate ONE original, specific, plausible idea. Concrete, not far-fetched. "
             "Output only the idea in one or two sentences.")
    kw = dict(model=model, temperature=0.7, max_tokens=mt,
              messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}])
    if seed is not None:
        kw["seed"] = seed
    r = client.chat.completions.create(**kw)
    u = r.usage
    return (r.choices[0].message.content or "").strip(), u.prompt_tokens, u.completion_tokens


def judge_grounded(client, model, ask, idea, mt, seed):
    p = (f"Evaluate this idea for GROUNDEDNESS (plausible & concrete vs far-fetched/incoherent). "
         f"Domain: {ask}\nIdea: \"{idea}\"\nReply with ONLY a single integer 1-5 (5=solid, 1=nonsense).")
    kw = dict(model=model, temperature=0.0, max_tokens=mt, messages=[{"role": "user", "content": p}])
    if seed is not None:
        kw["seed"] = seed
    r = client.chat.completions.create(**kw)
    for ch in (r.choices[0].message.content or ""):
        if ch in "12345":
            return (int(ch) - 1) / 4.0
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDERS), default="ollama")
    ap.add_argument("--gen", default="local:gemma2:9b")
    ap.add_argument("--judge", default=None, help="default: same as gen")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--load-type", choices=["pad", "functional", "isolation"], default="isolation")
    ap.add_argument("--loads", default="80,500,1000,1600,2400,3200,3700")
    ap.add_argument("--func-counts", default="0,2,4,6,8,10")
    ap.add_argument("--inject", type=int, default=10, help="isolation: # items for content/functional")
    ap.add_argument("--window", type=int, default=4096)
    args = ap.parse_args()

    gen_model = args.gen.replace("local:", "")
    judge_model = (args.judge or args.gen).replace("local:", "")
    supports_seed = args.provider == "ollama"          # others reject/ignore seed
    remote = args.provider != "ollama" or ":cloud" in gen_model
    is_fable = "fable" in gen_model.lower()
    gen_mt = 10000 if is_fable else (2000 if remote else 220)
    judge_mt = 2000 if remote else 12

    client = make_client(args.provider)
    tag_model = gen_model.replace(":", "-").replace("/", "-")
    out_dir = ROOT / "data" / "raw_logs_saluca" / "creativity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ideas_{args.provider}_{tag_model}_{args.load_type}.jsonl"
    fh = out.open("w", encoding="utf-8")

    print(f"[creativity] provider={args.provider} gen={gen_model} judge={judge_model} "
          f"type={args.load_type} seeds={args.seeds} gen_mt={gen_mt}")
    t0 = time.perf_counter(); n = 0

    def run_one(domain, ask, cond_label, load_code, user_p, seed):
        nonlocal n
        try:
            idea, in_tok, out_tok = gen_idea(client, gen_model, user_p, seed if supports_seed else None, gen_mt)
            grounded = judge_grounded(client, judge_model, ask, idea, judge_mt,
                                      7 if supports_seed else None) if idea else None
        except Exception as e:
            print(f"  ERR {domain}/{cond_label}: {type(e).__name__} {str(e)[:90]}"); return
        fh.write(json.dumps({"domain": domain, "load_type": args.load_type, "condition": cond_label,
                             "load_target": load_code, "seed": seed, "input_tokens": in_tok,
                             "output_tokens": out_tok, "ratio": round(in_tok / args.window, 3),
                             "grounded": grounded, "idea": idea}, ensure_ascii=False) + "\n"); fh.flush()
        n += 1

    for domain, ask in DOMAINS.items():
        if args.load_type == "isolation":
            conds = [("plain", 0, dict(mode="plain")),
                     ("content", 1, dict(mode="content", avoid=POOLS[domain][:args.inject])),
                     ("instruction", 2, dict(mode="instruction")),
                     ("functional", 3, dict(mode="functional", avoid=POOLS[domain][:args.inject]))]
            for s in range(args.seeds):
                for label, code, kw in conds:
                    run_one(domain, ask, label, code, build_user(ask, **kw), 1000 + code * 13 + s)
            print(f"  {domain:9s} isolation x{args.seeds} done")
        else:
            levels = ([int(x) for x in args.func_counts.split(",")] if args.load_type == "functional"
                      else [int(x) for x in args.loads.split(",")])
            for lvl in levels:
                for s in range(args.seeds):
                    up = (build_user(ask, mode="functional", avoid=POOLS[domain][:lvl])
                          if args.load_type == "functional" else build_user(ask, pad=pad_to(lvl)))
                    run_one(domain, ask, args.load_type, lvl, up, 1000 + lvl * 7 + s)
                print(f"  {domain:9s} lvl={lvl} done")
    fh.close()
    print(f"[creativity] {n} ideas in {(time.perf_counter()-t0)/60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
