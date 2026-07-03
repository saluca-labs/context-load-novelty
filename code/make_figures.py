#!/usr/bin/env python3
"""Generate publication figures from the creativity-ratio data.

Fig1  cross-family isolation: plain vs content drift per model (effect replicates)
Fig2  confound isolation: plain/content/instruction/functional drift per model
Fig3  frontier dose-response: drift + groundedness vs load, gemma2 vs Fable (switch)
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "data" / "raw_logs_saluca" / "creativity"
FIG = ROOT / "saluca" / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ISO = {"gemma2:9b": CDIR / "ideas_ollama_gemma2-9b_isolation.jsonl",
       "haiku": CDIR / "ideas_claude_cli_haiku_isolation.jsonl",
       "opus": CDIR / "ideas_claude_cli_opus_isolation.jsonl",
       "Fable": CDIR / "ideas_claude_cli_claude-fable-5_isolation.jsonl"}
FRO = {"gemma2:9b (4K window)": CDIR / "frontier" / "ideas_frontier_gemma2-9b.jsonl",
       "Fable (200K window)": CDIR / "frontier" / "ideas_frontier_claude-fable-5.jsonl"}
CONDS = ["plain", "content", "instruction", "functional"]


def embed(texts):
    out = []
    for i in range(0, len(texts), 64):
        r = requests.post("http://localhost:11434/api/embed",
                          json={"model": "nomic-embed-text", "input": texts[i:i+64]}, timeout=180)
        r.raise_for_status(); out += r.json()["embeddings"]
    return np.array(out, dtype=float)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load(path, key):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("idea")]
    for r, e in zip(rows, embed([r["idea"] for r in rows])):
        r["_e"] = e
    # drift baseline per domain: isolation -> 'plain'; frontier -> min load
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
    for dom, rs in by_dom.items():
        if key == "iso":
            base = [x["_e"] for x in rs if x["condition"] == "plain"]
        else:
            bl = min(x["load_target"] for x in rs)
            base = [x["_e"] for x in rs if x["load_target"] == bl]
        c = np.mean(base, axis=0)
        for x in rs:
            x["_drift"] = 1 - cos(x["_e"], c)
    return rows


# ---- Fig 1 & 2: isolation ----
iso = {m: load(p, "iso") for m, p in ISO.items()}
models = list(ISO)


def cond_stat(rows, cond, field="_drift"):
    v = [r[field] for r in rows if r["condition"] == cond and r.get(field) is not None]
    return float(np.mean(v)) if v else np.nan


# Fig1: plain vs content
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(models)); w = 0.36
plain = [cond_stat(iso[m], "plain") for m in models]
content = [cond_stat(iso[m], "content") for m in models]
ax.bar(x - w/2, plain, w, label="plain (no context)", color="#b0b7c3")
ax.bar(x + w/2, content, w, label="+ relevant context (no instruction)", color="#2f7ec4")
for i, (p, c) in enumerate(zip(plain, content)):
    ax.annotate(f"+{100*(c-p)/p:.0f}%", (i + w/2, c), ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("off-mainstream drift")
ax.set_title("Relevant context-load moves ideation off-mainstream (all families)")
ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig1_cross_family.png", dpi=160); plt.close(fig)

# Fig2: 4 conditions per model
fig, ax = plt.subplots(figsize=(8, 4.4))
w = 0.2
colors = {"plain": "#b0b7c3", "content": "#2f7ec4", "instruction": "#e0a23a", "functional": "#4a9e6f"}
for j, cond in enumerate(CONDS):
    vals = [cond_stat(iso[m], cond) for m in models]
    ax.bar(x + (j - 1.5) * w, vals, w, label=cond, color=colors[cond])
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("off-mainstream drift")
ax.set_title("Isolation: content-alone (no instruction) drives the effect")
ax.legend(frameon=False, fontsize=9, ncol=4); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig2_isolation.png", dpi=160); plt.close(fig)

# ---- Fig 3: frontier (groundedness panel shows BOTH judges: weak vs strong) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
mcolor = {}
for i, (m, p) in enumerate(FRO.items()):
    rows = load(p, "fro")
    col = plt.cm.tab10(i)
    mcolor[m] = col
    loads = sorted(set(r["load_target"] for r in rows))
    drift = [np.mean([r["_drift"] for r in rows if r["load_target"] == L]) for L in loads]
    g_weak = [np.mean([r["grounded"] for r in rows if r["load_target"] == L and r.get("grounded") is not None]) for L in loads]
    g_strong = [np.mean([r["grounded_strong"] for r in rows if r["load_target"] == L and r.get("grounded_strong") is not None]) for L in loads]
    g_human = [np.mean([r["grounded_human"] for r in rows if r["load_target"] == L and r.get("grounded_human") is not None]) for L in loads]
    a1.plot(loads, drift, "-o", color=col, label=m)
    a2.plot(loads, g_strong, "-o", color=col, alpha=0.8, label=f"{m} — opus judge")
    a2.plot(loads, g_weak, "--", color=col, alpha=0.4, label=f"{m} — gemma2 judge")
    if any(x == x for x in g_human):  # human rated Fable frontier only
        a2.plot(loads, g_human, "-s", color="black", lw=2, label=f"{m} — HUMAN")
a1.set_xlabel("relevant context injected (tokens)"); a1.set_ylabel("off-mainstream drift")
a1.set_title("Frontier: novelty saturates (a switch, not a dial)")
a1.axvspan(0, 800, color="#2f7ec4", alpha=0.06); a1.legend(frameon=False, fontsize=8)
a1.spines[["top", "right"]].set_visible(False)
a2.set_xlabel("relevant context injected (tokens)"); a2.set_ylabel("groundedness")
a2.set_ylim(0, 1); a2.set_title("Groundedness vs load: HUMAN is flat; LLM judges disagree & run lenient")
a2.legend(frameon=False, fontsize=7); a2.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig3_frontier.png", dpi=160); plt.close(fig)

print("figures written to", FIG)
for f in sorted(FIG.glob("*.png")):
    print(" ", f.name)
