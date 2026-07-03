#!/usr/bin/env python3
"""Analyze the confound sweep: does collapse track token-length or entity-count?

Reads step logs under data/raw_logs_saluca/confound/step_*.jsonl, groups by the
verbosity condition encoded in the filename, and reports per condition:
  - mean input tokens / step  (the manipulation check: verbose >> compact)
  - mean world_state_accuracy (fidelity)
  - mean action_valid
  - collapse onset: first step where a 5-step rolling world_state_accuracy < 0.5
    (the paper's tau_W definition), averaged over episodes that collapse

Verdict logic:
  If verbose collapses EARLIER (smaller tau_W) / LOWER fidelity than compact at
  the same state_card, that supports the token-length / attention-load reading
  (the paper's unaddressed confound). If they're indistinguishable, that supports
  the paper's genuine-capacity reading.

Usage:
    .\.venv\Scripts\python.exe saluca\analyze_confound.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data" / "raw_logs_saluca" / "confound"


def rolling_below(vals, window=5, thresh=0.5):
    """First index where the trailing `window`-mean drops below thresh (tau_W)."""
    for i in range(len(vals)):
        w = vals[max(0, i - window + 1): i + 1]
        if len(w) >= window and sum(w) / len(w) < thresh:
            return i
    return None


def main() -> int:
    steps = sorted(LOG_DIR.glob("step_*.jsonl"))
    if not steps:
        print(f"No step logs in {LOG_DIR}. Run run_confound.py first.")
        return 1

    # group rows by (condition, episode)
    by_cond_ep = defaultdict(lambda: defaultdict(list))
    for f in steps:
        # filename: step_sc{K}_dd{D}_{verbosity}.jsonl
        cond = f.stem.replace("step_", "")
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            # group by run_id: unique per episode, robust to repeated task_ids
            # and to multiple runs appended to the same log file.
            by_cond_ep[cond][r.get("run_id", r.get("task_id", "?"))].append(r)

    print(f"{'condition':28s} {'eps':>4s} {'in_tok/step':>11s} "
          f"{'wsa':>6s} {'valid':>6s} {'tauW':>6s} {'collapse%':>9s}")
    print("-" * 78)
    for cond in sorted(by_cond_ep):
        eps = by_cond_ep[cond]
        in_toks, wsas, valids, tauws, n_collapse = [], [], [], [], 0
        for ep_rows in eps.values():
            ep_rows.sort(key=lambda r: r.get("step", 0))
            wsa_seq = [float(r.get("world_state_accuracy", 0) or 0) for r in ep_rows]
            in_toks += [r.get("input_tokens_this_step", 0) for r in ep_rows]
            wsas += wsa_seq
            valids += [1 if r.get("action_valid") else 0 for r in ep_rows]
            tw = rolling_below(wsa_seq)
            if tw is not None:
                tauws.append(tw)
                n_collapse += 1
        n = len(eps)
        print(f"{cond:28s} {n:4d} {st.mean(in_toks):11.0f} "
              f"{st.mean(wsas):6.3f} {st.mean(valids):6.3f} "
              f"{(st.mean(tauws) if tauws else float('nan')):6.1f} "
              f"{100*n_collapse/max(n,1):8.0f}%")

    print("\nRead: compare same-state_card rows across verbosity. If `verbose` shows")
    print("lower wsa / earlier tauW than `compact`, collapse tracks TOKEN-LENGTH")
    print("(context/attention-load) — the paper's unpatched confound. If equal,")
    print("it tracks ENTITY-COUNT (the paper's capacity reading).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
