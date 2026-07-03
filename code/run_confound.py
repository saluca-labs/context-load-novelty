#!/usr/bin/env python3
"""Saluca confound sweep — token-length vs. entity-count as the driver of collapse.

Design (Cristian's hypothesis):
  Hold entity-count FIXED (state_card constant) and vary only the token-length of
  the serialized state via SALUCA_STATE_VERBOSITY (compact | indent | verbose).
  Run N seeds per condition.

Read-out:
  If world-model collapse tracks state-text TOKEN-LENGTH -> the `verbose` condition
  degrades world_state_accuracy faster / lower than `compact` at the SAME state_card
  (supports "it's really a context/attention-load effect", the paper's unpatched
  confound). If collapse tracks ENTITY-COUNT -> the conditions are indistinguishable
  (supports the paper's genuine-capacity reading).

IMPORTANT: set SALUCA_STATE_VERBOSITY in the environment BEFORE launching each
condition (prompts._json_short reads it at import time). This script runs ONE
condition per invocation; loop conditions from PowerShell (see README_saluca.md).

Run (PowerShell), one condition at a time:
    $env:OPENAI_BASE_URL="http://localhost:11434/v1"; $env:OPENAI_API_KEY="ollama"
    $env:SALUCA_STATE_VERBOSITY="compact"; .\.venv\Scripts\python.exe saluca\run_confound.py --seeds 5 --state-card 10
    $env:SALUCA_STATE_VERBOSITY="verbose"; .\.venv\Scripts\python.exe saluca\run_confound.py --seeds 5 --state-card 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.llm_client import LLMClient  # noqa: E402
from src.runner import CellSpec, CostTracker, EpisodeOutcome, run_pilot_slice  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="local:gemma2:9b")
    ap.add_argument("--seeds", type=int, default=5, help="episodes per condition")
    ap.add_argument("--state-card", type=int, default=10, help="HELD FIXED across conditions")
    ap.add_argument("--dep-density", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cloud", action="store_true",
                    help="reasoning/cloud model: raise per-call token caps so hidden "
                         "thinking doesn't starve the visible JSON (else empty-200 parse fails)")
    args = ap.parse_args()

    # Cloud/reasoning models need bigger token budgets (glm-*:cloud, gpt-oss, qwen3.5...).
    # _default_mt() in llm_client reads these at call time, so setting them here works.
    if args.cloud or ":cloud" in args.model:
        os.environ.setdefault("SALUCA_MT_PLANNER", "2000")
        os.environ.setdefault("SALUCA_MT_UPDATER", "3000")
        os.environ.setdefault("SALUCA_MT_SELFDIAG", "1500")

    verbosity = os.environ.get("SALUCA_STATE_VERBOSITY", "compact")
    # model slug in the tag so cloud/other-model runs don't collide with local gemma
    mslug = args.model.replace("local:", "").replace(":", "-").replace("/", "-")
    tag = f"{mslug}_sc{args.state_card}_dd{args.dep_density}_{verbosity}"
    log_dir = ROOT / "data" / "raw_logs_saluca" / "confound"
    log_dir.mkdir(parents=True, exist_ok=True)

    backdrop = {
        "T": args.horizon, "state_card": args.state_card, "branching": 4,
        "obs_noise": "clean", "mut_rate": "static", "dep_density": args.dep_density,
    }
    cells = []
    for i in range(args.seeds):
        cells.append(CellSpec(
            env_name="stateful_puzzle",
            model=args.model,
            stress_config=backdrop,
            task_config={"archetype": "saluca_confound", "stress_config": backdrop},
            task_seed=910000 + args.state_card * 1000 + i,
            decoding_seed=42,
            world_regime=f"saluca_confound_{verbosity}",
            task_id=f"confound_{tag}_s{i:02d}",
            memory_mode="C_struct",
        ))

    client = LLMClient()
    ct = CostTracker(out_path=log_dir / f"cost_{tag}.jsonl",
                     phase="saluca_confound", slice_name=tag, emit_every=1)

    def progress(i, n, o: EpisodeOutcome):
        state = "OK" if o.error is None else f"ERR({o.error[:100]})"
        print(f"[{tag} {i}/{n}] {o.cell.task_id} "
              f"{'solved' if o.success else 'unsolved'} {state} steps={o.steps}")

    print(f"[confound] verbosity={verbosity} state_card={args.state_card} "
          f"dd={args.dep_density} T={args.horizon} seeds={args.seeds} model={args.model}")
    t0 = time.perf_counter()
    outcomes = run_pilot_slice(
        cells=cells, client=client,
        step_jsonl_path=log_dir / f"step_{tag}.jsonl",
        episode_jsonl_path=log_dir / f"episode_{tag}.jsonl",
        cost_tracker=ct, n_workers=args.workers, progress_fn=progress,
    )
    elapsed = time.perf_counter() - t0
    n_succ = sum(1 for o in outcomes if o.success)
    n_err = sum(1 for o in outcomes if o.error is not None)
    print(f"\n[confound] {tag}: {n_succ}/{len(outcomes)} solved, "
          f"{n_err} errors, {elapsed/60:.1f} min")
    print(f"[confound] step log: {log_dir / f'step_{tag}.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
