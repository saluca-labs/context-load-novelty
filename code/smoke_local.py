#!/usr/bin/env python3
"""Saluca local smoke — ONE episode of stateful_puzzle on a local Ollama model.

Proves the end-to-end loop (env -> 3-call agent -> gold-state scoring -> JSONL)
runs against a local model via Ollama's OpenAI-compatible endpoint, at $0.

Run (PowerShell):
    $env:OPENAI_BASE_URL = "http://localhost:11434/v1"
    $env:OPENAI_API_KEY  = "ollama"
    .\.venv\Scripts\python.exe saluca\smoke_local.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.llm_client import LLMClient  # noqa: E402
from src.runner import CellSpec, CostTracker, EpisodeOutcome, run_pilot_slice  # noqa: E402

MODEL = "local:gemma2:9b"  # `local:` prefix routes to the Ollama endpoint


def main() -> int:
    log_dir = ROOT / "data" / "raw_logs_saluca"
    log_dir.mkdir(parents=True, exist_ok=True)

    backdrop = {
        "T": 40, "state_card": 10, "branching": 4,
        "obs_noise": "clean", "mut_rate": "static", "dep_density": 4,
    }
    cell = CellSpec(
        env_name="stateful_puzzle",
        model=MODEL,
        stress_config=backdrop,
        task_config={"archetype": "saluca_smoke", "stress_config": backdrop},
        task_seed=900000,
        decoding_seed=42,
        world_regime="saluca_local_smoke",
        task_id="saluca_smoke_00",
        memory_mode="C_struct",
    )

    client = LLMClient()
    ct = CostTracker(
        out_path=log_dir / "cost_tracker.jsonl",
        phase="saluca_smoke",
        slice_name="saluca_smoke_local",
        emit_every=1,
    )

    def progress(i, n, o: EpisodeOutcome):
        s = "OK" if o.error is None else f"ERR({o.error[:120]})"
        success = "[success]" if o.success else "[fail]"
        print(f"[smoke {i}/{n}] {o.cell.task_id} {success} {s} "
              f"steps={o.steps} cost=${o.cost_usd:.4f}")

    print(f"[smoke] model={MODEL} — one episode, state_card=10, T=40")
    t0 = time.perf_counter()
    outcomes = run_pilot_slice(
        cells=[cell],
        client=client,
        step_jsonl_path=log_dir / "saluca_smoke_step.jsonl",
        episode_jsonl_path=log_dir / "saluca_smoke_episode.jsonl",
        cost_tracker=ct,
        n_workers=1,
        progress_fn=progress,
    )
    elapsed = time.perf_counter() - t0
    o = outcomes[0]
    print("\n[smoke] === RESULT ===")
    print(f"  success={o.success} steps={o.steps} error={o.error}")
    print(f"  wall_clock={elapsed:.1f}s")
    print(f"  step log: {log_dir / 'saluca_smoke_step.jsonl'}")
    return 0 if o.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
