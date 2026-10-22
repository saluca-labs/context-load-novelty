# AGENTS.md

Instructions for AI coding agents working in this repository.

## Setup

```bash
pip install numpy requests matplotlib openai
ollama pull nomic-embed-text
# the scripts expect a parent-checkout layout; stage it (these paths are not tracked)
mkdir -p data/raw_logs_saluca/creativity/frontier saluca/paper/figures
cp data/ideas_*_isolation.jsonl data/raw_logs_saluca/creativity/
cp data/ideas_frontier_*.jsonl data/raw_logs_saluca/creativity/frontier/
```

## Test

There is no test suite. The check that the analysis still reproduces the paper:

```bash
python code/analyze_isolation.py
python code/analyze_frontier.py
python code/make_figures.py
```

Passing looks like the isolation tables for the four models (gemma2:9b content drift 0.164
against plain 0.110), the frontier tables for gemma2-9b and claude-fable-5, and three figures in
`saluca/paper/figures/` that match `figures/`.

## Conventions

- Python 3, numpy, requests, matplotlib, openai. Do not add dependencies without a reason.
- No em-dashes in docs.
- If a doc changes, regenerate `llms-full.txt` in the same change or delete it.

## Do not

- Edit `data/`, `figures/`, `human_rating/rate_me.csv`, `paper.md` or `paper.pdf`: they are the
  released record behind a published Zenodo preprint.
- Commit the staged `data/raw_logs_saluca/` or `saluca/` directories.
- Run `deposit_zenodo.py`. It is the author's deposit tooling; deposits are made by the author.
- Commit secrets, hostnames, internal paths or personal emails.

## Done means

The three Test commands run from a staged clean clone and the README still matches what the code
on `main` does.
