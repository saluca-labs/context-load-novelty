# Relevant Context-Load as a Saturating Novelty Knob — paper package

**Published:** Zenodo DOI [10.5281/zenodo.21149157](https://doi.org/10.5281/zenodo.21149157) (saluca community, CC-BY-4.0). Preprint, not peer-reviewed.

**Cite:** Ruvalcaba, C. and the Saluca Agentic AI Research Team (2026). *Relevant Context-Load as a Saturating Novelty Knob in Language Models: A Cross-Family Study.* Zenodo. https://doi.org/10.5281/zenodo.21149157

## Contents
- `paper.pdf` / `paper.md` — the preprint (figures embedded).
- `figures/` — fig1 cross-family, fig2 isolation, fig3 frontier dose-response (PNG, 160 dpi).
- `code/` — all generation, judging, analysis, and figure scripts + `references.json` (reference corpora).
- `data/` — raw per-idea JSONL logs (isolation ×4 models, frontier ×2 models, plus the earlier pad/functional sweeps).
- `.zenodo.json` — Zenodo deposition metadata.

## The finding, in one paragraph
Relevant context injected as neutral reference (no "be different" instruction) reliably moves LLM ideation off the mainstream answer — replicated on gemma2:9b, haiku, opus, and Fable (+49% to +147%), groundedness preserved. An isolation design shows it's the *load*, not an implicit instruction. But it is a **saturating switch, not a dial**: novelty saturates within a few hundred tokens and plateaus — there is **no optimal input/window ratio**. The threshold is absolute (~a few hundred tokens) and scale-invariant; the frontier model saturates *sooner* (~300 vs ~800 tokens).

## Before submitting (open items)
- [ ] **Confirm author name** in `paper.md` and `.zenodo.json` (`[Cristian — Saluca Labs]` placeholder).
- [ ] Confirm **license** (currently CC-BY-4.0) and the AI-assistance acknowledgement wording.
- [ ] Optional strengthening (not blocking): more seeds; a stronger/human groundedness judge; a non-reasoning cloud family; probe >54% window on a small-window model.
- [ ] Decide standalone vs. pairing with the world-model-collapse confound result (referenced in the intro).

## Reproduce
Requires local Ollama (`gemma2:9b`, `nomic-embed-text`) and, for the Claude models, the vendor CLI in an isolated config (see `code/run_claude_cli.py` / `run_frontier_claude.py`). Regenerate figures from the raw logs with `code/make_figures.py`.
