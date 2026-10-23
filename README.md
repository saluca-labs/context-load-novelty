# Relevant Context-Load as a Saturating Novelty Knob: paper package

Harness, raw per-idea logs, figures and preprint for a cross-family study of whether the amount of
relevant context in a language model's window changes the novelty of its ideas, for readers who
want to check or extend the result.

**Published:** Zenodo, concept DOI [10.5281/zenodo.21149156](https://doi.org/10.5281/zenodo.21149156) (saluca community, CC-BY-4.0). Preprint, not peer-reviewed.

## How it works

Re-analysing the released logs, which is what the Quickstart does. Every box is a file or script on
this branch; every arrow is a read, a write or a copy that the code performs. The staged paths are
the ones hard-coded in the scripts (`ROOT / "data" / "raw_logs_saluca" / "creativity"` and
`ROOT / "saluca" / "paper" / "figures"`).

```mermaid
flowchart LR
    REL["data/ (released logs): 4 isolation files, 2 frontier files"] -->|"cp, Quickstart step 1"| ISO["data/raw_logs_saluca/creativity/ (isolation logs)"]
    REL -->|"cp, Quickstart step 1"| FRO["data/raw_logs_saluca/creativity/frontier/ (frontier logs)"]
    ISO --> AI["code/analyze_isolation.py"]
    FRO --> AF["code/analyze_frontier.py"]
    ISO --> MF["code/make_figures.py"]
    FRO --> MF
    EMB["Ollama /api/embed, nomic-embed-text"] -->|"one vector per idea"| AI
    EMB --> AF
    EMB --> MF
    AI --> T1["stdout: grounded, drift, diverge per condition"]
    AF --> T2["stdout: grounded, drift, diverge per load"]
    MF --> FIG["saluca/paper/figures/fig1_cross_family.png, fig2_isolation.png, fig3_frontier.png"]
```

Novelty (drift) is computed at analysis time from the embeddings, so it needs Ollama. Groundedness
is read from fields already in the released logs: `grounded` (gemma2 judge), `grounded_strong`
(opus judge, added by `re_judge.py`) and, for the Fable frontier rows only, `grounded_human`
(added by `merge_human_rating.py`). How those logs were produced is in
[Architecture](#architecture).

## Status

Research artifact, published 2026-07-03 as version 1 of the Zenodo record. The released logs in
`data/` and the analysis and figure scripts in `code/` reproduce the paper's isolation and
frontier tables and its three figures (checked 2026-10-22, see Quickstart). The scripts were
written for a parent checkout with a different layout, so they do **not** run from this clone
as released; the Quickstart stages the files into the layout they expect. Generation needs a
local Ollama for gemma2:9b and the Claude CLI for the Claude models.

> **Correction, 2026-10-22.** This README said "Regenerate figures from the raw logs with
> `code/make_figures.py`". From a clean clone that command fails with `FileNotFoundError`: the
> scripts read logs from `data/raw_logs_saluca/creativity/` (and `.../frontier/`), write figures
> to `saluca/paper/figures/`, and the generators load `saluca/references.json` and
> `saluca/creativity_ratio.py`, none of which exist here. The Quickstart below gives the staging
> steps that make the released scripts run unchanged. This README also listed pre-submission open
> items (an author-name placeholder, confirming the licence). Those were stale: the preprint was
> published on Zenodo on 2026-07-03 with the author named, and an Apache-2.0 `LICENSE` was added
> on 2026-08-18. The remaining strengthening items are kept below as future work.

### The finding, in one paragraph

Relevant context injected as neutral reference (no "be different" instruction) reliably moves LLM
ideation off the mainstream answer, replicated on gemma2:9b, haiku, opus, and Fable (+49% to
+147%), groundedness preserved. An isolation design shows it's the *load*, not an implicit
instruction. But it is a **saturating switch, not a dial**: novelty saturates within a few
hundred tokens and plateaus, and there is **no optimal input/window ratio**. The threshold is
absolute (~a few hundred tokens) and scale-invariant; the frontier model saturates *sooner* (~300
vs ~800 tokens).

### Not done (future work)

- More seeds; a stronger or human groundedness judge beyond the single blinded rater; a
  non-reasoning cloud family; probing beyond 54% of the window on a small-window model.
- Expert-graded value: the paper's metrics give the load-response shape, not domain-expert
  judgment of the ideas.

## Install

Tested 2026-10-22 with Python 3.12.10, numpy 2.4.6, requests 2.33.0, matplotlib 3.10.9,
openai 2.16.0 and Ollama 0.34.4. There is no `requirements.txt`.

```bash
git clone https://github.com/saluca-labs/context-load-novelty
cd context-load-novelty
pip install numpy requests matplotlib openai
ollama pull nomic-embed-text   # embeddings for novelty (drift); needed by the analysis scripts
ollama pull gemma2:9b          # only to generate new ideas
```

## Quickstart

Re-analyse the released logs and regenerate the figures. The analysis embeds every idea through
the local Ollama (`http://localhost:11434/api/embed`, `nomic-embed-text`), so Ollama must be
running; `make_figures.py` took about 45 seconds.

```bash
# stage the released logs where the scripts expect them
mkdir -p data/raw_logs_saluca/creativity/frontier saluca/paper/figures
cp data/ideas_*_isolation.jsonl data/raw_logs_saluca/creativity/
cp data/ideas_frontier_*.jsonl data/raw_logs_saluca/creativity/frontier/

python code/analyze_isolation.py
python code/analyze_frontier.py
python code/make_figures.py      # writes saluca/paper/figures/fig1..fig3
```

Part of the `analyze_isolation.py` output from that run (the verdict line after each table is
omitted here):

```
=== ollama_gemma2-9b_isolation ===
condition      n  grounded   drift  diverge  in_tok
--------------------------------------------------
plain         30     0.900   0.110    0.450      70
content       30     0.892   0.164    0.446     142
instruction   30     0.850   0.146    0.470      83
functional    30     0.867   0.164    0.441     161
```

and of `analyze_frontier.py`:

```
=== gemma2-9b ===
  load  ratio   n  grounded   drift  diverge
----------------------------------------------
     0   0.02  18     0.847   0.087    0.437
   300   0.07  18     0.889   0.121    0.421
   800   0.15  18     0.833   0.148    0.449
  1600   0.28  18     0.792   0.144    0.437
  2400   0.40  18     0.833   0.150    0.460
  3200   0.54  18     0.819   0.140    0.459
```

The regenerated `fig1_cross_family.png` and `fig3_frontier.png` were compared by eye with the
committed `figures/` and match (same values and labels: +49%, +96%, +147%, +109%).

## Reproduce each result

Each headline result, the script that re-derives it from the released logs (after the Quickstart
staging), and the file it reads. The paper is archived under the concept DOI
[10.5281/zenodo.21149156](https://doi.org/10.5281/zenodo.21149156).

| Result in the paper | Command | Reads |
|---|---|---|
| Isolation tables: load moves drift, an implicit instruction does not explain it | `python code/analyze_isolation.py` | the four `data/ideas_*_isolation.jsonl` |
| Frontier tables: drift saturates within a few hundred tokens | `python code/analyze_frontier.py` | `data/ideas_frontier_gemma2-9b.jsonl`, `data/ideas_frontier_claude-fable-5.jsonl` |
| Figures 1 to 3, including the +49% to +147% cross-family gains | `python code/make_figures.py` | the four isolation and two frontier logs |
| gemma2 judge against opus judge | `python code/compare_judges.py` | the same six logs |
| Human rater against both judges (r = -0.12 with opus, 0.00 with gemma2; means 0.45, 0.85, 0.77) | correlate `grounded_human` with `grounded_strong` and `grounded` (`merge_human_rating.py` prints this, but needs the unreleased `_key.json`) | `data/ideas_frontier_claude-fable-5.jsonl`, 54 rated rows |

The last row was recomputed directly from the released log fields on 2026-10-22 and matches the
paper.

## Usage

The generators (each logs the JSONL schema the analysis scripts read). They also expect the
`saluca/` layout, so stage it first:

```bash
mkdir -p saluca && cp code/references.json code/creativity_ratio.py saluca/

# isolation design on a local model (plain / content / instruction / functional)
python code/creativity_ratio.py --provider ollama --gen local:gemma2:9b --load-type isolation --seeds 8

# frontier dose-response on a local model
python code/run_frontier.py --model gemma2:9b --seeds 6

# Claude models via the Claude CLI (`claude -p`), judged by local gemma2
export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama
python code/run_claude_cli.py --model haiku --seeds 3
python code/run_frontier_claude.py --model claude-fable-5 --seeds 3
```

These commands are taken from the scripts' own usage notes (translated from PowerShell) and were
not re-run on 2026-10-22. The Claude scripts hard-code the CLI path of the original author's
machine (`CLAUDE = ...` near the top of each file); edit it to your own `claude` executable before
running. To keep host hooks out of the output they run the CLI against a temporary config
directory and copy your `~/.claude/.credentials.json` into it.

Judging and human rating:

- `code/re_judge.py --judge opus`: re-scores groundedness with a stronger judge.
- `code/compare_judges.py`: gemma2 judge vs opus judge.
- `code/export_human_rating.py` then `code/merge_human_rating.py`: the blinded human pass;
  `human_rating/RATING_INSTRUCTIONS.md` is the rubric and `human_rating/rate_me.csv` the rated
  sheet.

## Architecture

The isolation design generates four conditions per domain and seed (`plain`, `content`,
`instruction`, `functional`) to separate load from instruction; the frontier design injects an
increasing amount of relevant reference text (`code/references.json`) and measures drift,
groundedness and divergence against load. Novelty is embedding drift from the domain's baseline
centroid; groundedness is an LLM judge score, cross-checked by a stronger judge and a blinded
human rater.

The full pipeline, from generation to figures, traced from the scripts' read and write paths. Paths
under `data/raw_logs_saluca/` and `saluca/` are the staged layout; the release keeps flat copies
in `data/` and `code/references.json`.

```mermaid
flowchart LR
    BR["build_reference.py"] --> REF["saluca/references.json"]
    REF --> RF["run_frontier.py"]
    REF --> RFC["run_frontier_claude.py"]
    CR["creativity_ratio.py"] --> LOGS[("data/raw_logs_saluca/creativity/ JSONL logs, one row per idea")]
    RCC["run_claude_cli.py"] --> LOGS
    RF --> LOGS
    RFC --> LOGS
    LOGS <-->|"adds grounded_strong"| RJ["re_judge.py"]
    LOGS --> EX["export_human_rating.py"]
    EX -->|"rate_me.csv, _key.json"| MG["merge_human_rating.py"]
    MG -->|"adds grounded_human"| LOGS
    LOGS --> AN["analyze_isolation.py, analyze_frontier.py, analyze_creativity.py, compare_judges.py"]
    LOGS --> MF["make_figures.py"]
    AN --> OUT["stdout tables"]
    MF --> FIG["saluca/paper/figures/*.png"]
```

| Script | Reads | Writes | Model calls |
|---|---|---|---|
| `build_reference.py` | nothing | `saluca/references.json` | gemma2:9b via Ollama |
| `creativity_ratio.py` | nothing | `ideas_PROVIDER_MODEL_LOADTYPE.jsonl` | generator and judge via Ollama or an OpenAI-compatible provider |
| `run_claude_cli.py` | `saluca/creativity_ratio.py` (prompts and judge) | `ideas_claude_cli_MODEL_isolation.jsonl` | `claude -p`; judge gemma2:9b via Ollama |
| `run_frontier.py` | `saluca/references.json` | `frontier/ideas_frontier_MODEL.jsonl` | generator and judge via Ollama |
| `run_frontier_claude.py` | `saluca/references.json` | `frontier/ideas_frontier_MODEL.jsonl` | `claude -p`; judge gemma2:9b via Ollama |
| `re_judge.py` | the six isolation and frontier logs | the same logs, with `grounded_strong` | `claude -p` (opus by default) |
| `export_human_rating.py` | the Fable frontier log (all six with `--all`) | `human_rating/rate_me.csv`, `human_rating/_key.json` | none |
| `merge_human_rating.py` | `rate_me.csv`, `_key.json` | the source logs, with `grounded_human` | none |
| `analyze_isolation.py` | `ideas_*_isolation.jsonl` | stdout | embeddings via Ollama |
| `analyze_frontier.py` | `frontier/ideas_frontier_*.jsonl` | stdout | embeddings via Ollama |
| `analyze_creativity.py` | every `ideas_*.jsonl` in the creativity folder (written for the pad and functional sweeps) | stdout | embeddings via Ollama |
| `compare_judges.py` | the six isolation and frontier logs | stdout | none |
| `make_figures.py` | the six isolation and frontier logs | `saluca/paper/figures/fig1..fig3` | embeddings via Ollama |

Log and `human_rating/` paths in the table are under `data/raw_logs_saluca/creativity/`.

Every generator also scores groundedness as it goes, through Ollama's OpenAI-compatible endpoint
(`http://localhost:11434/v1`). `run_claude_cli.py` and `run_frontier_claude.py` judge with
gemma2:9b; `creativity_ratio.py` (unless `--judge` is given) and `run_frontier.py` judge with the
generating model, gemma2:9b for the released local runs. The logs do not record the judge model.

`merge_human_rating.py` needs the private `_key.json` that `export_human_rating.py` writes. It is
not released, so the human pass cannot be re-merged from this clone; its result (`grounded_human`)
is already in the released Fable frontier log, and the released `human_rating/rate_me.csv` is the
sheet that was rated.

Contents:

- `paper.pdf` / `paper.md`: the preprint (figures embedded).
- `figures/`: fig1 cross-family, fig2 isolation, fig3 frontier dose-response (PNG, 160 dpi).
- `code/`: all generation, judging, analysis, and figure scripts + `references.json` (reference corpora).
- `data/`: raw per-idea JSONL logs (isolation ×4 models, frontier ×2 models, plus the earlier pad/functional sweeps).
- `human_rating/`: the blinded rating sheet and rubric.
- `.zenodo.json`: Zenodo deposition metadata. `deposit_zenodo.py` and `.zenodo_draft.json` are the
  author's deposit tooling and record of the published deposit; they are not part of the study.

`code/smoke_local.py`, `code/run_confound.py` and `code/analyze_confound.py` belong to the
companion world-model-collapse confound study referenced in the paper's introduction.
`smoke_local.py` and `run_confound.py` import a `src` package that is not in this repository and
do not run here. `analyze_confound.py` needs no `src`, but it reads
`data/raw_logs_saluca/confound/step_*.jsonl`, which is not released, so it prints "No step logs"
and exits. `code/_smoke_cloud.py` is a feasibility probe for a cloud model, not part of the study's
results.

What an agent can and cannot reach from this clone:

| Part | In this repo | Where it is |
|---|---|---|
| Harness, analysis and figure scripts, released logs, figures, preprint, rating sheet | yes | `code/`, `data/`, `figures/`, `paper.*`, `human_rating/` |
| The `src` package the confound scripts import | no | not in this repository, and its location is not recorded here |
| The confound study's step logs | no | not released |
| `_key.json` for the human rating | no | private by design (it unblinds the sheet) |
| Ollama models (`nomic-embed-text`, `gemma2:9b`) and the Claude CLI | no | external tools you install |
| The archived preprint | no | Zenodo, concept DOI 10.5281/zenodo.21149156 |

## Screenshots

The three figures as released (they regenerate from `data/` with the Quickstart):

![Bar chart: off-mainstream drift with and without relevant context for gemma2:9b, haiku, opus and Fable, with gains of +49%, +96%, +147% and +109%](figures/fig1_cross_family.png)

![Bar chart: drift under the plain, content, instruction and functional conditions for each of the four models](figures/fig2_isolation.png)

![Two line charts: drift against injected tokens for gemma2:9b and Fable, saturating within a few hundred tokens; and groundedness against load for the opus judge, gemma2 judge and the human rater](figures/fig3_frontier.png)

## Citation

Ruvalcaba, C. and the Saluca Agentic AI Research Team (2026). *Relevant Context-Load as a
Saturating Novelty Knob in Language Models: A Cross-Family Study.* Zenodo.
https://doi.org/10.5281/zenodo.21149156

That is the concept DOI, which always resolves to the latest version. This README previously cited
the version 1 DOI, 10.5281/zenodo.21149157, which remains valid for that version. See
[`CITATION.cff`](CITATION.cff).

## License

Split by path, declared file by file in [`REUSE.toml`](REUSE.toml) ([REUSE](https://reuse.software)
layout; the licence texts are in [`LICENSES/`](LICENSES/)):

- **Code and repository docs** (`code/*.py`, `deposit_zenodo.py`, README, AGENTS.md, llms.txt,
  the Zenodo metadata files): **Apache-2.0**, see [`LICENSE`](LICENSE).
- **Paper, figures and data** (`paper.md`, `paper.pdf`, `figures/`, `data/`, `human_rating/`,
  `code/references.json`): **CC-BY-4.0**, the licence of the Zenodo record.

## Links

- Preprint: https://doi.org/10.5281/zenodo.21149156 (concept DOI)
- Zenodo record, version 1: https://zenodo.org/records/21149157

_Last updated: 2026-10-22_
