# Human groundedness-rating pass

**Goal:** settle whether groundedness really erodes with context-load on the frontier
model. The automated judges disagreed (opus: yes, 0.94→0.72; gemma2: flat) and they
correlate only r=0.20, so a human rating is the tiebreaker. The sheet is **blinded** —
the load/condition is hidden and the ideas are shuffled — so you can't bias toward the
expected answer.

## How to rate
1. Open `rate_me.csv` (Excel / Sheets / any editor).
2. For each row, read the `idea` and put a **1–5** integer in `groundedness_1to5`.
3. Save (keep it as CSV).
4. Run: `python saluca\merge_human_rating.py`

## The rubric — rate GROUNDEDNESS only, NOT novelty
Ask: *"Is this plausible, coherent, and concrete — could it be real?"* Do **not** reward or
penalize how original/surprising it is; a boring idea can be perfectly grounded, and a wild
idea can be grounded if it's internally coherent and not physically absurd.

- **5** — solid: plausible, specific, mechanistically coherent, could be a real hypothesis/feature/strategy.
- **4** — mostly solid: plausible with a minor stretch or vagueness.
- **3** — mixed: plausible in parts but hand-wavy, over-general, or one questionable leap.
- **2** — shaky: mostly implausible, confused, or internally inconsistent.
- **1** — nonsense: far-fetched, incoherent, or physically/logically absurd.

Rate on the idea's own terms within its domain (a sleep hypothesis, a note-app feature,
a coffee-roaster strategy). ~54 ideas; ~10–15 minutes.

## What happens next
`merge_human_rating.py` writes your scores back as `grounded_human`, then prints the
Fable-frontier groundedness **by load** for all three judges (human / opus / gemma2) and
the human-vs-model correlations. If human-rated groundedness **declines with load**, it
corroborates opus and the paper's erosion claim stands (and firms up). If it stays **flat**,
opus over-read it and we soften the claim further. Either way we then note the human pass
in the paper and re-push.
