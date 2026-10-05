# Stored per-example outputs

Everything the runs produced, distilled to what the tables need. About 15 MB in
total, so every table and figure can be rebuilt on a laptop, without a GPU or a model
download.

These are *outputs*, not summaries. Every reported number is computed from them by
`nashlib/metrics.py` rather than read from a score file.

| directory | what it holds | read by |
|---|---|---|
| `predictions/` | one gzipped JSONL per (dataset, family, system): the generated answer and the per-example telemetry, for every evaluated example | experiments 4, 5, 6, 7, 8, 9, 11 |
| `trajectories/` | the Nash gap and answer F1 at every update, for ModernBERT-Large on all three datasets under all-mask, and on CLAPNQ under L2R | experiment 5 |
| `wikitext/` | the 500 WikiText-103 refinement trajectories update by update (`refinement_trajectories.json`), and the GPT-2 XL referee's continuation log-perplexity along each one (`shard0-3.json`) | experiment 3 |
| `timing/` | wall-clock and forward-pass measurements for CoQA and CLAPNQ | experiment 9 |

`predictions/` holds two CoQA sets. `coqa__*` is the 1,814-turn cohort the tables
report; `coqa_full__*` is the same systems on all 7,983 development turns, which is
the right-hand block of Table 7. Table 7 reports Calls and Chars for the cohort, so
the cohort files carry the full telemetry and the full-set Nash files carry `calls`,
`calls_to_full`, `final_gap`, `slots`, `status` and `updates`; in the full-set files
`calls` counts one forward pass per answer position per scan.

## predictions/

Named `<dataset>__<family>__<system>.jsonl.gz`, one line per evaluated example:

| family | what produced it |
|---|---|
| `oneshot` | a frozen masked encoder filling every canvas slot from a single forward pass |
| `autoregressive` | greedy left-to-right decoding at the oracle budget |
| `nash_maxgap` | Nash decoding from the all-[MASK] canvas |
| `nash_l2r` | Nash decoding from left-to-right construction |
| `coordinate` | the shared-checkpoint coordinate-rule comparison |
| `init` | refinement only, from a supplied initial sequence |
| `qaprompt_*` | the same decoders under the prompt ablation's Q/A prompt |

Every line carries `id` and `text`. A one-shot row records `slots` and `calls`, which
is 1 by construction. An autoregressive row records its `budget`, which is also its
forward-pass count: a greedy autoregressive model performs one forward pass per
generated token. A Nash row records `slots`, `status` (`equilibrium`, `cycle` or
`cap`), `updates`, `unmask_updates`, `refine_updates`, `final_gap`, `calls`,
`calls_to_full` and `step_last_mask_filled`. Those fields are the `Calls` column of
the appendix tables, the convergence rates, and the split of the cost bars into
construction and refinement.

Read them through `nashlib/artifacts.py`:

```python
from nashlib import artifacts
rows = artifacts.load_predictions("clapnq", "nash_maxgap", "modernbert_large")
artifacts.telemetry(rows)   # calls, seconds, equilibrium rate, cycles, ...
```

## Trajectories

`trajectories/` carries the gap and F1 curves the figures are drawn from, and
`wikitext/` carries the update-by-update trajectories of experiment 3. The complete
raw trajectories of the question-answering runs -- the gap, the pseudo-log-likelihood
and the full token state after every update, for every example of every system -- are
roughly 64 GB compressed, so the repository stores these compact files instead.

Re-running a system writes its raw trajectories under `experiments/*/runs/`, which
`.gitignore` excludes. `tools/build_trajectories.py` turns a run into the compact
file: it replays the run's per-example update log and scores every intermediate canvas
with a tokenizer alone -- no model, no GPU.

## Notes on the contents

**These are verbatim model outputs.** Nothing is filtered, truncated or cleaned.

**Wall-clock comparisons use `timing/`.** The cost figures are drawn from the timing
measurements, which were taken on B200 hardware for all five systems.

**Forward-pass counts are hardware-independent.** They should be read alongside the
wall-clock times, because a call represents different amounts of work in the two model
families: an autoregressive step processes one token against a cached prefix, while a
masked scan re-encodes the prompt and canvas.
