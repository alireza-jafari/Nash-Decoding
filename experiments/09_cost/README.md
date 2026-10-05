# 9. The cost of reaching an equilibrium

**Paper:** Figure 4 (body), Figure 6 and Appendix I. **Needs a GPU:** only to
measure on your own device.

```bash
python experiments/09_cost/measure_cost.py
python experiments/09_cost/measure_cost.py --measure coqa --gpu 0
```

## What it shows

Table 1 reports quality at a matched token budget. This reports the price. Five systems
on each dataset, on B200 hardware:

| system | CoQA s | calls | PubMedQA s | calls | CLAPNQ s | calls |
|---|---:|---:|---:|---:|---:|---:|
| ModernBERT-L, all-mask | **0.49** | 48 | 10.5 | 1520 | 17.4 | 2710 |
| ModernBERT-L, L2R | 0.21 | **20** | 1.2 | 150 | 1.7 | 249 |
| Falcon 7.2B | **0.26** | 8 | 0.8 | 49 | 1.1 | 64 |
| GPT-2 Large 774M | 0.07 | 7 | 0.3 | 50 | 0.4 | 60 |
| OPT-350M 331M | 0.04 | 7 | 0.2 | 50 | 0.3 | 60 |

Of the all-mask row's 0.49 s and 48 calls on CoQA, **0.41 s and 38 calls fall before
the canvas is full** and the rest is refinement of a complete canvas; the L2R row splits
into 8 construction calls and 12 refinement calls. That split is what Figure 4 draws as
the light and dark segment of each bar.

Mean answer budgets are 7.5, 48.3 and 63.8 tokens, so the growth is **faster than
linear in the budget**: the gap is re-evaluated at every position after each committed
token, so the number of scored sequences grows with the product of answer length and
update count.

**L2R construction reduces this cost.** It requires approximately 20, 150 and 249 calls,
compared with 48, 1,520 and 2,710 for all-mask decoding: reductions of approximately
2.4x, 10.1x and 10.9x. On the full evaluation sets, the corresponding ModernBERT-Large
F1 scores are 56.93 versus 58.12 on CoQA, 21.91 versus 24.28 on PubMedQA, and 36.12
versus 38.04 on CLAPNQ. The two constructions also distribute computation differently:
L2R fills the canvas in T sequential predictions before refinement begins, so
refinement accounts for most of its model calls, while most all-mask calls occur before
the last mask is filled -- on CoQA, 8 calls of construction and 12 of refinement for
L2R, 38 and 9 for all-mask.

## How to read these numbers

**Model-call counts should be interpreted alongside wall-clock times**, because a call
represents different amounts of work across the two model families. After its initial
prompt evaluation, an autoregressive decoder processes each new token using a cached
prefix, whereas the masked implementation re-encodes the prompt-and-canvas sequence for
its conditional evaluations.

**The overhead is primarily architectural rather than inherent to the equilibrium
objective**: masked models that can evaluate conditional probabilities for multiple
positions in parallel, without a need for masking, can substantially reduce the cost.
The timings characterize these models and this implementation under the stated
hardware and numerical settings.

## How the measurement was made

All five systems were measured on B200 hardware under the same numerical settings:
float32 with TF32 disabled.

| dataset | measured on | measurements |
|---|---|---|
| CoQA | the full 1,814-turn cohort, `Q:` / `A:` dialogue prompt | `artifacts/timing/timing_coqa_numbers.json` |
| PubMedQA | the full 500 questions | `figure_4_6_cost` in `paper/paper_values.json` |
| CLAPNQ | a fixed 60-item subsample, every fifth item, shared by all five systems | `artifacts/timing/clapnq_timing_b200.json` |

The CLAPNQ subsample has a mean reference answer length of 49.0 words, compared with
51.7 for the full set; its mean L2R cost is 249 calls, compared with 248 on all 300
items.

`measure_cost.py` prints three blocks:

* the figure's seconds and calls;
* the forward passes the evaluation runs recorded on the full evaluation sets, in
  `artifacts/predictions/`, split at the update that fills the last [MASK];
* the timing files.

To measure on your own device:

```bash
python experiments/09_cost/measure_cost.py --measure coqa --gpu 0
```

That runs the five systems one at a time on one GPU and reports seconds and model calls
per question for that device.

## Data

`data/clapnq_sub.jsonl` -- the fixed 60-item CLAPNQ subsample. CoQA and PubMedQA are
timed over their full evaluation sets; on CoQA that is the 1,814-turn cohort of
`data/coqa_paperprompt.jsonl.gz`, not all 7,983 turns.

## Results

`results/cost.json` holds three blocks:

* `figure_6_b200` -- the seconds and calls of Figures 4 and 6, read from
  `paper/paper_values.json`.
* `recorded` -- the forward-pass counts the evaluation runs themselves recorded on the
  full evaluation sets, per system, split at the update that fills the last [MASK].
* `timing` -- the CoQA and CLAPNQ timing files, normalised into the figure's row order.
