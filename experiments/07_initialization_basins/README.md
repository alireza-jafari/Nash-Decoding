# 7. Initialization and equilibrium basins

**Paper:** Table 2 (body), Appendix G.4-G.5. **Needs a GPU:** only to
regenerate.

Table 2 is **Nash Equilibria Basins on PubMedQA** -- the four largest backbones, both
constructions, on PubMedQA. This experiment computes the comparison for all six
backbones, both constructions, on all three datasets; Table 2 is its PubMedQA column
for the four largest backbones.

```bash
python experiments/07_initialization_basins/score_initialization.py
python experiments/07_initialization_basins/score_initialization.py --extra-inits
```

## What it shows

Multiple Nash equilibria exist, and which one a run reaches depends on where it
starts. This experiment holds everything else -- prompt, oracle budget, banned tokens,
the max-gap refinement rule, tolerance zero -- and varies only how the first completed
sequence comes into existence. Six backbones, three datasets, two constructions.

| PubMedQA (Table 2) | Size | L2R F1 | R-L | R-Ls | All-mask F1 | R-L | R-Ls |
|---|---:|---:|---:|---:|---:|---:|---:|
| mmBERT-base | 308M | 4.22 | 6.11 | 5.38 | 3.89 | 5.30 | 4.59 |
| RoBERTa-Large | 355M | 22.23 | 19.79 | 20.77 | **24.99** | **21.17** | **22.69** |
| ModernBERT-Large | 395M | 21.91 | 19.88 | 19.99 | 24.28 | 20.33 | 21.36 |
| Ettin-400m | 396M | 17.41 | 17.29 | 17.36 | 22.77 | 20.05 | 21.65 |

**All-mask** does not separate construction from refinement. The max-gap rule is
applied directly to the all-[MASK] canvas; at an unwritten position the current token
is [MASK], which is a legal action, so writing an empty slot and rewriting a filled one
are scored on the same scale and compete directly, and the two can interleave. With
ModernBERT-Large, **181 of 1,814** CoQA trajectories, **276 of 500** PubMedQA
trajectories and **198 of 300** CLAPNQ trajectories revise an already-written position
before the canvas is complete.

**Left-to-right** fills the T slots in index order, one model call per position with
[MASK] excluded so every step commits a real token, and revises nothing until all T
slots are full; then the identical refinement runs. Under this construction the
interleaving count above is zero by definition.

Left-to-right is a masked model *writing* in left-to-right order, not a simulation of
an autoregressive model. The backbone is bidirectional and does attend to positions to
the right of i; those positions simply carry no content yet.

## Different initializations, different equilibria

The two schemes follow different early decoding trajectories, after which both proceed
to equilibrium. The resulting differences in final performance indicate that the two
trajectories can converge to different equilibria. Although the performance of Nash
decoding depends on initialization, all-mask initialization is often effective, as
shown in Table 1.

## More initializations

`--extra-inits` adds refinement-only runs that start from a supplied sequence --
uniformly random tokens, a one-shot parallel decode of the whole canvas, the L2R
construction and the confidence-ordered construction -- for ModernBERT-Large on CoQA
and PubMedQA. For each start the script reports F1 before and after refinement, the
number of updates and the number of model calls.

## Data

`data/coqa_paperprompt.jsonl.gz`, read as the 1,814-turn `coqa` cohort,
`data/pubmedqa_softinstr.jsonl` (500) and `data/clapnq_softinstr.jsonl` (300) -- the
same files the result tables use: the initialization ablation uses the same
dataset-specific prompt and changes only the initialization procedure. See
`data/README.md` in this folder.

## Results

`results/initialization.json` -- `all_backbones`, both constructions on all three
datasets for all six backbones; `table_2`, its PubMedQA column restricted to the four
largest backbones; the interleaving counts; and the extra initializations.
