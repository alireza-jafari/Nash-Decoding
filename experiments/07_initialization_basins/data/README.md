# Data for experiment 7

The three evaluation sets, unchanged — the constructions differ in the decoder, not
in the data:

* `../../../data/coqa_paperprompt.jsonl.gz`, cohort `coqa` (1,814 of its 7,983 rows)
* `../../../data/pubmedqa_softinstr.jsonl` (500)
* `../../../data/clapnq_softinstr.jsonl` (300)

CoQA uses the dialogue prompt of Appendix E.1, the same one the result tables use, so
the two constructions are compared under the prompt the paper reports them under. The
masked encoders receive its whitespace-normalised form; both constructions receive the
same one, which is what makes the pair a controlled comparison.

The outputs compared here are in `../../../artifacts/predictions/`:

| family | what produced it |
|---|---|
| `<dataset>__nash_maxgap__<backbone>` | all-mask construction |
| `<dataset>__nash_l2r__<backbone>` | left-to-right construction |
| `<dataset>__init__{random,one_shot,l2r,confidence}` | refinement only, from a supplied start (`--extra-inits`) |

Six backbones on the first two families, on all three datasets: ModernBERT-Base,
Ettin-150m, mmBERT-base, RoBERTa-Large, ModernBERT-Large, Ettin-400m. The
refinement-only runs are ModernBERT-Large on CoQA and PubMedQA.
