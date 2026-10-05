# Data for experiment 11

Nothing is generated here; the examples are selected from outputs that already exist.

| | |
|---|---|
| evaluation set | `../../../data/clapnq_softinstr.jsonl` (300 questions) |
| outputs | `../../../artifacts/predictions/clapnq__nash_maxgap__modernbert_large.jsonl.gz` |
| system | ModernBERT-Large, Nash decoding from the all-[MASK] canvas |

`--percentiles --dataset coqa` and `--percentiles --dataset pubmedqa` select from the
corresponding files instead. The question shown for each example is recovered from the
prompt the system received, not from a separate field.
