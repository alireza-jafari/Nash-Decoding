# Data for experiment 5

Not an evaluation set: this experiment reads trajectories, which are in
`../../../artifacts/trajectories/`.

| file | rows | what a row holds |
|---|---:|---|
| `coqa_all_mask.json` | 1,814 | `gap` and `f1` at every update, plus `slots` and `status` |
| `pubmedqa_all_mask.json` | 500 | the same |
| `clapnq_all_mask.json` | 300 | the same |
| `clapnq_l2r.json` | 300 | the same, under left-to-right construction |

All four are ModernBERT-Large. The F1 at update k is the score of the
partially-decoded answer as it stands at that point, with unwritten slots
contributing nothing.

The revision share the script reports comes from the run telemetry in
`../../../artifacts/predictions/<dataset>__nash_maxgap__modernbert_large.jsonl.gz`,
not from the trajectory length, because the decoder records `refine_updates`
directly.
