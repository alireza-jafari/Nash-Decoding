# Data for experiment 9

`../../../data/clapnq_sub.jsonl` — the fixed 60-item CLAPNQ subsample the timing
figure is measured on: every fifth item of the 300, shared by all five systems. Its
mean gold answer is 49.0 words against 51.7 for the full set, and it gives 249
left-to-right calls against the 248 measured on all 300.

CoQA and PubMedQA are timed over their full evaluation sets, so they use
`../../../data/coqa_paperprompt.jsonl.gz` and `../../../data/pubmedqa_softinstr.jsonl`
directly. On CoQA that means the reported 1,814-turn cohort, not all 7,983 turns of
the file: the cost figure accompanies Table 1, and `measure_cost.py` loads the corpus
through `datasets.load_dataset(..., dataset="coqa")`, which applies the cohort filter.

The measurements are in `../../../artifacts/timing/` (`timing_coqa_numbers.json` and
`clapnq_timing_b200.json`) and, for PubMedQA, in `../../../paper/paper_values.json`.
