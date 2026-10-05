# Data for experiment 8

Two things, both specific to this experiment.

**The grounded prompts.** The soft-instruction files with one clause appended,
`Answer using the {passage|abstract}'s own words.`:

| file | rows |
|---|---:|
| `../../../data/coqa_grounded.jsonl.gz` | 1,814 |
| `../../../data/pubmedqa_grounded.jsonl` | 500 |
| `../../../data/clapnq_grounded.jsonl` | 300 |

All three coordinate rules read the same file, so the prompt is byte-identical
across the arms.
Ids, targets, references and `source_text` are the same as in the soft-instruction
files.

**The development/test split**, in `dev_test_split.json` here rather than recomputed.

| dataset | development | test |
|---|---:|---:|
| CoQA | 1,078 | 736 |
| PubMedQA | 302 | 198 |
| CLAPNQ | 181 | 119 |

Prompt variants and ban levels were chosen on the development portion; the tables
report the full sets. The split is stored as an id list.

The outputs are in `../../../artifacts/predictions/` as
`<dataset>__coordinate__{random,low_confidence,nash}.jsonl.gz`. The file names are
the internal arm names: `random` is the paper's **Uniform** rule and
`low_confidence` its **Confidence** rule.
