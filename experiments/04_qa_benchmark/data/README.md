# Data for experiment 4

The evaluation sets, one file per dataset subfolder — but four dataset ids, because
CoQA is scored twice out of one file:

| dataset id | file | rows scored | references / question |
|---|---|---:|---:|
| `coqa` | `../../../data/coqa_paperprompt.jsonl.gz` | 1,814 | 4 |
| `coqa_full` | the same file | 7,983 | 4 |
| `pubmedqa` | `../../../data/pubmedqa_softinstr.jsonl` | 500 | 1 |
| `clapnq` | `../../../data/clapnq_softinstr.jsonl` | 300 | 1.62 |

`coqa` is the turns whose gold answer has at least five ModernBERT tokens, and it is
*derived* from the file by `metadata.gold_mb_tokens >= 5` rather than written out as a
second file.

CoQA's `prompt` is the `Q:` / `A:` dialogue format of Radford et al. (2019) and its
`modernbert_prompt` is that string whitespace-normalised, as Appendix E.1 specifies;
PubMedQA and CLAPNQ give every system the same string.

Each dataset's own README (`coqa/README.md`, `pubmedqa/README.md`,
`clapnq/README.md`) describes that dataset — the CoQA length filter and dialogue
history, PubMedQA's withheld conclusion, CLAPNQ's multiple references — and which
models the paper reports for it.

The per-example outputs the tables are rebuilt from are in
`../../../artifacts/predictions/`, named `<dataset>__<family>__<system>.jsonl.gz`.
Row counts and sha256s: `../../../data/checksums.json`.
