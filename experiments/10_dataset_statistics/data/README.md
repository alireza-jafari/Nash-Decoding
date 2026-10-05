# Data for experiment 10

The three evaluation sets, and nothing else — every number in Tables 5 and 6 is
computed from them plus a tokenizer:

* `../../../data/clapnq_softinstr.jsonl` (300)
* `../../../data/pubmedqa_softinstr.jsonl` (500)
* `../../../data/coqa_paperprompt.jsonl.gz`, cohort `coqa` (1,814 of its 7,983 rows)

Table 5's CoQA column describes the cohort, so the statistics are computed on the
1,814 rows rather than the whole file. The prompt decomposition is dataset-specific:
the current question sits on the last `Q:` line of the CoQA dialogue prompt and on the
`Question:` line of the other two, and `nashlib.datasets.current_question` handles
both layouts.

Tokenizers: `answerdotai/ModernBERT-large` @ `45bb4654` for every count, and `gpt2`
@ `607a30d7` for the one row that shows how much a different tokenizer moves the
same answers. Answers are tokenized as `" " + target`, with the leading space,
matching how the decoding budget is computed.
