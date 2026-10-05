# Data for experiment 6

Two arms, paired example by example, differing only in the prompt string.

| arm | file | rows |
|---|---|---|
| instruction (the paper's prompt) | `../../../data/clapnq_softinstr.jsonl` | 300 |
| | `../../../data/pubmedqa_softinstr.jsonl` | 500 |
| Q/A (the instruction line deleted) | `../../../data/clapnq_qa_prompt.jsonl` | 300 |
| | `../../../data/pubmedqa_qa_prompt.jsonl` | 500 |

```
instruction:  {evidence} Question: {q}
              This question is answered completely with evidence from the {passage|abstract}.
              Answer:

Q/A:          {evidence} Question: {q} Answer:
```

Only `prompt` and `modernbert_prompt` differ. Ids, targets, references and
`source_text` are carried through untouched, which is what makes the two arms a
paired comparison rather than two separate runs.

sha256 of each file: `../../../data/checksums.json`.
