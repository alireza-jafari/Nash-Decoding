# 10. What the three benchmarks contain

**Paper:** Tables 5 and 6, Appendix D. **Needs a GPU:** no (a tokenizer download).

```bash
python experiments/10_dataset_statistics/compute_statistics.py
python experiments/10_dataset_statistics/compute_statistics.py --table 6 --reference both
```

## Why these three benchmarks

They were chosen to vary answer length, domain, and the relation between the answer
and its context -- conversational conditioning, long-form composition, and scientific
conclusion generation. The statistics below are what make that concrete, and they
explain why no single backbone wins everywhere.

**CoQA: predominantly span-based.** About four in five answers are a contiguous
context span after normalization, the longest shared run accounts for 91.9% of answer
length, and only 0.9% of answer tokens are absent from the context. These are computed
on the 1,814-turn cohort, which is what Tables 5 and 6 describe; `coqa_full` has its own
length distribution and is not tabulated here. The task largely involves selecting a
short span, with its length supplied by the oracle budget. The full-context copying
score of 4.9 F1 reflects the difference in length between a ~355-token passage and a
7.5-token answer; it does not measure the difficulty of locating the span.

**CLAPNQ: composition from passage wording.** 97.6% of reference tokens occur in the
passage, but only 4% of complete answers form a contiguous span, and the longest
shared run covers just over half an answer. Answers typically combine material from
two or three sentences about three and a half positions apart. Copying the whole
passage scores 50.7 F1 and the best single sentence 67.6 -- reference-informed
diagnostics showing that passage reuse receives substantial credit under an overlap
metric.

**PubMedQA: greater lexical novelty.** Two fifths of reference tokens and more than
three quarters of reference bigrams are absent from the abstract; 41.0% of answers
have fewer than half their content words in the context, and none is a complete
contiguous span. Selecting more sentences cannot recover words that are absent from
the entire abstract. Meanwhile 70.9% of *question* tokens occur in the abstract -- the
highest of the three, against 59.9% on CLAPNQ and 53.6% on CoQA -- so the question
overlaps its context heavily even when the conclusion requires considerable new
wording.

## Scale and length (Table 5)

| | CLAPNQ | PubMedQA | CoQA |
|---|---:|---:|---:|
| Questions scored | 300 | 500 | 1,814 |
| Distinct contexts | 297 | 500 | 459 |
| References per question | 1.62 | 1.00 | 4.00 |
| Context tokens | 195.3 | 287.0 | 355.2 |
| Context words | 170.8 | 202.3 | 267.0 |
| Question tokens | 10.4 | 17.8 | 6.6 |
| Full prompt tokens | 222.7 | 321.7 | 510.9 |
| Dialogue history + question | -- | -- | 155.7 |
| Answer tokens | 63.8 | 48.3 | 7.5 |
| Answer words | 51.7 | 39.1 | 5.6 |
| Answer median | 56 | 45 | 7 |
| Answer 90th percentile | 107 | 77 | 11 |
| Answer maximum | 231 | 154 | 40 |
| Answer GPT-2 BPE tokens | 63.1 | 50.2 | 7.4 |
| Sentences per answer | 2.29 | 1.91 | 1.03 |

The prompt rows measure the prompt as assembled. On CoQA that is the `Q:` / `A:`
dialogue format of Radford et al. (2019), and "dialogue history + question" is the
prompt minus the context, so it includes the `Q:` / `A:` scaffolding.

## Lexical overlap and arrangement (Table 6)

| | CLAPNQ | PubMedQA | CoQA |
|---|---:|---:|---:|
| Gold tokens in context (%) | 97.6 | 60.0 | 99.1 |
| Gold content words in context (%) | 97.4 | 53.8 | 99.0 |
| Novel bigrams (%) | 11.5 | 76.8 | 8.7 |
| Question tokens in context (%) | 59.9 | 70.9 | 53.6 |
| Answers below 50% overlap (%) | 0.0 | 41.0 | 0.5 |
| Answer is a contiguous span (%) | 4.0 | 0.0 | 78.8 |
| Longest run / answer length (%) | 53.7 | 12.8 | 91.9 |
| Sentences selected (90% target) | 2.6 | 3.4 | 1.1 |
| Answers with two or more sentences (%) | 92.7 | 95.2 | 5.2 |
| Answers with three or more sentences (%) | 40.3 | 71.4 | 0.4 |
| First-to-last sentence distance | 3.7 | 6.0 | 0.2 |
| Copy the whole context (F1) | 50.7 | 18.4 | 4.9 |
| Copy the best sentence (F1) | 67.6 | 34.7 | 47.2 |

## Conventions that change the numbers

**Token counts** use `answerdotai/ModernBERT-Large` at the pinned revision, with the
answer tokenized as `" " + target` -- with the leading space, matching how the decoding
budget is computed. The GPT-2 row is included to show how much a different tokenizer
moves the same answer.

**Lexical statistics** use the evaluation normalizer -- lowercase, drop ASCII
punctuation and the articles, collapse whitespace -- so "appears in the context" means
the same thing here as it does inside the F1 metric. The content-word measures use the
stopword list in `compute_statistics.py` (`STOPWORDS`).

**Multiple references** need a convention. `--reference best-overlap` (the default)
picks one reference per question, the one with the highest token overlap with the
context, and computes everything from it; `--reference per-measure` lets each statistic
pick its own independently. They agree on PubMedQA, which has one reference, and diverge
on CoQA, which has four: the contiguous-span rate is 78.8% under the first and 92.8%
under the second. The table above is the default; `--reference both` prints them side
by side.

## Data

The three evaluation sets, and nothing else. Every number is computed from them. On CoQA
that is `data/coqa_paperprompt.jsonl.gz` read as the 1,814-turn `coqa` cohort -- the same
file and the same filter the result tables use.

## Results

`results/dataset_statistics.json`.
