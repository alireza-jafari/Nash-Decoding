# Evaluation data

One line per evaluated question, with the prompt already assembled. The prompt is a
field of the data rather than a template applied at run time, so what each model
family receives is a property of the data.

```
id                 stable example identifier
prompt             what the autoregressive systems receive
modernbert_prompt  what the masked encoders receive
target             the gold answer, which sets the oracle length budget
references         every reference answer, for best-reference scoring
source_text        the passage or abstract, for the dataset statistics
metadata           per-dataset provenance
```

**The two prompt fields.** On PubMedQA and CLAPNQ every system receives the same
string. CoQA uses the dialogue prompt of Radford et al. (2019), and Appendix E.1
specifies that the masked encoders receive it whitespace-normalised: the `Q:`/`A:`
line breaks become single spaces and one leading space is prepended
(`nashlib.datasets.masked_prompt_for_coqa`). The passage's own paragraph breaks are
kept.

## The three evaluation sets

| file | rows | evaluated as | references / question | mean answer (ModernBERT tokens) |
|---|---:|---|---:|---:|
| `coqa_paperprompt.jsonl.gz` | 7,983 | `coqa` (1,814) and `coqa_full` (7,983) | 4 | 7.5 and 3.2 |
| `pubmedqa_softinstr.jsonl` | 500 | `pubmedqa` | 1 | 48.3 |
| `clapnq_softinstr.jsonl` | 300 | `clapnq` | 1.62 | 63.8 |

**CoQA** — one file of all 7,983 development turns, scored two ways. `coqa` is the
1,814 turns whose gold answer has at least five ModernBERT tokens, which is what
Table 1 and the left block of Table 7 report: 460 conversations over 459 distinct
passages, spanning Wikipedia (447), CNN (400), RACE (360), Project Gutenberg (351)
and MCTest (256). `coqa_full` is every turn, Table 7's right block.

The cohort is **derived** from the file: every row carries
`metadata.gold_mb_tokens`, and `coqa` is the rows where that is at least 5, so both
cohorts share one copy of every prompt, reference and gold answer.

Why the filter exists: nearly half the gold answers in the full development split are
a single word, and 18.4% are a bare yes or no — 1,472 of 7,983, with 1,538 turns
answering exactly `yes`, `no` or `unknown`. A one-token continuation gives the token
game a single player, with no other generated tokens to influence or respond to, so
the threshold leaves enough positions for meaningful token interactions. It is an
experimental design choice, not an algorithmic requirement, and every system is
evaluated on the same retained questions. The results on the full development set,
including all questions, are reported alongside (Table 7).

Dialogue history is folded into the prompt using the **gold** answers of the preceding
turns, so every turn is an independent evaluation and all models receive the same
conversational context.

**PubMedQA** — the official 500-question test half of PQA-L, reconstructed from the
released seed-0 two-fold split. Each question has a distinct abstract and one
reference: the abstract's conclusion, which is excluded from the input context. The
corpus also labels each question yes / no / maybe (276 / 169 / 55 here); that label
is never used as a target, because the task here is to generate the conclusion, not
to classify.

**CLAPNQ** — pairs a Natural Questions passage with either a cohesive long-form answer
or no answer at all; the 300 *answerable* questions of its development split are used.
They cover 297 distinct passages and have 485 reference answers in total: 160
questions have one, 131 have two and 9 have seven.

## Prompt variants

| file | used by | difference |
|---|---|---|
| `clapnq_qa_prompt.jsonl`, `pubmedqa_qa_prompt.jsonl` | experiment 06 | the instruction line is deleted, leaving `{evidence} Question: {q} Answer:` |
| `coqa_grounded.jsonl.gz`, `pubmedqa_grounded.jsonl`, `clapnq_grounded.jsonl` | experiment 08 | one clause appended: `Answer using the {passage\|abstract}'s own words.` |
| `clapnq_sub.jsonl` | experiment 09 | the fixed 60-item CLAPNQ subsample, every fifth item, shared by all five timed systems |

**`coqa_grounded.jsonl.gz` holds the CoQA prompt of Appendix H.** The coordinate-rule
comparison specifies its own prompt and canvas and uses the soft-instruction wording,
while the result tables use the dialogue prompt. `nashlib.datasets.load_dataset` takes
the prompt shape explicitly (`prompt_style="soft_instruction"`).

In each variant the `prompt` and `modernbert_prompt` fields change and ids, targets,
references and `source_text` are carried through untouched, so the arms are paired
example by example.

## Other data

`wikitext512_500.json` holds the 500 WikiText-103 test prompts of experiment 3, stored
as **token ids** rather than text. "Exactly 512 tokens" is a property of the ids;
re-tokenizing the text would not guarantee it, and the experiment's whole point is a
controlled continuation length.

**Two files ship gzipped.** `coqa_paperprompt.jsonl.gz` is 44.5 MB as text and 1.1 MB
compressed, and `coqa_grounded.jsonl.gz` 11.2 MB against 0.6 MB, because a CoQA
conversation's passage is repeated on every one of its turns and most turns share one.
The smaller files ship as text, where compressing buys little and makes them harder to
look at. `nashlib.datasets` opens either extension the same way, so nothing that reads
them needs to know which it is, and `write_jsonl` zeroes the gzip timestamp so a
rewritten file has the same bytes and therefore the same checksum.

`checksums.json` records the sha256 of every file here.

## Rebuilding these files

The evaluation sets are derived from three public corpora, each distributed under its
own license, which is why they are included here in derived form rather than
redistributed wholesale:

* **CoQA** — Reddy et al., 2019. Development split, CC BY-SA 4.0 for the Wikipedia
  and MCTest portions and per-source terms otherwise. Prompt format after
  Radford et al., 2019.
* **PubMedQA** — Jin et al., 2019. PQA-L, MIT license.
* **CLAPNQ** — Rosenthal et al., 2025. Development split, Apache 2.0, derived from
  Natural Questions (CC BY-SA 3.0).

What the derivation did is written down above and in
`experiments/10_dataset_statistics`: the >= 5 ModernBERT-token CoQA filter
and the `Q:`/`A:` dialogue layout, the seed-0 PubMedQA split and the withheld
conclusion, the answerable-only CLAPNQ selection, and the prompt assembly.

## The `metadata` field

The dataset files carry no author, machine or run information, and no timestamps.

The `metadata` field carries corpus provenance — a conversation id and source for
CoQA, a passage title for CLAPNQ, the split name and yes/no/maybe label for PubMedQA
— and, on the CLAPNQ and PubMedQA files, `prompt_style`, a name from the pipeline
that assembled the prompts.
