# CoQA

**Table 7**, which has two blocks. The left block is the 1,814 development turns whose
gold answer has at least five ModernBERT tokens -- 460 conversations, 459 distinct
passages, four references each, mean gold answer 7.5 ModernBERT tokens -- and is what
Table 1 reports. The right block is all 7,983 development turns, mean gold answer 3.2
tokens. Both blocks come out of one file.

```bash
python experiments/04_qa_benchmark/coqa/models_coqa.py          # the 20 rows
python experiments/04_qa_benchmark/coqa/run_ar_coqa.py --all --gpu 0
python experiments/04_qa_benchmark/coqa/run_nash_coqa.py --all --gpu 0
python experiments/04_qa_benchmark/coqa/run_oneshot_coqa.py --all --gpu 0
```

## Models: 3 one-shot masked + 14 autoregressive + 3 Nash backbones

From GPT-2 Small (124M) to OLMo-1.7 (7B): CoQA answers are short, so a run is cheap. It
is also the shortest-answer benchmark, which is why the cost comparison of Figure 4 is
drawn here. BLOOM-3B is the only
checkpoint in the repository for which `transformers` ships no SDPA kernel, so it runs
under eager attention; that is not a flag a caller passes but a field on its `ModelSpec`,
`attention="eager"`.

The three masked encoders -- RoBERTa-Large, ModernBERT-Large and Ettin-400m -- each
appear twice, decoded one-shot and under Nash decoding; mmBERT-base, the fourth masked
model Section 4 names, is run here too, for the initialization study. `../README.md`
compares the three model sets.

## What makes CoQA different

**The prompt is the dialogue format.** CoQA follows Radford et al. (2019), as Appendix
E.1 specifies: the passage, then the preceding turns as `Q: {q}` / `A: {a}` lines
carrying the **gold** answers of those turns, then the current question and a bare
`A:`. Because the history supplies gold answers, each question is evaluated
independently and all models receive the same conversational context. PubMedQA and
CLAPNQ use the soft-instruction prompt.

**The masked encoders receive a whitespace-normalised version of that prompt**
(Appendix E.1): the `Q:`/`A:` line breaks are replaced by single spaces and one leading
space is added, so the canvas is appended to a string ending in `A:`. The
transformation is `masked_prompt_for_coqa()` in `nashlib/datasets.py`; the passage's
own paragraph breaks are kept.

**Two cohorts, one file.** `data/coqa_paperprompt.jsonl.gz` holds all 7,983 turns. Dataset
id `coqa` is the 1,814 turns retained by `metadata.gold_mb_tokens >= 5`; dataset id
`coqa_full` is the whole file. The cohort is *derived* by that filter rather than stored
a second time. Calls and Chars are reported for the 1,814 cohort only.

**Why the filter exists.** Nearly half the gold answers in the full development split are
a single word -- 3,695 of 7,983 -- and 18.4% are a bare yes or no: 1,472 of 7,983, with
1,538 turns answering exactly `yes`, `no` or `unknown` after lowercasing and stripping
whitespace. A one-token continuation
gives the token game a single player: its best response depends on the fixed prompt, with
no other generated tokens to influence or respond to. The threshold focuses the evaluation
on answers with several token players whose preferences can change as other players move.
It is an experimental design choice, not an algorithmic requirement, every system is
evaluated on the same retained questions, and the unfiltered results are reported beside
them.

**Multi-turn, and the longest prompt of the three benchmarks.** 510.9 ModernBERT tokens
on average, of which 155.7 is dialogue history plus the current question. The context
limit is 1,024 tokens with one margin token reserved; a longer prompt is truncated from
the left, preserving the question, which sits at the end (Appendix E.5). Under the
ModernBERT tokenizer that applies to 9 of the 1,814 prompts. RoBERTa-Large has 514
position embeddings with a padding offset of two, giving it an effective limit of 512
tokens.

**Four references.** For CoQA the official leave-one-annotator-out accumulation is
computed as well as best-reference F1, on the 1,814-turn cohort. Both are stored for
every system; the tables report the best-reference column, and the official score
preserves the ordering of the systems.

**Answers are spans.** About four in five gold answers are a contiguous context span
after normalization (Appendix D.6 / experiment 10), so this benchmark rewards locating
a span more than composing one.

## Budgets by tokenizer family

Because the budget is `|tok(" " + gold)|` in each model's own tokenizer, models
sharing a vocabulary get identical budgets:

| mean T | systems |
|---:|---|
| 8.249 | Danube3-4B |
| 7.628 | SmolLM2-135M, SmolLM2-360M |
| 7.509 | ModernBERT-Large, Ettin-400m, Pythia x3, OLMo-1.7 |
| 7.482 | Qwen2.5 |
| 7.362 | GPT-2 x4, OPT-350M, RoBERTa-Large |
| 7.325 | BLOOM-3B |
| 7.274 | mmBERT-base |

## Data

`data/coqa_paperprompt.jsonl.gz` -- 7,983 rows, read as `coqa` (1,814) and as `coqa_full`
(7,983). The soft-instruction CoQA prompt is in `data/coqa_grounded.jsonl.gz`, which
only experiment 8 reads.
