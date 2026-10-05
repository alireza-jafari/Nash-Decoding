# 8. From diffusion updates to equilibrium updates

**Paper:** Table 3 (body), Table 11 and Appendix H. **Checkpoint:**
`dllm-hub/ModernBERT-Large-chat-v0.1`. **Needs a GPU:** only to regenerate.

```bash
python experiments/08_coordinate_rules/score_coordinate_rules.py
python experiments/08_coordinate_rules/run_coordinate_rules.py --dataset clapnq --all-rules --gpu 0
```

## What it shows

The closest existing method to Nash decoding is single-token masked-diffusion
decoding. Both start from an all-[MASK] sequence and construct the output through one
full-context token update at a time. They differ in exactly two things: how the
updated position is selected, and whether previous decisions can be revised. This
experiment removes everything else so that those two things are what is being
measured.

| coordinate rule | CoQA F1 | PubMedQA F1 | CLAPNQ F1 |
|---|---:|---:|---:|
| Uniform | 38.65 | 30.50 | 56.15 |
| Confidence | 41.85 | 31.74 | 58.84 |
| **Nash gap** | **43.07** | **32.11** | **59.16** |

Nash decoding is ahead of both diffusion rules on all three benchmarks and all three
metrics, including against the stronger of the two rules, which is confidence order
everywhere. Paired margins in F1 against confidence order: **+1.23** on CoQA
(bootstrap CI [+0.61, +1.82], permutation p = 0.000), **+0.37** on PubMedQA
([+0.09, +0.68], p = 0.008), **+0.32** on CLAPNQ ([-0.39, +1.03], p = 0.386).

Because the backbone is the same for all three arms, the gain is not explained by the
bidirectional backbone.

## Why this checkpoint and not a frozen one

`ModernBERT-Large-chat-v0.1` keeps the ModernBERT-Large architecture and is
supervised fine-tuned on instruction data for masked-diffusion generation, so the
comparison runs on the checkpoint the diffusion rules were designed for. No weights
are modified and a single checkpoint serves every method.

## This experiment keeps the soft-instruction CoQA prompt

The result tables of experiment 4 use the `Q:` / `A:` dialogue prompt of Radford et al.
(2019) on CoQA. This experiment reads `data/coqa_grounded.jsonl.gz` instead, which keeps
the soft-instruction wording plus the appended `Answer using the passage's own words.`
clause, as Appendix H specifies: the comparison is between three coordinate rules on one
checkpoint, and everything outside the rule is held fixed at the values that appendix
names. `nashlib.datasets.load_dataset` takes the prompt shape explicitly
(`prompt_style`), so the two CoQA files cannot be mixed up.

## What is held fixed

**Prompt** -- the instruction of Appendix E with one clause appended,
`Answer using the {passage|abstract}'s own words.`, presented as raw text with no chat
template, no system message and no added special tokens. **Canvas** --
`[CLS] prompt [MASK]xT [SEP]`, T the oracle budget in this checkpoint's tokenizer.
The longest prompt-plus-canvas across the three datasets is 1,288 tokens, well inside
the checkpoint's 8,192-token context, so nothing is truncated. **Action set** -- one
set for every dataset and rule: the four non-mask special ids plus eleven further
tokens (three pieces of the answer terminator, four chat-template markers, four
newline tokens), leaving 50,352 actions, plus [MASK], which only Nash decoding may
play. **Decoding** -- greedy, float32, TF32 off, SDPA.

The two diffusion rules freeze a position once written, so each performs exactly T
updates. Nash decoding rescans all T positions after every commitment, may rewrite a
position it has already written, and halts only at G(x) <= 0.

Configuration choices were made on a development portion only: each dataset is split
60/40 by a hash of the example id (CoQA 1,078/736, PubMedQA 302/198, CLAPNQ 181/119),
and prompt variants and ban levels were compared there. The tables report the full
sets; the script prints both portions. The split is stored in
`experiments/08_coordinate_rules/data/dev_test_split.json`.

## Configuration details

**The eleven banned tokens.** In this checkpoint's vocabulary:

| role | token | id |
|---|---|---:|
| answer terminator | `[/` | 32871 |
| answer terminator | `Answer` | 32869 |
| answer terminator | `]` | 62 |
| chat-template marker | `[` | 60 |
| chat-template marker | `SYS` | 9316 |
| chat-template marker | `Response` | 9604 |
| chat-template marker | `Question` | 23433 |
| newline | `\n` | 187 |
| newline | `\n\n` | 535 |
| newline | ` \n` (a space, then a newline) | 2490 |
| newline | `\n\n\n` | 2756 |

`run_coordinate_rules.py` requires each of the eleven strings to tokenize to exactly
the id in the table.

**The uniform rule's order.** Uniform order is random only in which position it fills
next. The order is a function of the example alone: a generator on the GPU is seeded
with the first 32 bits of the SHA-256 of the example id (`nashlib.diffusion.seed_for`),
and at every step one uniform number is drawn for each position of the whole sequence
-- prompt and special tokens included -- and the still-masked answer slot with the
largest draw is written. Because the draws come from the device's generator, the order
is defined on a CUDA device.

**Slots still masked when Nash decoding stops.** A trajectory that halts at a repeated
state before every slot has been written still has to return T tokens, because the two
diffusion rules always do. Any remaining [MASK] slots are then filled left to right
with [MASK] removed from the action set (`complete_masked_slots=True`), one forward
pass per slot.

## Data

`data/{coqa,pubmedqa,clapnq}_grounded.jsonl` -- the soft-instruction files with the
one extra clause. `experiments/08_coordinate_rules/data/dev_test_split.json` -- the
60/40 id lists.

## Results

`results/coordinate_rules.json` -- Tables 3 and 11, the paired tests and the split
scores.
