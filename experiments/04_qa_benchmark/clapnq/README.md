# CLAPNQ

**Table 9.** The 300 answerable development questions, over 297 distinct passages,
with 485 reference answers in total (160 questions have one, 131 have two, 9 have
seven). Mean gold answer 63.8 ModernBERT tokens, up to 231.

```bash
python experiments/04_qa_benchmark/clapnq/models_clapnq.py
python experiments/04_qa_benchmark/clapnq/run_ar_clapnq.py --all --gpu 0
python experiments/04_qa_benchmark/clapnq/run_nash_clapnq.py --all --gpu 0
python experiments/04_qa_benchmark/clapnq/run_oneshot_clapnq.py --all --gpu 0
```

## Models: 3 one-shot masked + 16 autoregressive + 3 Nash backbones

Between the other two in size, and again **not a subset of CoQA's**. CLAPNQ keeps the
whole GPT-2 family (Small through XL) and adds five models CoQA does not have -- Amber,
Falcon, Llama 1, Mistral v0.1 and OLMo-2 -- while dropping three: Danube3-4B,
Pythia-6.9B and Qwen2.5. BLOOM-3B is the only checkpoint with no SDPA kernel in
`transformers` and runs under eager attention; its `ModelSpec` carries that as
`attention="eager"`.

The three masked encoders each appear twice, decoded one-shot and under Nash decoding.
mmBERT-base, the fourth masked model Section 4 names, is run here too, for the
initialization study.

## What makes CLAPNQ different

**Long answers, and the cost that comes with them.** At 63.8 tokens on average this is
by a wide margin the most expensive dataset for Nash decoding: about 2,900 forward
passes per question against 64 for an autoregressive model, because the gap is
re-evaluated at all T positions after every committed token, so the work grows with
the product of answer length and update count. The 231-slot questions cost roughly
138 times what the shortest ones do in forward passes -- 28,906 against 209 -- and about 360 times in seconds, which is why the work queue hands examples out
longest first.

**Composition, not extraction.** 97.6% of reference tokens occur in the passage, but
only 4% of complete answers form a contiguous span, and the longest shared run covers
just over half an answer. Answers typically combine material from two or three
sentences about three and a half positions apart. This is the setting Nash decoding is
strongest in: ModernBERT-Large reaches 38.04 F1, the best result on this dataset among
every system evaluated here.

**Multiple references.** Scoring takes the best reference, and 140 of the 300
questions have more than one.

**Every system receives the same string.** CLAPNQ and PubMedQA both use the
soft-instruction prompt, and the autoregressive and masked systems are all handed it
byte for byte.

**Every answer has exactly T tokens.** If a trajectory stops at a repeated state while
some slots still hold [MASK], those slots are filled before the answer is returned --
left to right, with [MASK] removed from the action set
(`nashlib.datasets.COMPLETE_MASKED_SLOTS`).

**mmBERT-base: pass `--batch-size 16`.** Its vocabulary has 256k entries and the
decoder materializes logits at every position of every sequence in a scan, so the
longest CLAPNQ canvases take a smaller batch than the default of 64.

**The development split.** CLAPNQ pairs a Natural Questions passage with either a
cohesive long-form answer or no answer at all; the 300 *answerable* questions of its
development split are used.

## Where the other CLAPNQ experiments live

CLAPNQ carries three further experiments, because its long answers make the effects
visible: the prompt-format ablation of Table 10 (experiment 6), the cost measurement of
Figure 6 on a fixed 60-item subsample (experiment 9), and the ten example generations
of Appendix G.2 (experiment 11).

## Data

`data/clapnq_softinstr.jsonl`.
