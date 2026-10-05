# PubMedQA

**Table 8.** The official 500-question test half of PQA-L, reconstructed from the
released seed-0 two-fold split. One reference per question. Mean gold answer 48.3
ModernBERT tokens.

```bash
python experiments/04_qa_benchmark/pubmedqa/models_pubmedqa.py
python experiments/04_qa_benchmark/pubmedqa/run_ar_pubmedqa.py --all --gpu 0
python experiments/04_qa_benchmark/pubmedqa/run_nash_pubmedqa.py --all --gpu 0
python experiments/04_qa_benchmark/pubmedqa/run_oneshot_pubmedqa.py --all --gpu 0
```

## Models: 3 one-shot masked + 18 autoregressive + 3 Nash backbones

The widest autoregressive set of the three, and **not a superset of CoQA's**. Against
CoQA it adds the eight 7B-class models -- Amber, Falcon, Granite, Llama 1, Llama 3.1,
Mistral v0.1, Mistral v0.3 and OpenLLaMA -- and drops four small ones: GPT-2 Small,
GPT-2 Medium, Pythia-410M and Qwen2.5. BLOOM-3B is the only checkpoint with no SDPA
kernel in `transformers` and runs under eager attention, which its `ModelSpec` carries
as `attention="eager"`.

The three masked encoders each appear twice, decoded one-shot and under Nash decoding.
mmBERT-base, the fourth masked model Section 4 names, is reported on this dataset only
in Table 2, the initialization study.

## What makes PubMedQA different

**The one word that differs.** Of the two datasets that use the soft-instruction prompt,
this is the one whose instruction line says `abstract` rather than `passage`, because its
evidence is a biomedical abstract. That word is the only difference between the two
templates. (CoQA uses the `Q:` / `A:` dialogue format of Appendix E.1 instead.) As on
CLAPNQ, every system receives the same string byte for byte.

**The task is generation, not classification.** The target is the conclusion withheld
from the input abstract. The corpus also labels each question yes / no / maybe
(276 / 169 / 55 here); that label is never used as a target.

**Greater lexical novelty.** Two fifths of reference tokens and more than three
quarters of reference bigrams are absent from the abstract, so the model must infer
the conclusion rather than extract or repeat text from the input. Meanwhile 70.9% of
*question* tokens occur in the abstract -- the highest of the three, against 59.9% on
CLAPNQ and 53.6% on CoQA -- so the question overlaps its context heavily even when the
conclusion requires considerable new wording.

**RoBERTa-Large is the strongest Nash backbone here** (24.99 F1), ahead of
ModernBERT-Large (24.28) and Ettin-400m (22.77), while Ettin-400m leads on CoQA and
ModernBERT-Large on CLAPNQ: Nash decoding remains dependent on the underlying model
used for conditional density estimation, and better estimation leads to better
performance.

**The soft instruction matters most here.** With the instruction line ModernBERT-Large
scores 10.4 F1 higher than without it; see experiment 6.

## Data

`data/pubmedqa_softinstr.jsonl`.
