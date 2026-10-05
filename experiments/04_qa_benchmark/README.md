# 4. Question-answering benchmark

**Paper:** Table 1 (main body), Tables 7, 8, 9 (appendix). **Needs a GPU:** only to
regenerate the model outputs.

Table 7 is two tables side by side: CoQA scored on the 1,814-turn cohort and on all
7,983 development turns. `score_benchmark.py --all` writes both blocks.

```bash
python experiments/04_qa_benchmark/score_benchmark.py --all    # Tables 7, 8, 9
python experiments/04_qa_benchmark/make_main_table.py          # Table 1
```

Both read `artifacts/predictions/` and compute every metric; neither needs a model
or a GPU. Table 1 is a *selection* from Tables 7-9 and reads the same score files.

## The result

With Nash decoding, masked language models become effective zero-shot question
answerers, outperforming autoregressive models 18× their size in F1, ROUGE-L and
ROUGE-Lsum. The masked encoders are **frozen** — no training, no fine-tuning, no
adapters. The same frozen encoders decoded *one-shot* -- every slot filled from a
single forward pass -- are the lower-cost non-autoregressive baseline, and under the
same protocol they stay below the best Nash-decoding answers on all three datasets.

The strongest backbone varies by dataset — Ettin-400m on CoQA, RoBERTa-Large on
PubMedQA, ModernBERT-Large on CLAPNQ — showing that Nash decoding remains dependent on
the underlying model used for conditional density estimation: better estimation leads
to better performance.

Every system here is evaluated zero-shot, so the comparison isolates decoding methods
with pretrained models under a shared evaluation protocol rather than targeting
state-of-the-art performance; dataset leaderboards typically include models fine-tuned
on task-specific training data.

## The model sets are different on the three datasets

This is the thing to get right before re-running anything.

| dataset | one-shot masked | autoregressive | Nash backbones | rows |
|---|---:|---:|---:|---:|
| CoQA | 3 | **14** | 3 | 20 |
| PubMedQA | 3 | **18** | 3 | 24 |
| CLAPNQ | 3 | **16** | 3 | 22 |

The autoregressive sets are neither equal nor nested. Against CoQA's 14, PubMedQA adds
eight 7B-class models -- Amber, Falcon, Granite, Llama 1, Llama 3.1, both Mistral
versions and OpenLLaMA -- and drops four: GPT-2 Small, GPT-2 Medium, Pythia-410M and
Qwen2.5. CLAPNQ adds four of those same large models -- Amber, Falcon, Llama 1 and
Mistral v0.1 -- plus OLMo-2, and drops three: Danube3-4B, Pythia-6.9B and Qwen2.5. **Eight
autoregressive models appear in all three tables** -- BLOOM-3B, GPT-2 Large, GPT-2 XL,
OLMo-1.7, OPT-350M, Pythia-160M, SmolLM2-135M and SmolLM2-360M. The three masked
encoders -- RoBERTa-Large, ModernBERT-Large and Ettin-400m -- appear in all three as
well, each twice, once decoded one-shot and once under Nash decoding, so 14 of the rows
are common and the rest are not.

mmBERT-base, the fourth masked model Section 4 names, is reported in Table 2 (PubMedQA)
and in the initialization study of experiment 07. BLOOM-3B is
the only checkpoint with no SDPA kernel in `transformers`, so it runs under eager
attention, carried on its `ModelSpec` as `attention="eager"` rather than passed at a
call site.

```bash
python experiments/04_qa_benchmark/coqa/models_coqa.py
python experiments/04_qa_benchmark/pubmedqa/models_pubmedqa.py
python experiments/04_qa_benchmark/clapnq/models_clapnq.py
```

Membership is not written down twice: `nashlib/registry.py` reads it from
`paper/paper_values.json`, the paper's tables in machine-readable form.

## CoQA: the dialogue prompt, and two cohorts

**Prompt.** PubMedQA and CLAPNQ use the soft-instruction prompt and give every system
the same string, byte for byte. CoQA follows Radford et al. (2019) -- the
passage, the dialogue history as `Q:` / `A:` lines carrying the gold answers of the
earlier turns, then the current question and a bare `A:` (Appendix E.1). The masked
encoders receive that string whitespace-normalised, with the line breaks flattened to
single spaces and one leading space prepended (`masked_prompt_for_coqa()` in
`nashlib/datasets.py`).

**Cohorts.** `data/coqa_paperprompt.jsonl.gz` holds all 7,983 development turns. Dataset id
`coqa` is the 1,814 turns whose gold answer has at least five ModernBERT tokens -- Table
1 and the left block of Table 7 -- and dataset id `coqa_full` is all 7,983, the right
block. The cohort is *derived* from the file by `metadata.gold_mb_tokens >= 5`, not
stored a second time. `coqa/README.md` has the rest.

Experiment 8 uses the soft-instruction prompt on CoQA as well, because Appendix H
specifies its own prompt.

## Layout

```
coqa/       models_coqa.py        the exact CoQA set, with pinned revisions
            run_ar_coqa.py        14 autoregressive baselines
            run_nash_coqa.py      Nash decoding, all-mask or left-to-right
            run_oneshot_coqa.py   the one-shot rows: one forward pass per canvas
pubmedqa/   the same four files, with the PubMedQA set
clapnq/     the same four files, with the CLAPNQ set
score_benchmark.py    Tables 7, 8 and 9
make_main_table.py    Table 1
results/    the scores and the rendered tables
```

The run scripts are separate per dataset because the model set, the evidence wording
and the dataset-specific notes differ. The decoding rule, the length budget and the
action set are **not** duplicated: they live in `nashlib/runner.py` and are called by
all of them.

## Re-running

```bash
# one system
python experiments/04_qa_benchmark/clapnq/run_nash_clapnq.py --model modernbert_large --gpu 0

# every autoregressive baseline this dataset reports
python experiments/04_qa_benchmark/coqa/run_ar_coqa.py --all --gpu 0

# left-to-right construction instead (experiment 7's other row)
python experiments/04_qa_benchmark/pubmedqa/run_nash_pubmedqa.py --all --construction l2r --gpu 0

# a 20-example smoke test
python experiments/04_qa_benchmark/coqa/run_nash_coqa.py --model ettin_400m --limit 20 --gpu 0
```

Runs are resumable and can be shared between GPUs; see `environment.md`.

## Reading the Calls column

`Calls` is the mean number of forward passes per example. For an autoregressive model
it equals the oracle budget B, because one forward pass produces one token. For Nash
decoding it is the measured count, and it is large: the gap is re-evaluated at all T
positions after every committed token, so the cost grows with the product of answer
length and update count — 47.7 on CoQA, 1,520 on PubMedQA, 2,967 on CLAPNQ for
ModernBERT-Large. On CoQA the Calls and Chars columns are reported for the 1,814-turn
cohort only, not for the 7,983-turn block beside them.

Model-call counts should be interpreted alongside wall-clock times, because a call
represents different amounts of work across the two model families: an autoregressive
step processes a single token against a cached prefix, while a masked scan re-encodes
the whole prompt and canvas. Experiment 9 reports the seconds.

## Results

```
results/coqa_scores.json      every metric and telemetry value, per system, 1,814 turns
results/coqa_full_scores.json the same systems over all 7,983 turns
results/coqa_table.tex        Table 7, both blocks
results/coqa_table.md         the same, readable
... and the same for pubmedqa and clapnq, which have one cohort each
results/main_table.tex        Table 1
results/main_table.md
```
