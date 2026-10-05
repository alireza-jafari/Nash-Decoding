# 6. Prompt-format ablation

**Paper:** Table 10. **Needs a GPU:** no.

```bash
python experiments/06_prompt_ablation/score_prompt_ablation.py
```

## What it shows

The prompt used throughout the paper adds one instruction line between the question
and the answer marker:

```
{evidence} Question: {question}
This question is answered completely with evidence from the {passage|abstract}.
Answer:
```

The ablation deletes exactly that line and changes nothing else:

```
{evidence} Question: {question} Answer:
```

Same evidence, same question, same oracle budget, same banned tokens, same decoding,
same references, paired example by example. Table 10 reports the CLAPNQ rows for
Falcon-7B under autoregressive decoding and ModernBERT-Large under Nash decoding:

| system | prompt | F1 | R-L | R-Ls |
|---|---|---:|---:|---:|
| Falcon-7B | Q/A | 32.87 | 29.80 | 32.61 |
| Falcon-7B | soft | **34.39** | **32.91** | **34.50** |
| ModernBERT-Large | Q/A | 37.00 | 33.25 | 36.39 |
| ModernBERT-Large | soft | **38.04** | **34.79** | **37.42** |

## Beyond the four rows of Table 10

The script scores every system that was run under both prompts, on CLAPNQ and on
PubMedQA, and reports a paired bootstrap interval over questions for each difference.
`results/prompt_ablation.json` holds all of it.

The two arms are paired example by example: the two prompt files have the same ids,
targets, references and `source_text`, and differ only in the prompt.

## Running the Q/A-prompt arm

The outputs of this arm come from the same run scripts as the result tables, with one
flag:

```bash
python experiments/04_qa_benchmark/clapnq/run_nash_clapnq.py --model modernbert_large --prompt qa --gpu 0
python experiments/04_qa_benchmark/clapnq/run_ar_clapnq.py --model falcon_7b --prompt qa --gpu 0
```

and likewise for PubMedQA. `--prompt qa` reads the Q/A file and writes to its own run
directory, `runs/qaprompt_nash/` or `runs/qaprompt_autoregressive/`.

## Data

`data/clapnq_qa_prompt.jsonl` and `data/pubmedqa_qa_prompt.jsonl`.

## Results

`results/prompt_ablation.json` -- both arms, all six systems, both datasets, with the
paired differences and their intervals.
