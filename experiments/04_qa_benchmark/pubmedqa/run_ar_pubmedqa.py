#!/usr/bin/env python3
"""Autoregressive greedy decoding on PubMedQA (Table 8, autoregressive block).

    python experiments/04_qa_benchmark/pubmedqa/run_ar_pubmedqa.py --model gpt2_large
    python experiments/04_qa_benchmark/pubmedqa/run_ar_pubmedqa.py --all --gpu 0

Each model emits exactly B = |tok(' ' + gold)| tokens in its own tokenizer,
with end-of-sequence banned so nothing stops early. One forward pass per
generated token.

PubMedQA is the one dataset whose instruction says `abstract` rather than
`passage`, because its evidence is a biomedical abstract. That single word is the
only difference between the three prompt templates.

The task is to generate the conclusion that was withheld from the input abstract,
not to classify the question. The corpus also labels each question yes / no /
maybe (276 / 169 / 55 here) and that label is never used as a target.

PubMedQA has the greatest lexical novelty of the three: two fifths of reference
tokens and more than three quarters of reference bigrams do not appear in the
abstract, so the model must infer the conclusion rather than extract or repeat text
from the input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import datasets, registry, runner  # noqa: E402

DATASET = "pubmedqa"
DATASET_PATH = REPOSITORY_ROOT / "data" / "pubmedqa_softinstr.jsonl"
EXPECTED_ROWS = 500

OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "autoregressive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="a single model tag, e.g. gpt2_large")
    parser.add_argument("--all", action="store_true", help="run every pubmedqa baseline")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N examples, for a smoke test")
    parser.add_argument("--prompt", choices=("soft", "qa"), default="soft",
                        help="soft = the soft-instruction prompt of the result tables; "
                             "qa = the Question:/Answer: prompt of the Table 10 ablation, "
                             "which omits the instruction sentence")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    arguments = parser.parse_args()
    if not arguments.all and not arguments.model:
        parser.error("give --model TAG or --all")

    if arguments.prompt == "qa":
        # The prompt-format ablation of Table 10 (experiment 06): the same questions
        # with the instruction line deleted, written to their own run directory.
        examples = datasets.load_dataset(
            datasets.DATA_ROOT / datasets.QA_PROMPT_FILE[DATASET],
            dataset=DATASET, prompt_style="qa_prompt",
        )
    else:
        examples = datasets.load_dataset(DATASET_PATH, dataset=DATASET)
    if arguments.limit:
        examples = examples[: arguments.limit]
    models = (
        registry.models_for(DATASET, "autoregressive")
        if arguments.all
        else (registry.spec(arguments.model),)
    )
    output_root = Path(arguments.output_root)
    if arguments.prompt == "qa" and output_root == OUTPUT_ROOT:
        output_root = OUTPUT_ROOT.parent / "qaprompt_autoregressive"
    for model in models:
        runner.run_autoregressive(
            model,
            examples,
            output_root / model.tag,
            device=f"cuda:{arguments.gpu}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
