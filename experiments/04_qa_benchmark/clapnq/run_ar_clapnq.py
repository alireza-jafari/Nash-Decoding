#!/usr/bin/env python3
"""Autoregressive greedy decoding on CLAPNQ (Table 9, autoregressive block).

    python experiments/04_qa_benchmark/clapnq/run_ar_clapnq.py --model gpt2_large
    python experiments/04_qa_benchmark/clapnq/run_ar_clapnq.py --all --gpu 0

Each model emits exactly B = |tok(' ' + gold)| tokens in its own tokenizer,
with end-of-sequence banned so nothing stops early. One forward pass per
generated token.

CLAPNQ pairs a Natural Questions passage with either a cohesive long-form answer
or no answer at all; the 300 answerable questions of its development split are
used.

It has the longest answers (63.8 ModernBERT tokens on average, up to 231) and
multiple references: 160 questions have one, 131 have two and 9 have seven, a mean
of 1.62. Scoring takes the best reference.

Long answers make this the most expensive dataset for Nash decoding by a wide
margin -- about 2,900 forward passes per question against 64 for an autoregressive
model -- because the gap is re-evaluated at all T positions after every committed
token, so the work grows with the product of answer length and update count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import datasets, registry, runner  # noqa: E402

DATASET = "clapnq"
DATASET_PATH = REPOSITORY_ROOT / "data" / "clapnq_softinstr.jsonl"
EXPECTED_ROWS = 300

OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "autoregressive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="a single model tag, e.g. gpt2_large")
    parser.add_argument("--all", action="store_true", help="run every clapnq baseline")
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
