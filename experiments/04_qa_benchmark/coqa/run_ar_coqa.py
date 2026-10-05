#!/usr/bin/env python3
"""Autoregressive greedy decoding on CoQA (Table 7, autoregressive block).

    python experiments/04_qa_benchmark/coqa/run_ar_coqa.py --model gpt2_large
    python experiments/04_qa_benchmark/coqa/run_ar_coqa.py --all --gpu 0

Each model emits exactly B = |tok(' ' + gold)| tokens in its own tokenizer,
with end-of-sequence banned so nothing stops early. One forward pass per
generated token.

CoQA is multi-turn, and it is the one dataset that does not use the
soft-instruction prompt. It follows Radford et al. (2019): the passage, then the
preceding turns as `Q: {q}` / `A: {a}` lines carrying the gold answers, then the
current question and a bare `A:` (Appendix E.1). The masked encoders receive that
string whitespace-normalised -- dialogue line breaks flattened to spaces, one
leading space (`nashlib.datasets.masked_prompt_for_coqa`). It is the longest prompt
of the three benchmarks; a prompt longer than the context window is truncated from
the left, preserving the question, which sits at the end (Appendix E.5).

`data/coqa_paperprompt.jsonl.gz` holds all 7,983 development turns. `--dataset coqa`
evaluates the 1,814 of them whose gold answer has at least five ModernBERT tokens
(460 conversations, 459 distinct passages), which is what Table 1 and the left
block of Table 7 report; the threshold excludes bare yes/no and other one-word
answers, which would give the token game a single player and nothing to coordinate
with. `--dataset coqa_full` evaluates all 7,983, which is Table 7's right block.
Each question has four reference answers, so CoQA is also the only dataset where
the official leave-one-annotator-out accumulation differs from best-reference F1;
both are computed and stored, and the tables print the best-reference column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import datasets, registry, runner  # noqa: E402

DATASET_PATH = REPOSITORY_ROOT / "data" / "coqa_paperprompt.jsonl.gz"
COHORTS = ("coqa", "coqa_full")

OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "autoregressive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="coqa", choices=COHORTS,
                        help="coqa = the 1,814-turn cohort of Table 1 and Table 7's "
                             "left block; coqa_full = all 7,983 dev turns, its right block")
    parser.add_argument("--model", help="a single model tag, e.g. gpt2_large")
    parser.add_argument("--all", action="store_true", help="run every coqa baseline")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N examples, for a smoke test")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    arguments = parser.parse_args()
    if not arguments.all and not arguments.model:
        parser.error("give --model TAG or --all")

    examples = datasets.load_dataset(DATASET_PATH, dataset=arguments.dataset)
    if arguments.limit:
        examples = examples[: arguments.limit]
    models = (
        registry.models_for(arguments.dataset, "autoregressive")
        if arguments.all
        else (registry.spec(arguments.model),)
    )
    for model in models:
        runner.run_autoregressive(
            model,
            examples,
            Path(arguments.output_root) / arguments.dataset / model.tag,
            device=f"cuda:{arguments.gpu}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
