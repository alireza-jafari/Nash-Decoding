#!/usr/bin/env python3
"""One-shot masked-LM decoding on CoQA (Table 7, one-shot block, both cohorts).

    python experiments/04_qa_benchmark/coqa/run_oneshot_coqa.py --model modernbert_large
    python experiments/04_qa_benchmark/coqa/run_oneshot_coqa.py --all --dataset coqa_full

The lower-cost non-autoregressive baseline of the paper: the same frozen masked
encoders as the Nash rows, the same canvas of T = B [MASK] slots appended to the
same whitespace-normalised dialogue prompt, the same bans, precision and context --
but every slot is filled from a **single forward pass**, by the argmax over the
unbanned logits at all T positions at once, and nothing is revised. Calls is
therefore exactly 1.0 for every row.

`--dataset coqa` evaluates the 1,814-turn cohort of Table 1 and Table 7's left
block; `--dataset coqa_full` evaluates all 7,983 development turns, its right block.
The full-set run reproduces the cohort run exactly on the 1,814 shared turns, which
is what makes the two blocks one experiment rather than two.
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

OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "oneshot"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="coqa", choices=COHORTS,
                        help="coqa = the 1,814-turn cohort of Table 1 and Table 7's "
                             "left block; coqa_full = all 7,983 dev turns, its right block")
    parser.add_argument("--model", help="a masked encoder tag, e.g. modernbert_large")
    parser.add_argument("--all", action="store_true",
                        help="run the three backbones Table 7 reports")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N examples, for a smoke test")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    arguments = parser.parse_args()
    if not (arguments.all or arguments.model):
        parser.error("give --model TAG or --all")

    examples = datasets.load_dataset(DATASET_PATH, dataset=arguments.dataset)
    if arguments.limit:
        examples = examples[: arguments.limit]
    if arguments.all:
        models = registry.models_for(arguments.dataset, "oneshot")
    else:
        models = (registry.spec(arguments.model),)

    for model in models:
        runner.run_one_shot(
            model,
            examples,
            Path(arguments.output_root) / arguments.dataset / model.tag,
            device=f"cuda:{arguments.gpu}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
