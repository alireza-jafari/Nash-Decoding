#!/usr/bin/env python3
"""One-shot masked-LM decoding on CLAPNQ (Table 9, one-shot block).

    python experiments/04_qa_benchmark/clapnq/run_oneshot_clapnq.py --model modernbert_large
    python experiments/04_qa_benchmark/clapnq/run_oneshot_clapnq.py --all

The lower-cost non-autoregressive baseline of the paper: the same frozen masked
encoders as the Nash rows, the same canvas of T = B [MASK] slots appended to the
same prompt, the same bans, precision and context -- but every slot is filled from
a **single forward pass**, by the argmax over the unbanned logits at all T positions
at once, and nothing is revised. Calls is therefore exactly 1.0 for every row.

Under the same evaluation protocol, these baselines remain below the best
Nash-decoding answers across all three benchmarks (Appendix F).
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

OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "oneshot"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="a masked encoder tag, e.g. modernbert_large")
    parser.add_argument("--all", action="store_true",
                        help="run the three backbones Table 9 reports")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N examples, for a smoke test")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    arguments = parser.parse_args()
    if not (arguments.all or arguments.model):
        parser.error("give --model TAG or --all")

    examples = datasets.load_dataset(DATASET_PATH, dataset=DATASET)
    if arguments.limit:
        examples = examples[: arguments.limit]
    if arguments.all:
        models = registry.models_for(DATASET, "oneshot")
    else:
        models = (registry.spec(arguments.model),)

    for model in models:
        runner.run_one_shot(
            model,
            examples,
            Path(arguments.output_root) / model.tag,
            device=f"cuda:{arguments.gpu}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
