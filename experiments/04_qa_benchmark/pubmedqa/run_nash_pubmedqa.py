#!/usr/bin/env python3
"""Nash decoding on PubMedQA (Table 8, Nash block; and the initialization study).

    python experiments/04_qa_benchmark/pubmedqa/run_nash_pubmedqa.py --model modernbert_large
    python experiments/04_qa_benchmark/pubmedqa/run_nash_pubmedqa.py --all --construction l2r

The backbone is frozen: no training, no fine-tuning, no adapters. A canvas of
T = B [MASK] slots is appended to the prompt and the maximum-gap best response
of Algorithm 1 is applied with tolerance 0 until no position can improve.

--construction all_mask is the main-table decoder and starts from the fully
masked canvas; --construction l2r first writes the canvas left to right, one
model call per position with [MASK] excluded, and then refines it under the
identical rule. Those are the two constructions of Table 2 and Appendix G.4.

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

OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "nash"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="a masked encoder tag, e.g. modernbert_large")
    parser.add_argument("--all", action="store_true",
                        help="run the three backbones Table 8 reports")
    parser.add_argument("--basins", action="store_true",
                        help="run all six backbones of the initialization study")
    parser.add_argument("--construction", default="all_mask", choices=("all_mask", "l2r"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="first N examples, for a smoke test")
    parser.add_argument("--prompt", choices=("soft", "qa"), default="soft",
                        help="soft = the soft-instruction prompt of the result tables; "
                             "qa = the Question:/Answer: prompt of the Table 10 ablation, "
                             "which omits the instruction sentence")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    arguments = parser.parse_args()
    if not (arguments.all or arguments.basins or arguments.model):
        parser.error("give --model TAG, --all or --basins")

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
    if arguments.basins:
        models = tuple(registry.spec(tag) for tag in registry.BASIN_BACKBONES)
    elif arguments.all:
        models = registry.models_for(DATASET, "nash")
    else:
        models = (registry.spec(arguments.model),)

    output_root = Path(arguments.output_root)
    if arguments.prompt == "qa" and output_root == OUTPUT_ROOT:
        output_root = OUTPUT_ROOT.parent / "qaprompt_nash"
    for model in models:
        runner.run_nash(
            model,
            examples,
            output_root / f"{model.tag}_{arguments.construction}",
            construction=arguments.construction,
            device=f"cuda:{arguments.gpu}",
            batch_size=arguments.batch_size,
            complete_masked_slots=datasets.COMPLETE_MASKED_SLOTS[DATASET],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
