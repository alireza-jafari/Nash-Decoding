#!/usr/bin/env python3
"""Appendix G.2: example generations, verbatim.

    python experiments/11_qualitative_examples/select_examples.py
    python experiments/11_qualitative_examples/select_examples.py --percentiles
    python experiments/11_qualitative_examples/select_examples.py --percentiles --dataset coqa -n 6

By default this writes the ten CLAPNQ generations of Appendix G.2 -- ModernBERT-Large
under Nash decoding from the all-[MASK] canvas -- each with its F1 and its rank within
the run. The first is the example discussed in Section 4.2, which the appendix marks
with a dagger.

`--percentiles` selects from any run instead: one example at each evenly spaced
percentile of the run's own F1 distribution, so the list spans the full range of the
run. On CLAPNQ the Section 4.2 example is always included. Selection is deterministic:
sort by F1, index by percentile, no sampling and no seed.

All text is verbatim model output. Nothing is truncated, cleaned or re-cased.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import artifacts, datasets, metrics  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

# The ten examples of Appendix G.2, by question, in the order the appendix lists them.
# All are from the CLAPNQ run of ModernBERT-Large.
PAPER_DATASET, PAPER_SYSTEM = "clapnq", "modernbert_large"
PAPER_EXAMPLES = (
    "where do they shoot guy's grocery games",
    "where do you get an at hop card",
    "where does water go after it enters a storm drain",
    "who wrote capitalism and underdevelopment in latin america",
    "what is the function of insulin and where is it produced",
    "what happens to toby at the end of the office",
    "what was agenda 21 of earth summit of rio de janeiro",
    "which sport awards the oldest trophy in international sports",
    "where do you expect to find a composite cone volcano",
    "who takes photos of planes in the air",
)
# The example discussed in Section 4.2, marked with a dagger in the appendix.
HIGHLIGHTED = {"clapnq": (PAPER_EXAMPLES[0],)}


def scored_run(dataset: str, system: str) -> list[dict]:
    """Every example of one run with its question and F1, best first."""
    rows = datasets.load_dataset(
        datasets.evaluation_path(dataset), dataset=dataset
    )
    by_id = {row["id"]: row for row in rows}
    predictions = artifacts.load_predictions(dataset, "nash_maxgap", system)

    scored = []
    for prediction in predictions:
        source = by_id[prediction["id"]]
        scored.append(
            {
                "id": prediction["id"],
                # Recovered from the prompt the system received; which line holds it
                # depends on the prompt format, so this defers to `nashlib.datasets`.
                "question": datasets.current_question(source, dataset),
                "prediction": prediction["text"],
                "references": source["references"],
                "f1": 100 * metrics.best_reference_f1(
                    prediction["text"], source["references"]
                ),
                "slots": prediction.get("slots"),
                "status": prediction.get("status"),
                "calls": prediction.get("calls"),
            }
        )
    scored.sort(key=lambda item: -item["f1"])
    return scored


def paper_examples() -> list[dict]:
    """The ten examples of Appendix G.2, each with its rank within the run."""
    scored = scored_run(PAPER_DATASET, PAPER_SYSTEM)
    rank_of = {item["question"].strip().lower(): rank for rank, item in enumerate(scored)}
    return [
        dict(
            scored[rank_of[question]],
            rank=rank_of[question],
            percentile=rank_of[question] / (len(scored) - 1),
            highlighted=question in HIGHLIGHTED[PAPER_DATASET],
        )
        for question in PAPER_EXAMPLES
    ]


def percentile_examples(dataset: str, system: str, count: int) -> list[dict]:
    """`count` examples at evenly spaced percentiles of the run's F1 distribution."""
    scored = scored_run(dataset, system)
    highlighted = [
        item
        for item in scored
        if item["question"].lower() in HIGHLIGHTED.get(dataset, ())
    ]
    for item in highlighted:
        item["highlighted"] = True

    remaining = count - len(highlighted)
    chosen = list(highlighted)
    if remaining > 0:
        taken = {item["id"] for item in highlighted}
        for step in range(remaining):
            # Evenly spaced percentiles of the F1 distribution, best to worst.
            index = round(step * (len(scored) - 1) / max(remaining - 1, 1))
            # If that example is already taken, walk outward rather than only
            # forward, so the end of the list cannot yield a duplicate.
            if scored[index]["id"] in taken:
                candidates = sorted(
                    (i for i in range(len(scored)) if scored[i]["id"] not in taken),
                    key=lambda i: abs(i - index),
                )
                if not candidates:
                    break
                index = candidates[0]
            chosen.append(dict(scored[index], highlighted=False))
            taken.add(scored[index]["id"])
    chosen.sort(key=lambda item: -item["f1"])
    return chosen


def render(dataset: str, system: str, examples: list[dict], selection: str) -> str:
    lines = [
        f"# Example generations on {dataset.upper()}",
        "",
        f"Nash decoding with `{system}` from the all-[MASK] canvas. {selection}",
        "",
    ]
    for number, item in enumerate(examples, start=1):
        marker = "+ " if item.get("highlighted") else ""
        lines += [
            f"### {marker}Example {number} - F1 {item['f1']:.1f}",
            "",
            f"**Question** {item['question']}",
            "",
            f"**Nash decoding** {item['prediction']}",
            "",
            f"**Reference** {item['references'][0]}",
            "",
        ]
        if len(item["references"]) > 1:
            lines += [
                f"*({len(item['references'])} references; the one above is the first. "
                "F1 is scored against the most favorable.)*",
                "",
            ]
        lines.append(
            f"<sub>{item['slots']} answer slots, {item['calls']} forward passes, "
            f"terminated at {item['status']}</sub>"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--percentiles", action="store_true",
                        help="select at evenly spaced percentiles of a run's F1 "
                             "distribution instead of the examples of Appendix G.2")
    parser.add_argument("--dataset", default=None, choices=("coqa", "pubmedqa", "clapnq"),
                        help="with --percentiles: the run to select from (default clapnq)")
    parser.add_argument("--system", default=None,
                        help="with --percentiles: the backbone (default modernbert_large)")
    parser.add_argument("-n", "--count", type=int, default=10,
                        help="with --percentiles: how many examples")
    arguments = parser.parse_args()

    if arguments.percentiles:
        dataset = arguments.dataset or PAPER_DATASET
        system = arguments.system or PAPER_SYSTEM
        examples = percentile_examples(dataset, system, arguments.count)
        stem = f"{dataset}_{system}_examples"
        selection = (
            "Examples are taken one at each evenly spaced percentile of this run's F1 "
            "distribution"
            + ("; the example discussed in Section 4.2 is marked +."
               if any(item.get("highlighted") for item in examples) else ".")
        )
    else:
        if arguments.dataset not in (None, PAPER_DATASET) or arguments.system not in (
            None, PAPER_SYSTEM
        ):
            parser.error(
                f"the examples of Appendix G.2 are from {PAPER_DATASET}/{PAPER_SYSTEM}; "
                "pass --percentiles to select from another run"
            )
        dataset, system = PAPER_DATASET, PAPER_SYSTEM
        examples = paper_examples()
        stem = f"{dataset}_{system}_paper_examples"
        selection = (
            "The ten examples of Appendix G.2; the one discussed in Section 4.2 is "
            "marked +."
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stem}.md").write_text(render(dataset, system, examples, selection))
    (RESULTS / f"{stem}.json").write_text(json.dumps(examples, indent=1) + "\n")

    print(f"{dataset} / {system}: {len(examples)} examples")
    for number, item in enumerate(examples, start=1):
        marker = "+" if item.get("highlighted") else " "
        rank = f"  rank {item['rank']:3d}" if "rank" in item else ""
        print(f"  {marker} {number:2d}  F1 {item['f1']:5.1f}{rank}  {item['question'][:60]}")
    print(f"\n-> results/{stem}.md, results/{stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
