#!/usr/bin/env python3
"""Score every system on one dataset and render that dataset's appendix table.

    python experiments/04_qa_benchmark/score_benchmark.py coqa
    python experiments/04_qa_benchmark/score_benchmark.py --all

Reads the stored per-example predictions in `artifacts/predictions/`, recomputes
token F1, ROUGE-L and ROUGE-Lsum with `nashlib.metrics`, and writes

    results/<dataset>_scores.json    every metric and telemetry value
    results/<dataset>_table.tex      Table 7, 8 or 9
    results/<dataset>_table.md       the same table as readable text

No GPU and no model download: the decoding already happened, and this is the
scoring half of the pipeline. Which systems belong in the table comes from
`nashlib.registry`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import artifacts, datasets, metrics, registry, tables  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
DISPLAY = {"coqa": "CoQA", "coqa_full": "CoQA (all turns)",
           "pubmedqa": "PubMedQA", "clapnq": "CLAPNQ"}
# The Nash rows of Tables 7-9 are the all-mask construction; the left-to-right
# rows are the initialization study of experiment 07.
FAMILY_TO_ARTIFACT = {
    "oneshot": "oneshot",
    "autoregressive": "autoregressive",
    "nash": "nash_maxgap",
}

CAPTION = {
    "coqa": (
        "Results on CoQA under the GPT-2 paper prompt. The left block scores the "
        "1,814 questions whose gold answer has $\\geq 5$ ModernBERT tokens; the "
        "right block scores all 7,983 CoQA development turns, including the "
        "yes/no/unknown answers. Protocol is identical throughout and identical to "
        "Table 1: oracle answer length in each model's own tokenizer, "
        "structural-token bans only, greedy decoding, fp32. Calls is the mean "
        "number of forward passes per example; for autoregressive models it equals "
        "the oracle token budget $B$. Char is the mean character "
        "length of the scored answer. Calls and Char are reported for the "
        "$\\geq 5$ cohort."
    ),
    "pubmedqa": "Results on PubMedQA.",
    "clapnq": "Results on CLAPNQ.",
}


def score_dataset(dataset: str) -> dict[str, dict[str, dict[str, float]]]:
    """Scores keyed by family and then by tag.

    The three masked encoders appear twice in every table, decoded one-shot and under
    Nash decoding, under the same tag; a flat dictionary would let the second family
    silently overwrite the first.
    """
    rows = datasets.load_dataset(
        datasets.evaluation_path(dataset), dataset=dataset
    )
    references = datasets.references_by_id(rows)
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for family, artifact_family in FAMILY_TO_ARTIFACT.items():
        scores[family] = {}
        for model in registry.models_for(dataset, family):
            predictions = artifacts.load_predictions(
                dataset, artifact_family, model.tag
            )
            entry = metrics.score_predictions(
                {row["id"]: row["text"] for row in predictions},
                references,
                official_coqa=(dataset == "coqa"),
            )
            entry.update(artifacts.telemetry(predictions))
            entry["family"] = family
            entry["hf_id"] = model.hf_id
            entry["revision"] = model.revision
            scores[family][model.tag] = entry
    return scores


def _first_entry(scores: dict) -> dict:
    return next(iter(next(iter(scores.values())).values()))


def render(dataset: str, scores: dict[str, dict[str, dict[str, float]]]) -> None:
    blocks = {
        family: registry.models_for(dataset, family)
        for family in ("oneshot", "autoregressive", "nash")
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{dataset}_scores.json").write_text(
        json.dumps(scores, indent=1, sort_keys=True) + "\n"
    )
    if dataset == "coqa_full":
        # The full development set has no table of its own: it is the right block
        # of Table 7, which `coqa` renders from this score file.
        return
    if dataset == "coqa":
        # Table 7 is one row list scored on two cohorts, so it is rendered once,
        # from both score files, rather than as two tables that could disagree
        # about which systems belong in it.
        full = json.loads((RESULTS / "coqa_full_scores.json").read_text())
        (RESULTS / "coqa_table.tex").write_text(
            tables.latex_cohort_pair(
                blocks, scores, full, caption=CAPTION[dataset], label="tab:coqa"
            )
            + "\n"
        )
        (RESULTS / "coqa_table.md").write_text(
            tables.markdown_cohort_pair(
                blocks, scores, full,
                title="CoQA under the GPT-2 paper prompt",
            )
            + "\n"
        )
        return
    (RESULTS / f"{dataset}_table.tex").write_text(
        tables.latex_single_dataset(
            blocks,
            scores,
            caption=CAPTION[dataset],
            label=f"tab:{dataset}",
        )
        + "\n"
    )
    (RESULTS / f"{dataset}_table.md").write_text(
        tables.markdown_single_dataset(
            blocks, scores,
            title=f"{DISPLAY[dataset]} ({_first_entry(scores)['n']:,} examples)",
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", choices=registry.SCORED_SETS)
    parser.add_argument("--all", action="store_true",
                        help="score all three datasets, CoQA on both its cohorts")
    arguments = parser.parse_args()
    if not arguments.all and arguments.dataset is None:
        parser.error("give a dataset name or --all")
    # `coqa_full` is scored before `coqa`, because Table 7 is rendered from both.
    targets = (
        ("coqa_full",) + registry.DATASETS
        if arguments.all
        else (("coqa_full", "coqa") if arguments.dataset == "coqa" else (arguments.dataset,))
    )

    for dataset in targets:
        scores = score_dataset(dataset)
        render(dataset, scores)
        flat = {
            f"{family}/{tag}": entry
            for family, entries in scores.items()
            for tag, entry in entries.items()
        }
        print(f"\n{DISPLAY[dataset]}  ({len(flat)} systems)")
        print(f"  {'system':34s} {'F1':>7} {'R-L':>7} {'R-Ls':>7} {'calls':>8} {'chars':>6}")
        for key, entry in sorted(flat.items(), key=lambda kv: -kv[1]["F1"]):
            print(
                f"  {key:34s} {entry['F1']:7.2f} {entry['RL']:7.2f} {entry['RLsum']:7.2f} "
                f"{entry.get('calls', float('nan')):8.1f} {entry['chars']:6.0f}"
            )
        restated = sorted(
            key for key, entry in flat.items()
            if entry.get("calls_measured") is False
        )
        if restated:
            print(f"  Calls for {len(restated)} autoregressive system(s) is the oracle "
                  f"budget B: greedy autoregressive decoding performs one forward pass "
                  f"per generated token.")
        print(f"  -> results/{dataset}_scores.json, _table.tex, _table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
