#!/usr/bin/env python3
"""Table 2 and Appendix G.4: the equilibrium a trajectory reaches depends on where it starts.

    python experiments/07_initialization_basins/score_initialization.py
    python experiments/07_initialization_basins/score_initialization.py --extra-inits

The token game is a potential game under a true joint density, but a masked
encoder's conditionals are only an estimate of one, and the game they define can
have many equilibria. Two things decide which one a run lands in: the conditional
estimator, which fixes the landscape, and the initialization, which fixes the
trajectory through it. This experiment holds everything else -- prompt, oracle
budget, banned tokens, the max-gap refinement rule, tolerance zero -- and varies
only how the first completed sequence comes into existence.

**All-mask** does not separate construction from refinement at all. The max-gap rule
is applied straight to the all-[MASK] canvas; at an unwritten position the current
token is [MASK], which is a legal action, so writing an empty slot and rewriting a
filled one are scored on the same scale and compete directly. They do interleave:
with ModernBERT-Large, 181 of the 1,814 CoQA trajectories, 276 of the 500 PubMedQA
trajectories and 198 of the 300 CLAPNQ ones revise an already-written position before
the canvas is full.

**Left-to-right** fills the T slots in index order, one model call per position with
[MASK] excluded so every step commits a real token, and revises nothing until all T
slots are full; then the identical refinement runs. This is a masked model *writing*
in left-to-right order, not a simulation of an autoregressive model -- the backbone
is bidirectional and can see how many slots remain, which an autoregressive decoder
never can. Under this construction the interleaving count above is zero by
definition.

**Different initializations, different equilibria.** The two schemes follow different
early decoding trajectories, after which both proceed to equilibrium. The resulting
differences in final performance indicate that the two trajectories can converge to
different equilibria. Although the performance of Nash decoding depends on
initialization, all-mask initialization is often effective, as Table 1 shows.

`--extra-inits` adds refinement-only runs from a supplied sequence: uniformly random
tokens, a one-shot parallel decode of the whole canvas, and the L2R and
confidence-ordered constructions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import artifacts, datasets, metrics, registry  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
DISPLAY = {"coqa": "CoQA", "pubmedqa": "PubMedQA", "clapnq": "CLAPNQ"}
CONSTRUCTIONS = (("l2r", "left-to-right"), ("maxgap", "all-mask"))
# Table 2 in the body is the four largest backbones on PubMedQA; the extension of
# Appendix G.4 is all six backbones on all three datasets.
EXTRA_INITS = ("random", "one_shot", "l2r", "confidence")


def references_for(dataset: str) -> dict[str, list[str]]:
    rows = datasets.load_dataset(
        datasets.evaluation_path(dataset), dataset=dataset
    )
    return datasets.references_by_id(rows)


def score(dataset: str, family: str, system: str, references) -> dict | None:
    if not artifacts.has_predictions(dataset, family, system):
        return None
    predictions = artifacts.load_predictions(dataset, family, system)
    entry = metrics.score_predictions(
        {row["id"]: row["text"] for row in predictions}, references
    )
    entry.update(artifacts.telemetry(predictions))
    entry["interleaved"] = count_interleaved(predictions)
    return entry


def count_interleaved(predictions: list[dict]) -> int:
    """Trajectories that revised a written position before the canvas was full.

    The decoder computes this itself and stores it as `interleaved`, because it is
    the only place that can: it needs to know whether any of the updates *before*
    the canvas filled was a revision, which the summary counters do not say. Older
    records predate the field, so they fall back to the summary test, which agrees
    with the exact one on every stored run.
    """
    total = 0
    for row in predictions:
        if "interleaved" in row:
            total += bool(row["interleaved"])
            continue
        step_full = row.get("step_last_mask_filled", row.get("step_canvas_full"))
        unmasked = row.get("unmask_updates")
        total += bool(
            step_full is not None and unmasked is not None and step_full > unmasked
        )
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-inits", action="store_true",
                        help="also score the random and one-shot starts")
    arguments = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    report: dict = {"all_backbones": {}, "table_2": {}, "extra_inits": {}}

    print("Appendix G.4: left-to-right against all-mask, every backbone, all three datasets")
    header = f"  {'backbone':18s} {'size':>5s} {'construction':>14s}"
    for dataset in registry.DATASETS:
        header += f" | {DISPLAY[dataset]:>22s}"
    print(header)
    print(f"  {'':18s} {'':>5s} {'':>14s}" + " | " + " | ".join(
        f"{'F1':>6} {'R-L':>6} {'R-Ls':>6}" for _ in registry.DATASETS))

    references = {dataset: references_for(dataset) for dataset in registry.DATASETS}
    for order, label in CONSTRUCTIONS:
        report["all_backbones"][order] = {}
        for tag in registry.BASIN_BACKBONES:
            model = registry.spec(tag)
            line = f"  {model.display:18s} {model.parameters:>5s} {label:>14s}"
            entry_per_dataset = {}
            for dataset in registry.DATASETS:
                entry = score(dataset, f"nash_{order}", tag, references[dataset])
                entry_per_dataset[dataset] = entry
                if entry is None:
                    line += f" |    ---    ---    ---"
                else:
                    line += (f" | {entry['F1']:6.2f} {entry['RL']:6.2f} "
                             f"{entry['RLsum']:6.2f}")
            report["all_backbones"][order][tag] = entry_per_dataset
            print(line)
        print()

    # Which construction wins, per backbone and dataset.
    print("  all-mask minus left-to-right, F1:")
    for dataset in registry.DATASETS:
        deltas = []
        for tag in registry.BASIN_BACKBONES:
            left = report["all_backbones"]["l2r"][tag][dataset]
            right = report["all_backbones"]["maxgap"][tag][dataset]
            if left and right:
                deltas.append((registry.spec(tag).display, right["F1"] - left["F1"]))
        print(f"    {DISPLAY[dataset]:9s} "
              + ", ".join(f"{name} {delta:+.2f}" for name, delta in deltas))

    report["table_2"] = {
        order: {
            tag: report["all_backbones"][order][tag]["pubmedqa"]
            for tag in registry.BODY_BASIN_BACKBONES
        }
        for order, _ in CONSTRUCTIONS
    }
    print("\nTable 2 is the PubMedQA column of the four largest backbones, above. "
          "The full table is the extension of Appendix G.4 to every model across the "
          "three datasets.")

    # How often the all-mask rule actually interleaves writing and revising.
    print("\nTrajectories that revise an already-written position before the canvas "
          "is full (ModernBERT-Large, all-mask):")
    for dataset in registry.DATASETS:
        predictions = artifacts.load_predictions(dataset, "nash_maxgap", "modernbert_large")
        interleaved = count_interleaved(predictions)
        print(f"  {DISPLAY[dataset]:9s} {interleaved} of {len(predictions)}")
        report.setdefault("interleaving", {})[dataset] = {
            "interleaved": interleaved, "examples": len(predictions)
        }

    if arguments.extra_inits:
        print("\nMore initializations: refinement only, from a supplied sequence")
        print(f"  {'dataset':9s} {'init':12s} {'F1 initial':>11} {'F1 refined':>11} "
              f"{'updates':>8} {'calls':>8}")
        for dataset in ("coqa", "pubmedqa"):
            for init in EXTRA_INITS:
                if not artifacts.has_predictions(dataset, "init", init):
                    continue
                rows = artifacts.load_predictions(dataset, "init", init)
                refined = metrics.score_predictions(
                    {row["id"]: row["text"] for row in rows}, references[dataset]
                )
                initial = metrics.score_predictions(
                    {row["id"]: row.get("init_text", "") for row in rows},
                    references[dataset],
                )
                telemetry = artifacts.telemetry(rows)
                print(f"  {DISPLAY[dataset]:9s} {init:12s} {initial['F1']:11.2f} "
                      f"{refined['F1']:11.2f} {telemetry.get('updates', 0):8.1f} "
                      f"{telemetry.get('calls', 0):8.0f}")
                report["extra_inits"].setdefault(dataset, {})[init] = {
                    "initial": initial, "refined": refined, "telemetry": telemetry
                }

    (RESULTS / "initialization.json").write_text(json.dumps(report, indent=1) + "\n")
    print("\n-> results/initialization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
