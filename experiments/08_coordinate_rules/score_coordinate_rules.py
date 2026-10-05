#!/usr/bin/env python3
"""Tables 3 and 11: only the coordinate rule changes.

    python experiments/08_coordinate_rules/score_coordinate_rules.py

The comparison in Section 4.3 isolates what Nash decoding adds over masked-diffusion
decoding, by removing everything else. One checkpoint, one prompt, one canvas, one
budget, one action set; three ways to choose which position to write next:

    uniform      a masked position chosen uniformly at random
    confidence   the masked position whose predicted token has the highest
                 conditional probability
    Nash gap     the position with the largest token gap of Equation 3, evaluated
                 at all T positions rather than only the masked ones

The two diffusion rules freeze a position once written, so each performs exactly T
updates and halts. Nash decoding rescans all T positions after every commitment, may
rewrite a position it has already written, and halts only at G(x) <= 0. Sampling is
replaced by argmax in all three, so none of them draws a random token; the uniform
rule is random only in which position it fills next.

The checkpoint is `dllm-hub/ModernBERT-Large-chat-v0.1`: the ModernBERT-Large
architecture, supervised fine-tuned on instruction data for masked-diffusion
generation, so the comparison runs on the checkpoint the diffusion rules were
designed for.

Configuration choices were made on a development portion only: each dataset was split
60/40 by a hash of the example id, and prompt variants and ban levels were compared
on the development portion. The tables report the full sets, and this script reports
both portions. The split itself is stored in
`experiments/08_coordinate_rules/data/dev_test_split.json` rather than recomputed.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import artifacts, datasets, metrics  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
DISPLAY = {"coqa": "CoQA", "pubmedqa": "PubMedQA", "clapnq": "CLAPNQ"}
RULES = (("random", "Uniform"), ("low_confidence", "Confidence"), ("nash", "Nash gap"))
SPLIT_FILE = Path(__file__).resolve().parent / "data" / "dev_test_split.json"


def load_split() -> dict[str, dict[str, set[str]]]:
    raw = json.loads(SPLIT_FILE.read_text())
    return {
        dataset: {part: set(ids) for part, ids in parts.items()}
        for dataset, parts in raw.items()
    }


def paired_test(
    a: dict[str, float], b: dict[str, float], resamples: int = 4000, seed: int = 0
) -> dict:
    """Paired margin a - b with a bootstrap interval and a permutation p-value."""
    shared = sorted(set(a) & set(b))
    differences = [a[i] - b[i] for i in shared]
    generator = random.Random(seed)
    means = sorted(
        100 * statistics.mean(generator.choices(differences, k=len(differences)))
        for _ in range(resamples)
    )
    observed = abs(100 * statistics.mean(differences))
    extreme = sum(
        abs(100 * statistics.mean(
            value if generator.random() < 0.5 else -value for value in differences
        )) >= observed
        for _ in range(resamples)
    )
    return {
        "margin": 100 * statistics.mean(differences),
        "ci": [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]],
        # (extreme + 1) / (resamples + 1): the unbiased form, which cannot
        # report p = 0 from a finite number of permutations.
        "p": (extreme + 1) / (resamples + 1),
        "n": len(shared),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=4000)
    arguments = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    report: dict = {"table_3": {}, "table_11": {}, "splits": {}}

    print("Table 3: coordinate rules on dllm-hub/ModernBERT-Large-chat-v0.1")
    print(f"  {'dataset':9s} {'rule':12s} {'F1':>7} {'R-L':>7} {'R-Ls':>7} "
          f"{'calls':>8} {'cycles':>8}")
    per_example: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in ("coqa", "pubmedqa", "clapnq"):
        source = datasets.load_dataset(
            datasets.evaluation_path(dataset), dataset=dataset
        )
        references = datasets.references_by_id(source)
        report["table_3"][dataset] = {}
        report["table_11"][dataset] = {}
        per_example[dataset] = {}
        for arm, label in RULES:
            rows = artifacts.load_predictions(dataset, "coordinate", arm)
            entry = metrics.score_predictions(
                {row["id"]: row["text"] for row in rows}, references
            )
            telemetry = artifacts.telemetry(rows)
            per_example[dataset][arm] = {
                row["id"]: metrics.best_reference_f1(row["text"], references[row["id"]])
                for row in rows
            }
            cycles = telemetry.get("cycles", 0)
            print(f"  {DISPLAY[dataset]:9s} {label:12s} {entry['F1']:7.2f} "
                  f"{entry['RL']:7.2f} {entry['RLsum']:7.2f} "
                  f"{telemetry['calls']:8.0f} {cycles:8.0f}")
            report["table_3"][dataset][arm] = entry
            report["table_11"][dataset][arm] = {
                "mean_slots": telemetry["slots"],
                "calls": telemetry["calls"],
                "cycles": cycles,
                "cycle_pct": 100 * cycles / entry["n"],
            }
        print()

    print("Paired margin of Nash gap over the stronger diffusion rule, F1:")
    for dataset in per_example:
        stronger = max(
            ("random", "low_confidence"),
            key=lambda arm: statistics.mean(per_example[dataset][arm].values()),
        )
        test = paired_test(
            per_example[dataset]["nash"],
            per_example[dataset][stronger],
            resamples=arguments.resamples,
        )
        print(f"  {DISPLAY[dataset]:9s} vs {stronger:14s} {test['margin']:+6.2f} "
              f"[{test['ci'][0]:+.2f}, {test['ci'][1]:+.2f}]  p = {test['p']:.3f}  "
              f"(n = {test['n']})")
        report.setdefault("paired", {})[dataset] = dict(test, stronger=stronger)

    print("\nDevelopment / test portions, and F1 on each (the tables report the full sets):")
    split = load_split()
    print(f"  {'dataset':9s} {'dev/test':>12s}   "
          + "  ".join(f"{label:>22s}" for _, label in RULES))
    for dataset in per_example:
        parts = split[dataset]
        cells = []
        report["splits"][dataset] = {
            "development": len(parts["dev"]),
            "test": len(parts["test"]),
        }
        for arm, _ in RULES:
            scores = per_example[dataset][arm]
            values = {}
            for part in ("dev", "test"):
                subset = [v for i, v in scores.items() if i in parts[part]]
                values[part] = 100 * statistics.mean(subset)
            report["splits"][dataset][arm] = values
            cells.append(f"{values['dev']:10.2f} {values['test']:10.2f}")
        print(f"  {DISPLAY[dataset]:9s} {len(parts['dev']):5d}/{len(parts['test']):<6d} "
              f"  " + "  ".join(cells))
    print(f"  {'':9s} {'':>12s}   "
          + "  ".join(f"{'dev':>10s} {'test':>10s}" for _ in RULES))

    print("\nTable 11: cost and convergence")
    print(f"  {'dataset':9s} {'mean T':>7} {'diffusion calls':>16} {'Nash calls':>11} "
          f"{'ratio':>7} {'Nash cycles':>12}")
    for dataset, arms in report["table_11"].items():
        diffusion_calls = arms["random"]["calls"]
        nash_calls = arms["nash"]["calls"]
        print(f"  {DISPLAY[dataset]:9s} {arms['nash']['mean_slots']:7.1f} "
              f"{diffusion_calls:16.0f} {nash_calls:11.0f} "
              f"{nash_calls / diffusion_calls:6.1f}x "
              f"{arms['nash']['cycles']:6.0f} ({arms['nash']['cycle_pct']:.1f}%)")

    # How often the Nash arm stops at a repeated state rather than at an equilibrium.
    report["cycle_rates_pct"] = {
        d: report["table_11"][d]["nash"]["cycle_pct"] for d in report["table_11"]
    }

    (RESULTS / "coordinate_rules.json").write_text(json.dumps(report, indent=1) + "\n")
    print("\n-> results/coordinate_rules.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
