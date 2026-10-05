#!/usr/bin/env python3
"""Figures 3 and 5, Appendix G.1: F1 and the Nash gap along the trajectory.

    python experiments/05_decoding_dynamics/analyse_trajectories.py
    python experiments/05_decoding_dynamics/analyse_trajectories.py --figures

Reads `artifacts/trajectories/`, which holds, for every example of every dataset,
the Nash gap G(x^(k)) and the answer F1 of the partially-decoded sequence at every
update k of ModernBERT-Large's all-mask run. No GPU.

The two curves measure different things and that is the point of plotting them
together. F1 measures movement toward the reference answer; the Nash gap measures
the remaining incentive for any player to change its token. Neither is a function
of the other.

The shape that recurs on all three datasets:

    the gap *rises* first. Generation starts from an entirely masked sequence,
    while ModernBERT was pretrained at a 30% masking rate. With little context the
    model is uncertain and every conditional is flat, so the maximum gap is small;
    as early tokens are filled the emerging context sharpens the conditionals and
    the largest available improvement grows. The peak occurs later for
    longer-answer datasets: at k=3 on CoQA, k=10 on PubMedQA and k=21 on CLAPNQ, and
    at those peaks the answer is 45%, 25% and 43% written.

    Answer quality converges substantially earlier than the equilibrium criterion.
    F1 comes within 1% of its final value at k = 10, 55 and 76, while the
    trajectories run to 40, 156 and 240 updates. Most answer quality is obtained
    well before exact convergence; the additional computation primarily removes
    residual token-level instability and certifies equilibrium.

A partially-decoded sequence is scored as it stands, with unwritten slots
contributing nothing, so the F1 curve starts near zero rather than being undefined.
Once a trajectory terminates it holds its final value; the "questions still running"
fraction is reported alongside.

The update index k is 0 at the all-[MASK] canvas, matching the paper, so k is the
number of updates already applied rather than a one-based step counter.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

TRAJECTORIES = REPOSITORY_ROOT / "artifacts" / "trajectories"
PREDICTIONS = REPOSITORY_ROOT / "artifacts" / "predictions"
RESULTS = Path(__file__).resolve().parent / "results"

DATASETS = ("coqa", "pubmedqa", "clapnq")
DISPLAY = {"coqa": "CoQA", "pubmedqa": "PubMedQA", "clapnq": "CLAPNQ"}
# How far the figures are drawn. Trajectories longer than this are truncated, not
# held, so `final_f1` below means "F1 at the end of the plotted window" rather than
# at the end of the longest trajectory. The paper's figures use the same windows.
HORIZON = {"coqa": 50, "pubmedqa": 175, "clapnq": 250}


def load(dataset: str, construction: str = "all_mask") -> list[dict]:
    return json.loads((TRAJECTORIES / f"{dataset}_{construction}.json").read_text())


def curves(records: list[dict], horizon: int) -> dict:
    """Mean gap, mean F1 and the running fraction of examples still updating."""
    gap, f1, active = [], [], []
    for k in range(horizon):
        gap_values, f1_values, running = [], [], 0
        for record in records:
            gaps, scores = record["gap"], record["f1"]
            # After a trajectory stops it contributes a gap of zero. That is exact
            # for the ones that reached equilibrium, which is 1,806 of 1,814 on CoQA
            # and all of PubMedQA and CLAPNQ; for the eight that cycled it replaces a
            # small residual with zero and pulls the tail of the mean down slightly.
            gap_values.append(gaps[k] if k < len(gaps) else 0.0)
            f1_values.append(scores[min(k, len(scores) - 1)])
            running += k < len(gaps) - 1
        gap.append(statistics.mean(gap_values))
        f1.append(100 * statistics.mean(f1_values))
        active.append(running / len(records))
    filled = [
        statistics.mean(min(k, record["slots"]) / record["slots"] for record in records)
        for k in range(horizon)
    ]
    return {
        "k": list(range(horizon)),
        "gap": gap,
        "f1": f1,
        "active": active,
        "filled_fraction": filled,
    }


def revision_share(dataset: str) -> float:
    """Share of updates that rewrite an already-written position, from the run records.

    Computed from the per-example telemetry rather than from the trajectory length,
    because `refine_updates` is recorded directly by the decoder.
    """
    path = PREDICTIONS / f"{dataset}__nash_maxgap__modernbert_large.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    updates = sum(row["updates"] for row in rows)
    revisions = sum(row["refine_updates"] for row in rows)
    return 100 * revisions / max(updates, 1)


def summarize(dataset: str, records: list[dict], curve: dict) -> dict:
    statuses = [record["status"] for record in records]
    gap, f1 = curve["gap"], curve["f1"]
    peak_index = max(range(len(gap)), key=lambda i: gap[i])
    final_f1 = f1[-1]
    within_one_percent = next(
        (k for k, value in enumerate(f1)
         if abs(value - final_f1) <= 0.01 * final_f1),
        None,
    )
    half = len(f1) // 2
    return {
        "dataset": dataset,
        "examples": len(records),
        "converged": statuses.count("equilibrium"),
        "limit_cycles": statuses.count("cycle"),
        "cap_hits": statuses.count("cap"),
        "final_f1": final_f1,
        "mean_gap_start": gap[0],
        "mean_gap_peak": gap[peak_index],
        "peak_k": peak_index,
        "filled_fraction_at_peak_pct": 100 * curve["filled_fraction"][peak_index],
        "peak_increase_pct": 100 * (gap[peak_index] - gap[0]) / gap[0],
        "f1_within_1pct_at_k": within_one_percent,
        "f1_change_over_second_half": f1[-1] - f1[half],
        "active_fraction_at_within_1pct": (
            curve["active"][within_one_percent] if within_one_percent is not None else None
        ),
        "revision_share_pct": revision_share(dataset),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true",
                        help="draw the three panels of Figure 5 (needs matplotlib)")
    arguments = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries, all_curves = [], {}
    for dataset in DATASETS:
        records = load(dataset)
        curve = curves(records, HORIZON[dataset])
        all_curves[dataset] = curve
        summaries.append(summarize(dataset, records, curve))

    print(f"{'dataset':10s} {'n':>6} {'conv':>6} {'cyc':>4} {'final F1':>9} "
          f"{'gap@0':>7} {'peak':>7} {'at k':>5} {'rise':>6} {'filled':>7} "
          f"{'F1 within 1% at k':>18}")
    for s in summaries:
        print(
            f"{DISPLAY[s['dataset']]:10s} {s['examples']:>6} {s['converged']:>6} "
            f"{s['limit_cycles']:>4} {s['final_f1']:>9.2f} {s['mean_gap_start']:>7.2f} "
            f"{s['mean_gap_peak']:>7.2f} {s['peak_k']:>5} "
            f"{s['peak_increase_pct']:>5.0f}% {s['filled_fraction_at_peak_pct']:>6.0f}% "
            f"{str(s['f1_within_1pct_at_k']):>18}"
        )
    print("\nOver the second half of each trajectory F1 changes by at most "
          f"{max(abs(s['f1_change_over_second_half']) for s in summaries):.2f} points.")
    print("At those peaks the answer is on average "
          + ", ".join(f"{s['filled_fraction_at_peak_pct']:.0f}% written on "
                      f"{DISPLAY[s['dataset']]}" for s in summaries)
          + " (counting one slot filled per update).")
    print("Revisions -- updates that rewrite an already-written position -- account for "
          + ", ".join(f"{s['revision_share_pct']:.1f}% on {DISPLAY[s['dataset']]}"
                      for s in summaries) + ".")

    (RESULTS / "dynamics_summary.json").write_text(
        json.dumps({"summaries": summaries, "curves": all_curves}, indent=1) + "\n"
    )
    print("\n-> results/dynamics_summary.json")

    if arguments.figures:
        draw(all_curves, summaries)
    return 0


def draw(all_curves: dict, summaries: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    for axis, summary in zip(axes, summaries):
        curve = all_curves[summary["dataset"]]
        axis.plot(curve["k"], curve["gap"], color="rebeccapurple",
                  label=r"Nash gap $G(x^{(k)})$")
        axis.fill_between(curve["k"], curve["active"], color="0.85", zorder=0,
                          label="questions still running")
        axis.set_ylim(0, 1)
        axis.set_xlabel("update $k$")
        axis.set_title(DISPLAY[summary["dataset"]], fontsize=10)
        right = axis.twinx()
        right.plot(curve["k"], curve["f1"], color="seagreen", label="F1")
        # The paper's panels end the F1 axis at 60 for CoQA and at 40 for the other
        # two; a single limit of 45 would clip the CoQA curve, which ends at 58.
        right.set_ylim(0, 60 if summary["dataset"] == "coqa" else 40)
        right.axhline(summary["final_f1"], color="seagreen", lw=0.6, ls=":")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    figure.tight_layout()
    figure.savefig(RESULTS / "figure_5_dynamics.pdf")
    print("-> results/figure_5_dynamics.pdf")


if __name__ == "__main__":
    raise SystemExit(main())
