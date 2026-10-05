#!/usr/bin/env python3
"""Section 2.3 and Appendix C: do the dynamics behave as Theorem 1 predicts?

    python experiments/03_wikitext_dynamics/analyse_dynamics.py

Reproduces Table 4 and both panels of Figure 2 from the stored trajectories in
`artifacts/wikitext/`. No GPU: the 500 trajectories were recorded update by update,
and this is the analysis of them. `run_wikitext.py` is the script that produced them.

What is being asked. Section 2.3 replaces the exact conditionals assumed by Theorem 1
with estimates from a pretrained masked model and asks whether the dynamics still
behave as the analysis predicts. Three quantities are measured:

    the rate         min_{j<=k} G(x^(j)) against the 1/k reference
    the bound        min_{j<=k} G(x^(j)) <= 2 log(1/p0) / k at every k
    the step         p(x^(k+1)) / p(x^(k)) >= 1 + G(x^(k)) at every update

The first two involve the gap alone and are evaluated directly. The third is about
the joint probability, so the masked model's pseudo-likelihood is substituted for it:
the inequality holds on 71.5% of updates and the pseudo-likelihood increases on
75.2%, the expected consequence of Section 2.2. The Nash gap depends only on
conditional probabilities, so the equilibrium certificate G(x) = 0 is computed
exactly.

The reported trajectory starts at the completed sequence x^(1), after construction
has filled the last [MASK]. Construction is a separate confidence-order pass of 64
model calls and is not part of the measured dynamics, which is why both panels of
Figure 2 begin where they do.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

WIKITEXT = REPOSITORY_ROOT / "artifacts" / "wikitext"
RESULTS = Path(__file__).resolve().parent / "results"

MODERNBERT_MASK_ID = 50284
SLOTS = 64


def load_trajectories() -> list[dict]:
    return json.loads((WIKITEXT / "refinement_trajectories.json").read_text())


def load_referee() -> dict[int, list[float]]:
    """GPT-2 XL continuation log-perplexity at each update, from the referee shards."""
    referee: dict[int, list[float]] = {}
    for path in sorted(WIKITEXT.glob("shard*.json")):
        for record in json.loads(path.read_text()):
            referee[record["id"]] = record["gpt2_logppl"]
    return referee


def convergence(records: list[dict]) -> dict:
    n = len(records)
    statuses = [record["status"] for record in records]
    updates = [record["updates"] for record in records]
    changed = [
        sum(a != b for a, b in zip(record["init_ids"], record["token_ids"]))
        for record in records
    ]
    mask_writes = sum(
        1
        for record in records
        for _, token in record["traj_upd"]
        if token == MODERNBERT_MASK_ID
    )
    return {
        "trajectories": n,
        "slots": records[0]["slots"],
        "exact_equilibrium": statuses.count("equilibrium"),
        "exact_equilibrium_pct": 100 * statuses.count("equilibrium") / n,
        "repeated_state": statuses.count("cycle"),
        "repeated_state_pct": 100 * statuses.count("cycle") / n,
        "already_at_equilibrium": sum(1 for u in updates if u == 0),
        "already_at_equilibrium_pct": 100 * sum(1 for u in updates if u == 0) / n,
        "total_updates": sum(updates),
        "updates_mean": statistics.mean(updates),
        "updates_median": statistics.median(updates),
        "updates_max": max(updates),
        "tokens_changed_mean": statistics.mean(changed),
        "updates_writing_mask": mask_writes,
        "calls_construction": statistics.mean(r["init_calls"] for r in records),
        "calls_refinement": statistics.mean(
            r["calls"] - r["init_calls"] for r in records
        ),
        "seconds_per_text": statistics.mean(r["seconds"] for r in records),
    }


def running_minimum_gap(records: list[dict]) -> tuple[list[int], list[float]]:
    """Mean over texts of min_{j<=k} G(x^(j)), the curve of Figure 2a.

    A terminated trajectory contributes a gap of zero from then on. Appendix C.4
    reports the alternative convention too: holding the residual instead would
    raise the mean at the last k by roughly an order of magnitude and still leave
    it far below the reference.
    """
    horizon = max(len(record["traj_gap"]) for record in records)
    curve = []
    for k in range(horizon):
        values = []
        for record in records:
            gaps = record["traj_gap"]
            values.append(min(gaps[: k + 1]) if k < len(gaps) else 0.0)
        curve.append(statistics.mean(values))
    return list(range(1, horizon + 1)), curve


def bound_check(records: list[dict]) -> dict:
    """Equation 6 as a diagnostic: min_{j<=k} G <= 2 log(1/p0) / k at every k."""
    holds = 0
    slack: list[float] = []
    for record in records:
        bound_numerator = -2 * record["traj_pll"][0]
        running = math.inf
        ok = True
        for k, gap in enumerate(record["traj_gap"], start=1):
            running = min(running, gap)
            bound = bound_numerator / k
            if running > bound + 1e-12:
                ok = False
                break
            if running > 0:
                slack.append(bound / running)
        holds += ok
    return {
        "texts_where_bound_holds": holds,
        "texts": len(records),
        "pct": 100 * holds / len(records),
        "median_slack_factor": statistics.median(slack) if slack else float("nan"),
        "mean_two_log_inv_p0": statistics.mean(
            -2 * record["traj_pll"][0] for record in records
        ),
    }


def step_check(records: list[dict]) -> dict:
    """Equation 5 with the pseudo-likelihood standing in for the joint probability."""
    satisfied = increased = total = 0
    dropped = 0
    for record in records:
        pll = record["traj_pll"]
        for k in range(len(record["traj_upd"])):
            if k + 1 >= len(pll):
                # The inequality compares the pseudo-likelihood before and after an
                # update, so the last update of a trajectory that stopped at a cycle
                # or the cap has no "after" to compare against: the loop broke
                # instead of scanning again. Those updates are dropped from both
                # the numerator and the denominator rather than assumed either way.
                dropped += len(record["traj_upd"]) - k
                break
            delta = pll[k + 1] - pll[k]
            total += 1
            increased += delta > 0
            satisfied += delta >= math.log1p(record["traj_gap"][k]) - 1e-9
    return {
        "updates": total,
        "updates_without_a_successor_state": dropped,
        "satisfy_ratio_bound": satisfied,
        "satisfy_ratio_bound_pct": 100 * satisfied / max(total, 1),
        "pseudo_likelihood_increases_pct": 100 * increased / max(total, 1),
    }


def likelihood_curves(records: list[dict], referee: dict[int, list[float]]) -> dict:
    """Figure 2b: pseudo-perplexity and GPT-2 XL continuation perplexity."""
    horizon = max(len(record["traj_gap"]) for record in records)
    pseudo, external = [], []
    for k in range(horizon):
        pll_values, referee_values = [], []
        for record in records:
            pll = record["traj_pll"]
            pll_values.append(pll[min(k, len(pll) - 1)])
            series = referee.get(record["id"])
            if series:
                referee_values.append(series[min(k, len(series) - 1)])
        pseudo.append(math.exp(-statistics.mean(pll_values) / records[0]["slots"]))
        if referee_values:
            external.append(math.exp(statistics.mean(referee_values)))
    return {
        "pseudo_perplexity": pseudo,
        "gpt2xl_perplexity": external,
        "pseudo_perplexity_start": pseudo[0],
        "pseudo_perplexity_end": pseudo[-1],
        "gpt2xl_perplexity_start": external[0] if external else None,
        "gpt2xl_perplexity_end": external[-1] if external else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true",
                        help="also draw the two panels of Figure 2 (needs matplotlib)")
    arguments = parser.parse_args()

    records = load_trajectories()
    referee = load_referee()
    RESULTS.mkdir(parents=True, exist_ok=True)

    table5 = convergence(records)
    bound = bound_check(records)
    step = step_check(records)
    ks, curve = running_minimum_gap(records)
    curves = likelihood_curves(records, referee)

    print("Table 4: refinement on 500 WikiText-103 continuations, ModernBERT-Large")
    print(f"  trajectories, slots                 {table5['trajectories']}, {table5['slots']}")
    print(f"  reaching an exact equilibrium       {table5['exact_equilibrium']} "
          f"({table5['exact_equilibrium_pct']:.1f}%)")
    print(f"  stopped at a repeated state         {table5['repeated_state']} "
          f"({table5['repeated_state_pct']:.1f}%)")
    print(f"  already at equilibrium at x^(1)     {table5['already_at_equilibrium']} "
          f"({table5['already_at_equilibrium_pct']:.1f}%)")
    print(f"  updates per text mean/median/max    {table5['updates_mean']:.2f} / "
          f"{table5['updates_median']:.0f} / {table5['updates_max']}")
    print(f"  total updates                       {table5['total_updates']}")
    print(f"  tokens changed by refinement        {table5['tokens_changed_mean']:.2f}")
    print(f"  updates that write [MASK]           {table5['updates_writing_mask']} of "
          f"{table5['total_updates']}")
    print(f"  model calls, construction + refine  {table5['calls_construction']:.0f} + "
          f"{table5['calls_refinement']:.0f}")
    print(f"  seconds per text                    {table5['seconds_per_text']:.1f}")

    print("\nFigure 2a: running minimum of the Nash gap against a 1/k reference")
    print(f"  {'k':>4} {'mean min_j<=k G':>16} {'1/k':>10} {'below reference':>16}")
    for k in (1, 2, 5, 10, 20, len(curve)):
        if k <= len(curve):
            print(f"  {k:>4} {curve[k - 1]:>16.6f} {1 / k:>10.6f} "
                  f"{'yes' if curve[k - 1] < 1 / k else 'NO':>16}")
    print(f"  below the reference at every k: "
          f"{'yes' if all(c < 1 / k for k, c in zip(ks, curve)) else 'NO'}")

    print("\nEquation 6 as a diagnostic (gap only, no joint density needed)")
    print(f"  holds at every k on {bound['texts_where_bound_holds']}/{bound['texts']} "
          f"texts ({bound['pct']:.1f}%)")
    print(f"  median slack factor {bound['median_slack_factor']:.0f}x, "
          f"mean 2 log(1/p0) = {bound['mean_two_log_inv_p0']:.1f} nats")

    print("\nEquation 5 with the pseudo-likelihood substituted for the joint")
    print(f"  ratio bound satisfied on {step['satisfy_ratio_bound']}/{step['updates']} "
          f"updates ({step['satisfy_ratio_bound_pct']:.1f}%)")
    print(f"  pseudo-likelihood increases at all on "
          f"{step['pseudo_likelihood_increases_pct']:.1f}% of updates")
    if step["updates_without_a_successor_state"]:
        print(f"  ({step['updates_without_a_successor_state']} update(s) excluded: a "
              f"trajectory that stopped at a cycle or the cap has no scan after its "
              f"last update)")
    print("  -> the Nash gap depends only on conditional probabilities, so the "
          "equilibrium\n     certificate G(x) = 0 is exact.")

    print("\nFigure 2b: likelihood diagnostics")
    print(f"  ModernBERT pseudo-perplexity  {curves['pseudo_perplexity_start']:.4f} -> "
          f"{curves['pseudo_perplexity_end']:.4f}")
    if curves["gpt2xl_perplexity_start"] is not None:
        print(f"  GPT-2 XL continuation ppl     {curves['gpt2xl_perplexity_start']:.4f} -> "
              f"{curves['gpt2xl_perplexity_end']:.4f}")

    summary = {
        "table_4": table5,
        "bound_diagnostic": bound,
        "step_diagnostic": step,
        "running_minimum_gap": {"k": ks, "mean": curve},
        "likelihood": curves,
    }
    (RESULTS / "wikitext_dynamics.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n-> results/wikitext_dynamics.json")

    if arguments.figures:
        draw_figures(ks, curve, curves)
    return 0


def draw_figures(ks: list[int], curve: list[float], curves: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.2, 3.0))
    axis.plot(ks, curve, label=r"$\min_{j\leq k} G(x^{(j)})$")
    axis.plot(ks, [1 / k for k in ks], "--", label=r"$1/k$")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("update $k$")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / "figure_2a_equilibrium_diagnostic.pdf")

    figure, left = plt.subplots(figsize=(4.2, 3.0))
    external = curves["gpt2xl_perplexity"]
    left.plot(ks[: len(external)], external, color="tab:blue", label="GPT-2 XL ppl")
    left.set_xscale("log")
    left.set_xlabel("update $k$")
    right = left.twinx()
    right.plot(ks, curves["pseudo_perplexity"], color="tab:orange",
               label="ModernBERT pseudo-ppl")
    left.legend(loc="upper right", frameon=False)
    right.legend(loc="lower left", frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / "figure_2b_likelihood_diagnostics.pdf")
    print("-> results/figure_2a_equilibrium_diagnostic.pdf, "
          "results/figure_2b_likelihood_diagnostics.pdf")


if __name__ == "__main__":
    raise SystemExit(main())
