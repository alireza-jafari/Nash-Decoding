#!/usr/bin/env python3
"""Lemma 1: greedy autoregressive decoding can be exponentially less probable.

    python experiments/02_separation_lemma/compute_separation.py

No model and no GPU. The lemma is a statement about a distribution that is written
down explicitly, so this script builds that distribution, enumerates {0,1}^n, and
computes each quantity the lemma states:

    the greedy output is (0,1,...,0,1) and is not an equilibrium;
    the unique Nash equilibrium is (1,1,...,1,1);
    they differ at exactly n/2 positions;
    p(nash) / p(ar) = (539/289)^{n/2} = exp(gamma n) with gamma ~ 0.31164;
    the Nash gap at the greedy output exceeds 0.30.

The block distribution is p(a, b) = p1(a) p2(b | a) on {0,1}^2 with

    p1(0) = 0.51   p1(1) = 0.49
    p2(1 | 0) = 0.51   p2(1 | 1) = 0.99

and the full distribution is n/2 independent copies of it. Greedy picks a = 0
because 0.51 > 0.49, then b = 1; but p(0,1) = 0.2601 against p(1,1) = 0.4851, so
once b = 1 is on the table the full-context best response at the first position is
a = 1. Every block pays the same multiplicative price, which is where the
exponential comes from.

Enumeration is exact for the small n it is run at; the closed form is then
evaluated on its own for every even n up to 2,000, where enumeration is impossible.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"

P1 = {0: Fraction(51, 100), 1: Fraction(49, 100)}
P2 = {0: {1: Fraction(51, 100)}, 1: {1: Fraction(99, 100)}}
for a in (0, 1):
    P2[a][0] = 1 - P2[a][1]

GAMMA = 0.5 * math.log(539 / 289)


def block_probability(a: int, b: int) -> Fraction:
    return P1[a] * P2[a][b]


def joint(sequence: tuple[int, ...]) -> Fraction:
    """p(x) for the n/2 independent blocks."""
    probability = Fraction(1)
    for index in range(0, len(sequence), 2):
        probability *= block_probability(sequence[index], sequence[index + 1])
    return probability


def greedy(n: int) -> tuple[int, ...]:
    """Equation 1: argmax over the next token given the prefix only."""
    output: list[int] = []
    for index in range(n):
        if index % 2 == 0:
            output.append(max((0, 1), key=lambda a: P1[a]))
        else:
            previous = output[-1]
            output.append(max((0, 1), key=lambda b: P2[previous][b]))
    return tuple(output)


def token_gap(sequence: tuple[int, ...], position: int) -> Fraction:
    """g_i(x) of Equation 3, computed exactly from the joint distribution."""
    alternatives = {}
    for value in (0, 1):
        candidate = list(sequence)
        candidate[position] = value
        alternatives[value] = joint(tuple(candidate))
    total = alternatives[0] + alternatives[1]
    conditional = {v: alternatives[v] / total for v in (0, 1)}
    return max(conditional.values()) - conditional[sequence[position]]


def nash_gap(sequence: tuple[int, ...]) -> Fraction:
    return max(token_gap(sequence, i) for i in range(len(sequence)))


def enumerate_equilibria(n: int) -> list[tuple[int, ...]]:
    return [
        sequence
        for sequence in itertools.product((0, 1), repeat=n)
        if nash_gap(sequence) == 0
    ]


def enumerate_length(n: int) -> dict:
    """Every quantity of Lemma 1, at one sequence length, by enumeration."""
    ar = greedy(n)
    equilibria = enumerate_equilibria(n)
    nash = tuple([1] * n)
    hamming = sum(a != b for a, b in zip(ar, nash))
    ratio = joint(nash) / joint(ar)
    closed_form = Fraction(539, 289) ** (n // 2)
    gap_at_ar = nash_gap(ar)

    findings = {
        "n": n,
        "greedy": "".join(map(str, ar)),
        "equilibria_found": ["".join(map(str, e)) for e in equilibria],
        "hamming_distance": hamming,
        "p_greedy": float(joint(ar)),
        "p_nash": float(joint(nash)),
        "ratio": float(ratio),
        "ratio_closed_form": float(closed_form),
        "exp_gamma_n": math.exp(GAMMA * n),
        "nash_gap_at_greedy": float(gap_at_ar),
    }
    assertions = [
        ("greedy output is (0,1) repeated", ar == tuple([0, 1] * (n // 2))),
        ("the equilibrium is unique", len(equilibria) == 1),
        ("the unique equilibrium is all ones", equilibria == [nash]),
        ("they differ at exactly n/2 positions", hamming == n // 2),
        ("probability ratio matches (539/289)^(n/2)", ratio == closed_form),
        ("ratio equals exp(gamma n) to 1e-9",
         abs(float(ratio) - math.exp(GAMMA * n)) < 1e-9 * float(ratio)),
        ("Nash gap at the greedy output exceeds 0.30", gap_at_ar > Fraction(3, 10)),
        ("that gap is exactly 125/414", gap_at_ar == Fraction(125, 414)),
    ]
    findings["assertions"] = {name: bool(value) for name, value in assertions}
    findings["all_hold"] = all(value for _, value in assertions)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths", type=int, nargs="*", default=[2, 4, 6, 8, 10, 12],
        help="even sequence lengths to enumerate in full",
    )
    parser.add_argument("--closed-form-up-to", type=int, default=2000)
    arguments = parser.parse_args()

    print(f"block: p(0,1) = {float(block_probability(0, 1)):.4f}   "
          f"p(1,1) = {float(block_probability(1, 1)):.4f}")
    print(f"gamma = 0.5 * log(539/289) = {GAMMA:.5f}\n")
    print(f"{'n':>4} {'greedy':>14} {'equilibrium':>14} {'d_H':>4} "
          f"{'p(nash)/p(ar)':>14} {'exp(gamma n)':>14} {'G(x_ar)':>9}  ok")

    everything_holds = True
    records = []
    for n in arguments.lengths:
        if n % 2:
            raise SystemExit(f"Lemma 1 is stated for even n; got {n}")
        record = enumerate_length(n)
        records.append(record)
        everything_holds &= record["all_hold"]
        found = record["equilibria_found"]
        print(
            f"{n:>4} {record['greedy']:>14} "
            f"{(found[0] if found else '(none)'):>14} "
            f"{record['hamming_distance']:>4} {record['ratio']:>14.4f} "
            f"{record['exp_gamma_n']:>14.4f} {record['nash_gap_at_greedy']:>9.5f}  "
            f"{'yes' if record['all_hold'] else 'NO'}"
        )
        if not record["all_hold"]:
            for name, holds in record["assertions"].items():
                if not holds:
                    print(f"        FAILED: {name}")

    # Enumeration stops being possible long before the lemma stops being true, so
    # the closed form is evaluated on its own for every even n in a much larger range.
    mismatch = None
    for n in range(2, arguments.closed_form_up_to + 1, 2):
        exact = Fraction(539, 289) ** (n // 2)
        # log of a Fraction directly: float(exact) overflows above n ~ 2350.
        logarithm = math.log(exact.numerator) - math.log(exact.denominator)
        if abs(logarithm - GAMMA * n) > 1e-9 * GAMMA * n:
            mismatch = n
            break
    print(
        f"\nclosed form log((539/289)^(n/2)) = gamma n for every even n up to "
        f"{arguments.closed_form_up_to}: "
        + ("yes" if mismatch is None else f"NO, first failure at n={mismatch}")
    )
    everything_holds &= mismatch is None

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "separation.json").write_text(
        json.dumps(
            {"gamma": GAMMA, "closed_form_checked_to": arguments.closed_form_up_to,
             "records": records, "all_hold": everything_holds},
            indent=1,
        )
        + "\n"
    )
    print(f"-> results/separation.json")
    print("\nevery property of Lemma 1 holds at the lengths above" if everything_holds
          else "\na property of Lemma 1 does NOT hold; see above")
    return 0 if everything_holds else 1


if __name__ == "__main__":
    sys.exit(main())
