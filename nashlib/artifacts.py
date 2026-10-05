"""Reading the stored per-example outputs, so every table can be rebuilt on a laptop.

`artifacts/predictions/` holds one gzipped JSONL per (dataset, family, system): one
line per evaluated example, with the generated answer and the per-example telemetry
the tables need (forward passes, seconds, slot count, equilibrium status). They are
the *outputs* of the run scripts, not a summary of them, so every reported number is
recomputed from them rather than copied, and no GPU is needed to check a table.

Naming is `<dataset>__<family>__<system>.jsonl.gz`. `<dataset>` is `coqa`,
`pubmedqa`, `clapnq` or `coqa_full` -- the last being the same systems on all 7,983
CoQA development turns rather than the reported 1,814-turn cohort, which is the
right-hand block of Table 7. Family is one of

    oneshot             a frozen masked encoder filling every slot from one forward pass
    autoregressive      greedy left-to-right at the oracle budget
    nash_maxgap         Nash decoding from the all-[MASK] canvas
    nash_l2r            Nash decoding from left-to-right construction
    coordinate          the shared-checkpoint coordinate-rule comparison
    init                refinement only, from a supplied initial sequence

    qaprompt_*          the same decoders under the prompt ablation's Q/A prompt

`predictions/` is about 10 MB and the whole of `artifacts/` about 15 MB.
`artifacts/wikitext/` carries the update-by-update trajectories of the WikiText
experiment, and `artifacts/trajectories/` carries the gap and F1 curves the
question-answering figures are drawn from, which `tools/build_trajectories.py` builds
from a run.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_ROOT = REPOSITORY_ROOT / "artifacts" / "predictions"

FAMILIES = (
    "oneshot",
    "autoregressive",
    "nash_maxgap",
    "nash_l2r",
    "coordinate",
    "init",
)


def prediction_path(dataset: str, family: str, system: str) -> Path:
    return PREDICTIONS_ROOT / f"{dataset}__{family}__{system}.jsonl.gz"


def has_predictions(dataset: str, family: str, system: str) -> bool:
    return prediction_path(dataset, family, system).exists()


def load_predictions(dataset: str, family: str, system: str) -> list[dict]:
    path = prediction_path(dataset, family, system)
    if not path.exists():
        raise FileNotFoundError(
            f"no stored predictions for {dataset}/{family}/{system} at {path}"
        )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def texts(dataset: str, family: str, system: str) -> dict[str, str]:
    return {row["id"]: row["text"] for row in load_predictions(dataset, family, system)}


def available(dataset: str | None = None, family: str | None = None) -> list[tuple[str, str, str]]:
    """Every stored (dataset, family, system) triple, optionally filtered."""
    found = []
    for path in sorted(PREDICTIONS_ROOT.glob("*.jsonl.gz")):
        parts = path.name[: -len(".jsonl.gz")].split("__")
        if len(parts) != 3:
            continue
        if dataset and parts[0] != dataset:
            continue
        if family and parts[1] != family:
            continue
        found.append(tuple(parts))
    return found


def mean(rows: list[dict], field: str) -> float | None:
    """Mean of a telemetry field, or None when the field was not recorded."""
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def telemetry(rows: list[dict]) -> dict[str, float]:
    """The per-run telemetry the appendix tables print.

    `calls` is the mean number of forward passes per example; for an autoregressive
    model it equals the oracle budget B. `chars` is the mean character length of the
    scored answer.

    A run that did not record `calls` has it filled from `budget`, which is exact for
    a greedy autoregressive model -- the decoder asserts one forward pass per emitted
    token -- but means the value is a restatement of the budget rather than a
    measurement. `calls_measured` says which it is. The 16 CLAPNQ autoregressive files
    are the only ones this applies to.
    """
    summary: dict[str, float] = {
        "chars": sum(len(row["text"]) for row in rows) / len(rows)
    }
    for field in ("calls", "seconds", "slots", "budget", "updates", "final_gap",
                  "unmask_updates", "refine_updates", "calls_to_full"):
        value = mean(rows, field)
        if value is not None:
            summary[field] = value
    statuses = [row.get("status") for row in rows if row.get("status")]
    if statuses:
        # A zero-token budget played no game, so it is excluded from the rate
        # rather than counted as an equilibrium that was reached.
        played = [status for status in statuses if status != "empty"]
        if played:
            summary["equilibrium_pct"] = 100 * played.count("equilibrium") / len(played)
        summary["cycles"] = statuses.count("cycle")
        summary["cap_hits"] = statuses.count("cap")
        summary["empty_budget"] = statuses.count("empty")
    summary["calls_measured"] = "calls" in summary
    if "calls" not in summary and "budget" in summary:
        summary["calls"] = summary["budget"]
    return summary
