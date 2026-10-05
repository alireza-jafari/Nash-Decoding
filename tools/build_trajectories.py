#!/usr/bin/env python3
"""Build `artifacts/trajectories/<dataset>_<construction>.json` from a decoding run.

    RUN=experiments/04_qa_benchmark/coqa/runs/nash/coqa/modernbert_large_all_mask
    python tools/build_trajectories.py --run $RUN --dataset coqa \
        --tag allmask --out artifacts/trajectories/coqa_all_mask.json

The trajectory files record, for every example, the Nash gap and the answer quality
of the *partially decoded* canvas after every update. The decoder stores the raw
material for that (the gap at each step and the ordered list of updates) but not the
per-step scores, because scoring needs the tokenizer and the references. This script
is the step between the two.

No GPU and no model weights: a tokenizer is loaded, and only to turn token ids back
into text.

WHAT THE INPUT LOOKS LIKE
-------------------------
`--run` is either a directory of per-example JSON files (the decoder's `work/res/`)
or a single JSON file holding a list of those same records. Only these keys are read:

    slots        int    number of answer positions on the canvas
    status       str    "equilibrium", "cycle" or "cap"
    traj_gap     [float] the gap G(x^(k)), one entry per state the decoder scored
    traj_upd     [[position, token_id]]  every update, in the order applied

Records with `slots == 0` are skipped.
`token_ids` and `prompt_ids` are NOT used: the final answer is the last state of the
replay, and the prompt plays no part in scoring.

HOW EACH OUTPUT FIELD IS DERIVED
--------------------------------
    id, slots, status   copied through unchanged.
    tag                 the --tag value: "allmask" or "l2r".
    gap                 `traj_gap`, copied through unchanged.
    f1, rougeL          one score per canvas state, computed by the replay below.
                        State 0 is the all-[MASK] canvas; state k is the canvas after
                        the first k entries of `traj_upd`. So len == len(traj_upd) + 1.
    n_states            len(f1), i.e. len(traj_upd) + 1.
    gap_offset          n_states - len(gap).
                        This is 0 for an all-mask run that reached equilibrium (the
                        decoder appends the terminal gap of 0.0, so traj_gap has one
                        entry per state); 1 for a run that hit a limit cycle or the
                        update cap (no terminal entry is appended); and `slots` for an
                        L2R run, whose gap is only defined once the canvas is full.
                        Consumers read gap[k] for k < len(gap) and treat the rest as
                        0.0, which is what `analyse_trajectories.py` does.

HOW A PARTIAL CANVAS IS SCORED
------------------------------
The replay starts from `[None] * slots` -- every slot still holding [MASK] -- and
applies `traj_upd` one entry at a time. To score a state, the slots that are still
[MASK] are DROPPED, and the token ids of the written slots are decoded together, in
canvas position order. The mask token is never decoded and contributes no text, so a
half-written answer is scored as the short string it actually spells and state 0 scores
0.0 rather than being undefined. Revisions need no special handling: a second update to
the same position simply overwrites it.

Both metrics are the maximum over the reference list:

    f1      SQuAD token F1 from `nashlib.metrics`, the same implementation the
            result tables use.
    rougeL  the `rouge_score` package's ROUGE-L F-measure with **use_stemmer=False**,
            on the raw decoded string -- not the local normalisation above, and not
            the stemmed variant.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import datasets, metrics, registry  # noqa: E402

DEFAULT_TOKENIZER = registry.spec("modernbert_large").hf_id


def canvas_states(record: dict):
    """Yield the canvas after 0, 1, ... updates, each as a list of ids and Nones."""
    canvas = [None] * record["slots"]
    yield list(canvas)
    for position, token_id in record["traj_upd"]:
        canvas[position] = token_id
        yield list(canvas)


def load_run(path: str) -> list[dict]:
    target = Path(path)
    if target.is_dir():
        files = sorted(glob.glob(str(target / "*.json")))
        records = [json.loads(Path(name).read_text()) for name in files]
    else:
        records = json.loads(target.read_text())
    return [record for record in records if record.get("slots", 0) > 0]


def build(records: list[dict], references: dict[str, list[str]], tag: str,
          tokenizer, rouge) -> list[dict]:
    built = []
    for record in records:
        answers = references[record["id"]]
        f1_curve, rouge_curve = [], []
        for canvas in canvas_states(record):
            written = [token for token in canvas if token is not None]
            text = tokenizer.decode(written) if written else ""
            f1_curve.append(metrics.best_reference_f1(text, answers))
            rouge_curve.append(
                max(rouge.score(answer, text)["rougeL"].fmeasure for answer in answers)
            )
        built.append({
            "id": record["id"],
            "tag": tag,
            "slots": record["slots"],
            "status": record["status"],
            "n_states": len(f1_curve),
            "gap": record["traj_gap"],
            "gap_offset": len(f1_curve) - len(record["traj_gap"]),
            "f1": f1_curve,
            "rougeL": rouge_curve,
        })
    built.sort(key=lambda row: row["id"])
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a trajectory file from a decoding run.")
    parser.add_argument("--run", required=True,
                        help="decoder output: a work/res directory, or a JSON list of its records")
    parser.add_argument("--dataset", required=True, choices=registry.SCORED_SETS,
                        help="which evaluation set the run decoded, for its references")
    parser.add_argument("--tag", default="allmask", choices=("allmask", "l2r"))
    parser.add_argument("--out", required=True, help="where to write the trajectory file")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    arguments = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from rouge_score import rouge_scorer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    records = load_run(arguments.run)
    references = datasets.references_by_id(datasets.load_evaluation(arguments.dataset))
    missing = [record["id"] for record in records if record["id"] not in references]
    if missing:
        print(f"{len(missing)} run records have no reference, first {missing[0]}")
        return 1
    built = build(records, references, arguments.tag, tokenizer, rouge)

    Path(arguments.out).write_text(json.dumps(built))
    statuses = collections.Counter(row["status"] for row in built)
    endpoint = 100 * sum(row["f1"][-1] for row in built) / len(built)
    print(f"wrote {arguments.out}: {len(built)} records, "
          f"endpoint mean F1 {endpoint:.4f}, "
          f"max n_states {max(row['n_states'] for row in built)}, {dict(statuses)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
