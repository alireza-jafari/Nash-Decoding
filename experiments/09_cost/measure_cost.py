#!/usr/bin/env python3
"""Figures 4 and 6: what reaching an equilibrium costs.

    python experiments/09_cost/measure_cost.py                  # report stored timings
    python experiments/09_cost/measure_cost.py --measure coqa --gpu 0

Table 1 reports answer quality under reference-derived token budgets. This reports
the associated computational cost for five systems -- ModernBERT-Large under Nash
decoding from all-mask and from left-to-right, and Falcon-7B, GPT-2 Large and
OPT-350M under greedy autoregressive decoding -- measured on B200 hardware under the
same numerical settings.

Model-call counts should be interpreted alongside wall-clock times, because a call
represents different amounts of work across the two model families. After its
initial prompt evaluation, an autoregressive decoder processes each new token using a
cached prefix, whereas the masked implementation re-encodes the prompt-and-canvas
sequence for its conditional evaluations. This overhead is primarily architectural
rather than inherent to the equilibrium objective.

The Nash bars are split at the update that fills the last [MASK]: before it, the
work is mostly construction; after it, refinement of a full canvas. Left-to-right
inverts that split -- it fills the canvas in T calls and then spends most of its
budget refining -- and needs 2.4x to 11x fewer calls than all-mask for 1.2 to 2.4 F1.

Without `--measure` this reports three things: the B200 figures of Figure 6, read
from `paper/paper_values.json`; the forward-pass counts recorded by the evaluation
runs themselves on the full evaluation sets, computed from `artifacts/predictions/`;
and the CoQA and CLAPNQ timing measurements in `artifacts/timing/`. With
`--measure DATASET` it runs the five systems on the current device and reports
seconds and model calls per question for that device.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import artifacts, datasets, numerics, registry  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
TIMING = REPOSITORY_ROOT / "artifacts" / "timing"
# The datasets with a raw timing file.
TIMING_FILE = {
    "coqa": TIMING / "timing_coqa_numbers.json",
    "clapnq": TIMING / "clapnq_timing_b200.json",
}
DISPLAY = {"coqa": "CoQA", "pubmedqa": "PubMedQA", "clapnq": "CLAPNQ"}

# The five systems of Figure 6, in the order the figure draws them. Defined in the
# registry so this file and anything else that needs the list agree by construction.
SYSTEMS = registry.COST_SYSTEMS


def _figure_6() -> dict[str, dict[str, list]]:
    """Seconds and calls per question as Figure 6 draws them, in the order of `SYSTEMS`."""
    block = json.loads(
        (REPOSITORY_ROOT / "paper" / "paper_values.json").read_text()
    )["figure_4_6_cost"]
    return {
        dataset: {"seconds": block[dataset]["seconds"], "calls": block[dataset]["calls"]}
        for dataset in ("coqa", "pubmedqa", "clapnq")
    }


FIGURE_6 = _figure_6()
CLAPNQ_SUBSAMPLE = REPOSITORY_ROOT / "data" / "clapnq_sub.jsonl"


def timing_measurements(dataset: str) -> list[dict] | None:
    """The raw timing file for one dataset, normalised into the figure's row order.

    The two files on disk have different shapes -- CoQA records milliseconds split
    into construction and refinement, CLAPNQ records seconds and a `to_full` pair --
    so this puts both into one form, `{seconds, calls}` per system in the order the
    figure draws them, plus the split for the two Nash rows. `None` when the dataset
    has no file, which is PubMedQA.
    """
    path = TIMING_FILE.get(dataset)
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text())
    rows = []
    for tag, family, label in SYSTEMS:
        if dataset == "coqa":
            key = ("nash" if family == "nash_maxgap"
                   else "nash_l2r" if family == "nash_l2r" else tag)
            entry = raw[key]
            nash = family != "autoregressive"
            rows.append({
                "label": label,
                "seconds": (entry["t1"] + entry["t2"]) / 1000.0,
                "calls": entry["c1"] + entry["c2"],
                "seconds_to_full": entry["t1"] / 1000.0 if nash else None,
                "calls_to_full": entry["c1"] if nash else None,
                "n": entry["n"],
            })
        else:
            key = (f"Nash/{tag}_maxgap" if family == "nash_maxgap"
                   else f"Nash/{tag}_l2r" if family == "nash_l2r" else f"AR/{tag}")
            entry = raw[key]
            rows.append({
                "label": label,
                "seconds": entry["seconds"],
                "calls": entry["calls"],
                "seconds_to_full": entry.get("seconds_to_full"),
                "calls_to_full": entry.get("calls_to_full"),
                "n": entry["n"],
            })
    return rows


def stored_costs(dataset: str) -> list[dict]:
    """Calls and per-example seconds as recorded by the runs themselves.

    These come from the evaluation runs on the full evaluation sets; the call counts
    are hardware-independent, and the timing measurements give the seconds.
    """
    report = []
    for tag, family, label in SYSTEMS:
        if not artifacts.has_predictions(dataset, family, tag):
            report.append({"label": label, "calls": None, "seconds": None})
            continue
        rows = artifacts.load_predictions(dataset, family, tag)
        telemetry = artifacts.telemetry(rows)
        entry = {
            "label": label,
            "tag": tag,
            "family": family,
            "calls": telemetry.get("calls"),
            "seconds": telemetry.get("seconds"),
            "calls_to_canvas_full": telemetry.get("calls_to_full"),
        }
        if entry["calls_to_canvas_full"] and entry["calls"]:
            entry["calls_after_canvas_full"] = entry["calls"] - entry["calls_to_canvas_full"]
        report.append(entry)
    return report


def measure(dataset: str, device: str, limit: int, warmup: int) -> list[dict]:
    """Re-measure the five systems here, one at a time, on one device."""
    from nashlib.autoregressive import AutoregressiveDecoder
    from nashlib.masked import MaskedBackbone
    from nashlib.nash import nash_decode

    numerics.configure(device)
    # Decode each dataset the way its table was produced; see `nashlib.datasets`.
    complete = datasets.COMPLETE_MASKED_SLOTS[dataset]
    path = (
        CLAPNQ_SUBSAMPLE
        if dataset == "clapnq" and CLAPNQ_SUBSAMPLE.exists()
        else datasets.evaluation_path(dataset)
    )
    # The CLAPNQ subsample is a plain file; the three benchmark corpora are loaded
    # through `load_dataset` so that CoQA is cut to its reported 1,814-turn cohort
    # rather than timed over all 7,983 turns of the file.
    examples = (
        datasets.read_jsonl(path)
        if path == CLAPNQ_SUBSAMPLE
        else datasets.load_dataset(path, dataset=dataset)
    )
    if limit:
        examples = examples[:limit]
    print(f"measuring on {len(examples)} examples from {path.name}, device {device}")

    import gc

    report = []
    for tag, family, label in SYSTEMS:
        model = registry.spec(tag)
        seconds, calls = [], []
        # One model resident at a time. Falcon-7B in float32 is about 29 GB, so
        # holding two would need a card nothing in this list otherwise requires.
        loaded = None
        if family == "autoregressive":
            decoder = loaded = AutoregressiveDecoder(
                model.hf_id, device=device, revision=model.revision,
                attn_implementation=model.attention,
            )
            for example in examples[:warmup]:
                decoder.generate(example["prompt"], decoder.answer_budget(example["target"]))
            for example in examples:
                generation = decoder.generate(
                    example["prompt"], decoder.answer_budget(example["target"])
                )
                seconds.append(generation.seconds)
                calls.append(generation.forwards)
        else:
            construction = "all_mask" if family == "nash_maxgap" else "l2r"
            backbone = loaded = MaskedBackbone(
                model.hf_id, device=device, revision=model.revision
            )
            for example in examples[:warmup]:
                nash_decode(
                    backbone,
                    example["modernbert_prompt"],
                    len(backbone.encode_answer(example["target"])),
                    construction=construction,
                    complete_masked_slots=complete,
                )
            for example in examples:
                trajectory = nash_decode(
                    backbone,
                    example["modernbert_prompt"],
                    len(backbone.encode_answer(example["target"])),
                    example_id=example["id"],
                    construction=construction,
                    complete_masked_slots=complete,
                )
                seconds.append(trajectory.seconds)
                calls.append(trajectory.forwards)
        report.append({
            "label": label,
            "tag": tag,
            "family": family,
            "seconds": statistics.mean(seconds),
            "seconds_median": statistics.median(seconds),
            "calls": statistics.mean(calls),
            "examples": len(examples),
        })
        print(f"  {label:26s} {report[-1]['seconds']:8.3f} s/question "
              f"{report[-1]['calls']:8.0f} calls")
        del loaded
        if family == "autoregressive":
            del decoder
        else:
            del backbone
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=registry.DATASETS, default=None,
                        help="re-measure the five systems on this device")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    arguments = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    report: dict = {"figure_6_b200": FIGURE_6, "recorded": {}}

    print("Figure 6: B200, float32, TF32 disabled")
    print(f"  {'system':26s}" + "".join(
        f"{DISPLAY[d]:>22s}" for d in ("coqa", "pubmedqa", "clapnq")))
    print(f"  {'':26s}" + "".join(f"{'s/question':>11}{'calls':>11}"
                                  for _ in ("coqa", "pubmedqa", "clapnq")))
    for index, (_, _, label) in enumerate(SYSTEMS):
        line = f"  {label:26s}"
        for dataset in ("coqa", "pubmedqa", "clapnq"):
            line += (f"{FIGURE_6[dataset]['seconds'][index]:11.2f}"
                     f"{FIGURE_6[dataset]['calls'][index]:11d}")
        print(line)
    print("  CLAPNQ is measured on a fixed 60-item subsample, every fifth item, "
          "shared by all five systems.")
    for dataset in ("coqa", "pubmedqa", "clapnq"):
        path = TIMING_FILE.get(dataset)
        if path is not None and path.exists():
            print(f"  {DISPLAY[dataset]:9s} measurements: "
                  f"artifacts/timing/{path.name}")
            report.setdefault("timing_files", {})[dataset] = path.name
            report.setdefault("timing", {})[dataset] = timing_measurements(dataset)
        else:
            print(f"  {DISPLAY[dataset]:9s} measurements: paper/paper_values.json")

    print("\nForward passes recorded by the evaluation runs themselves")
    print(f"  {'system':26s}" + "".join(
        f"{DISPLAY[d]:>26s}" for d in ("coqa", "pubmedqa", "clapnq")))
    for dataset in ("coqa", "pubmedqa", "clapnq"):
        report["recorded"][dataset] = stored_costs(dataset)
    for index, (_, _, label) in enumerate(SYSTEMS):
        line = f"  {label:26s}"
        for dataset in ("coqa", "pubmedqa", "clapnq"):
            entry = report["recorded"][dataset][index]
            if entry["calls"] is None:
                line += f"{'---':>26s}"
            elif entry.get("calls_to_canvas_full"):
                line += (f"{entry['calls']:12.0f}"
                         f" ({entry['calls_to_canvas_full']:.0f} + "
                         f"{entry['calls_after_canvas_full']:.0f})".rjust(14))
            else:
                line += f"{entry['calls']:26.0f}"
        print(line)
    print("  for the Nash rows: total (before the canvas is full + after)")

    if report.get("timing"):
        print("\nTiming measurements")
        for dataset, rows in report["timing"].items():
            print(f"  {DISPLAY[dataset]} (n = {rows[0]['n']})")
            for row in rows:
                line = (f"    {row['label']:26s} {row['seconds']:7.3f} s "
                        f"{row['calls']:8.1f} calls")
                if row.get("calls_to_full") is not None:
                    line += (f"   before the canvas is full: "
                             f"{row['seconds_to_full']:.3f} s, "
                             f"{row['calls_to_full']:.1f} calls")
                print(line)

    if arguments.measure:
        report["measured_here"] = measure(
            arguments.measure, f"cuda:{arguments.gpu}", arguments.limit, arguments.warmup
        )

    (RESULTS / "cost.json").write_text(json.dumps(report, indent=1) + "\n")
    print("\n-> results/cost.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
