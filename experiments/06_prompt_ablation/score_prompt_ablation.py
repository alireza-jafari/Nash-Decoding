#!/usr/bin/env python3
"""Table 10: the soft-instruction prompt against the Question/Answer prompt.

    python experiments/06_prompt_ablation/score_prompt_ablation.py
    python experiments/06_prompt_ablation/score_prompt_ablation.py --dataset pubmedqa

The prompt used throughout the paper adds one instruction line between the question
and the answer marker:

    {evidence} Question: {question}
    This question is answered completely with evidence from the {passage|abstract}.
    Answer:

The ablation deletes exactly that line and changes nothing else:

    {evidence} Question: {question} Answer:

Same evidence, same question, same oracle budget, same banned tokens, same decoding,
same references. `data/clapnq_qa_prompt.jsonl` and `data/pubmedqa_qa_prompt.jsonl`
are the soft-instruction files with the prompt field rewritten and every other field
carried through unchanged, so the two arms are paired example by example.

Table 10 reports the CLAPNQ rows for Falcon-7B under autoregressive decoding and
ModernBERT-Large under Nash decoding: the soft instruction improves all three metrics
for both decoding paradigms. The script also scores the other systems that were run
under both prompts, on CLAPNQ and on PubMedQA, with a paired bootstrap interval over
questions for each difference.
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

QA_PROMPT_FILE = {
    "clapnq": "clapnq_qa_prompt.jsonl",
    "pubmedqa": "pubmedqa_qa_prompt.jsonl",
}
INSTRUCTION_ARM = {
    "autoregressive": "autoregressive",
    "nash": "nash_maxgap",
}
QA_ARM = {
    "autoregressive": "qaprompt_autoregressive",
    "nash": "qaprompt_nash_maxgap",
}
# The systems that were re-run under both prompts. Table 10 prints the two in bold.
SYSTEMS = [
    ("opt_350m", "OPT-350M", "autoregressive"),
    ("gpt2_large", "GPT-2 Large", "autoregressive"),
    ("falcon_7b", "Falcon-7B", "autoregressive"),
    ("ettin_400m", "Ettin-400m", "nash"),
    ("modernbert_large", "ModernBERT-Large", "nash"),
]
IN_TABLE_2 = {("clapnq", "falcon_7b"), ("clapnq", "modernbert_large")}


def paired_bootstrap(
    differences: list[float], resamples: int = 4000, seed: int = 0
) -> tuple[float, float]:
    """A 95% interval for the mean paired difference, resampling questions."""
    generator = random.Random(seed)
    means = sorted(
        statistics.mean(generator.choices(differences, k=len(differences)))
        for _ in range(resamples)
    )
    low = int(0.025 * resamples)
    high = int(0.975 * resamples) - 1
    return 100 * means[low], 100 * means[high]


def check_prompts_differ_only_by_the_instruction(dataset: str) -> dict:
    """The two files must agree on everything except the prompt string."""
    soft = {
        row["id"]: row
        for row in datasets.load_dataset(
            datasets.evaluation_path(dataset), dataset=dataset
        )
    }
    plain = {
        row["id"]: row
        for row in datasets.read_jsonl(REPOSITORY_ROOT / "data" / QA_PROMPT_FILE[dataset])
    }
    if set(soft) != set(plain):
        raise AssertionError(f"{dataset}: the two prompt files cover different questions")
    for example_id, row in soft.items():
        other = plain[example_id]
        for field in ("target", "references", "source_text"):
            if row[field] != other[field]:
                raise AssertionError(f"{dataset}/{example_id}: {field} differs")
        if other["prompt"] != other["modernbert_prompt"]:
            raise AssertionError(f"{dataset}/{example_id}: masked and causal prompts differ")
        if len(other["prompt"]) >= len(row["prompt"]):
            raise AssertionError(
                f"{dataset}/{example_id}: the Q/A prompt is not shorter than the "
                "instruction prompt, so the instruction was not removed"
            )
    return {
        "questions": len(soft),
        "mean_prompt_characters_instruction": statistics.mean(
            len(row["prompt"]) for row in soft.values()
        ),
        "mean_prompt_characters_qa": statistics.mean(
            len(row["prompt"]) for row in plain.values()
        ),
    }


def score_dataset(dataset: str) -> dict:
    rows = datasets.load_dataset(
        datasets.evaluation_path(dataset), dataset=dataset
    )
    references = datasets.references_by_id(rows)
    report = {"dataset": dataset, "prompts": check_prompts_differ_only_by_the_instruction(dataset),
              "systems": {}}
    for tag, display, family in SYSTEMS:
        arms = {}
        for name, family_map in (("instruction", INSTRUCTION_ARM), ("qa", QA_ARM)):
            if not artifacts.has_predictions(dataset, family_map[family], tag):
                arms[name] = None
                continue
            predictions = artifacts.load_predictions(dataset, family_map[family], tag)
            arms[name] = {
                "scores": metrics.score_predictions(
                    {row["id"]: row["text"] for row in predictions}, references
                ),
                "per_example": {
                    row["id"]: metrics.best_reference_f1(row["text"], references[row["id"]])
                    for row in predictions
                },
            }
        entry: dict = {"display": display, "family": family}
        for name in ("instruction", "qa"):
            entry[name] = None if arms[name] is None else {
                key: arms[name]["scores"][key] for key in ("F1", "RL", "RLsum")
            }
        if arms["instruction"] and arms["qa"]:
            shared = sorted(set(arms["instruction"]["per_example"]) & set(arms["qa"]["per_example"]))
            differences = [
                arms["qa"]["per_example"][i] - arms["instruction"]["per_example"][i]
                for i in shared
            ]
            low, high = paired_bootstrap(differences)
            entry["qa_minus_instruction_f1"] = 100 * statistics.mean(differences)
            entry["bootstrap_ci"] = [low, high]
            entry["interval_excludes_zero"] = (low > 0) or (high < 0)
        report["systems"][tag] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("clapnq", "pubmedqa"), default=None)
    arguments = parser.parse_args()
    targets = (arguments.dataset,) if arguments.dataset else ("clapnq", "pubmedqa")

    RESULTS.mkdir(parents=True, exist_ok=True)
    reports = {}
    for dataset in targets:
        report = score_dataset(dataset)
        reports[dataset] = report
        print(f"\n{dataset}  ({report['prompts']['questions']} questions, "
              f"prompt {report['prompts']['mean_prompt_characters_qa']:.0f} -> "
              f"{report['prompts']['mean_prompt_characters_instruction']:.0f} characters "
              f"with the instruction)")
        print(f"  {'system':18s} {'prompt':>12s} {'F1':>7} {'R-L':>7} {'R-Ls':>7}   "
              f"{'Q/A - instruction F1':>28}")
        for tag, entry in report["systems"].items():
            for name, label in (("qa", "Q/A"), ("instruction", "soft")):
                values = entry[name]
                marker = " *" if (dataset, tag) in IN_TABLE_2 else "  "
                if values is None:
                    print(f"  {entry['display']:18s} {label:>12s} {'---':>7} {'---':>7} {'---':>7}{marker}")
                    continue
                print(f"  {entry['display']:18s} {label:>12s} {values['F1']:7.2f} "
                      f"{values['RL']:7.2f} {values['RLsum']:7.2f}{marker}")
            if "qa_minus_instruction_f1" in entry:
                low, high = entry["bootstrap_ci"]
                flag = "" if entry["interval_excludes_zero"] else "   (interval spans 0)"
                print(f"  {'':18s} {'':>12s} {'':>7} {'':>7} {'':>7}   "
                      f"{entry['qa_minus_instruction_f1']:+7.2f} "
                      f"[{low:+.2f}, {high:+.2f}]{flag}")
        print("  * = printed in Table 10")

    (RESULTS / "prompt_ablation.json").write_text(json.dumps(reports, indent=1) + "\n")
    print("\n-> results/prompt_ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
