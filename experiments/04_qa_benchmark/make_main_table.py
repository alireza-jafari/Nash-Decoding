#!/usr/bin/env python3
"""Table 1: the main table, three metrics on each of three datasets.

    python experiments/04_qa_benchmark/make_main_table.py

Table 1 is a *selection* from Tables 7-9: every value here also appears in the
per-dataset table for that dataset, and this script reads the same scores files. It
is the table of the main body, which shows the comparison at a glance -- the
three masked encoders decoded one-shot, six autoregressive decoders spanning 331M
to 7B, and the same three masked encoders under Nash decoding.

CoQA is the 1,814-turn cohort here, the same one the left block of Table 7 scores.

Run `score_benchmark.py --all` first; this reads `results/<dataset>_scores.json`.

The rows are the ones Table 1 prints, read from `paper/paper_values.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import registry, tables  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
DATASETS = ("coqa", "pubmedqa", "clapnq")
DISPLAY = ("CoQA", "PubMedQA", "CLAPNQ")

CAPTION = (
    "Results across three question-answering benchmarks under a shared evaluation "
    "protocol. Size denotes the number of model parameters; for Nash decoding, it "
    "denotes the masked model size. Best results are bolded and second-best "
    "results are underlined. The complete benchmark results are reported in "
    "Appendix F."
)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    paper = json.loads((REPOSITORY_ROOT / "paper" / "paper_values.json").read_text())
    selection = paper["table_1_main"]["blocks"]

    scores = {}
    for dataset in DATASETS:
        path = RESULTS / f"{dataset}_scores.json"
        if not path.exists():
            raise SystemExit(
                f"{path} is missing; run `score_benchmark.py --all` first"
            )
        scores[dataset] = json.loads(path.read_text())

    blocks = {}
    for family in ("oneshot", "autoregressive", "nash"):
        tags = list(selection[family])
        missing = [
            (dataset, family, tag)
            for dataset in DATASETS
            for tag in tags
            if tag not in scores[dataset].get(family, {})
        ]
        if missing:
            raise SystemExit(
                f"Table 1 needs scores that are not present: {missing}. Every row of "
                "Table 1 is also a row of a per-dataset table, so this means one of "
                "those tables was not built."
            )
        blocks[family] = tuple(registry.spec(tag) for tag in tags)

    latex = tables.latex_three_datasets(
        blocks, scores, datasets=DATASETS, display_names=DISPLAY,
        caption=CAPTION, label="tab:main",
    )
    (RESULTS / "main_table.tex").write_text(latex + "\n")

    header = f"| Model | Size | " + " | ".join(
        f"{name} F1 | {name} R-L | {name} R-Ls" for name in DISPLAY
    ) + " |"
    lines = [
        "### Table 1 - main results",
        "",
        header,
        "|" + "|".join(["---", "---:"] + ["---:"] * 9) + "|",
    ]
    for family, models in blocks.items():
        lines.append(f"| **{tables.MAIN_TABLE_BLOCK_TITLES[family]}** |" + " |" * 10)
        for model in models:
            cells = []
            for dataset in DATASETS:
                entry = scores[dataset][family][model.tag]
                cells += [f"{entry[key]:.2f}" for key in ("F1", "RL", "RLsum")]
            lines.append(
                f"| {model.display} | {model.parameters} | " + " | ".join(cells) + " |"
            )
    (RESULTS / "main_table.md").write_text("\n".join(lines) + "\n")

    print(f"{sum(len(v) for v in blocks.values())} rows "
          f"({', '.join(f'{k} {len(v)}' for k, v in blocks.items())})")
    for family, models in blocks.items():
        for model in models:
            row = " ".join(
                f"{scores[d][family][model.tag]['F1']:6.2f}" for d in DATASETS
            )
            print(f"  {model.display:20s} {model.parameters:>5s}  {row}")
    print(f"\n-> results/main_table.tex, results/main_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
