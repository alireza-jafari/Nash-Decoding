"""Rendering the paper's tables from a scores file, in LaTeX and in Markdown.

Three properties matter more than the formatting.

*Every scored system in scope appears.* The renderer takes the row list from
`nashlib.registry`, which derives it from `paper/paper_values.json`.

*Scores are keyed by family and then by tag.* The three masked encoders appear in
two blocks of every result table -- decoded one-shot and under Nash decoding -- under
the same model tag, so a flat tag-keyed dictionary would let one block silently
overwrite the other. Every renderer here takes `scores[family][tag]`.

*A dash means "not run", never "zero" and never "failed".* A system with no score
raises unless the caller names it in `allowed_missing`, and so does a system whose
score is missing one of the requested metrics -- a partially-populated row is the
more likely accident of the two, since `artifacts.telemetry` only reports the fields
a run actually recorded.
"""

from __future__ import annotations

from collections.abc import Sequence

from .registry import ModelSpec

BLOCK_TITLES = {
    "oneshot": "One-shot masked LMs (not fine-tuned)",
    "autoregressive": "Autoregressive decoders",
    "nash": "Nash decoding with same masked LMs",
}

# Table 1 heads its autoregressive block differently from the appendix tables.
MAIN_TABLE_BLOCK_TITLES = {
    "oneshot": "One-shot masked LMs (not fine-tuned)",
    "autoregressive": "Autoregressive models",
    "nash": "Nash decoding with same masked LMs",
}

# Spelled as the paper spells them: "R-Ls", and no up-arrows.
METRIC_HEADERS = {
    "F1": r"\textbf{F1}",
    "RL": r"\textbf{R-L}",
    "RLsum": r"\textbf{R-Ls}",
    "calls": r"\textbf{Calls}",
    "chars": r"\textbf{Char}",
}


def _require_metrics(
    tag: str, entry: dict[str, float] | None, metrics: Sequence[str],
    allowed_missing: Sequence[str],
) -> None:
    """A row is either fully populated, or explicitly allowed to be absent."""
    if tag in allowed_missing:
        return
    if entry is None:
        raise KeyError(
            f"no score for {tag}; a table cell may only be dashed for a system that "
            f"was deliberately not run (pass allowed_missing)"
        )
    absent = [metric for metric in metrics if entry.get(metric) is None]
    if absent:
        raise KeyError(
            f"the score for {tag} is missing {absent}; a partially-populated row "
            f"would render as dashes that look like 'not run'"
        )


def _wrap(body: str, caption: str, label: str) -> str:
    """Put a rendered tabular inside a table environment.

    The label line is omitted rather than emitted empty, so that a caption which
    happens to contain a blank line is not mangled by a blanket newline squeeze.
    """
    lines = [r"\begin{table}[t]", r"\centering", body, rf"\caption{{{caption}}}"]
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _format(
    value: float | None,
    metric: str,
    bold: bool = False,
    underline: bool = False,
) -> str:
    if value is None:
        return "---"
    if metric == "calls":
        text = f"{value:.1f}" if value < 100 else f"{value:,.0f}"
    elif metric == "chars":
        text = f"{value:.0f}"
    else:
        text = f"{value:.2f}"
    if bold:
        return rf"\textbf{{{text}}}"
    if underline:
        return rf"\underline{{{text}}}"
    return text


def _best_two(values: Sequence[float | None], places: int = 2) -> tuple[float | None, float | None]:
    """The best and second-best *printed* value in a column.

    Table 1 bolds the best and underlines the second-best, in all nine of its metric
    columns rather than only in F1.

    Ranking is done on the value as printed, not on full precision. Two systems whose
    scores differ in the sixth decimal print the same two digits, and marking one bold
    and the other underlined would show the reader a distinction the table does not
    contain. Ranking on the rounded value makes them tie instead, and a tie for first
    means both are bold and the underline moves to the next distinct printed value --
    which is what a reader of the printed table would infer.
    """
    present = sorted(
        {round(value, places) for value in values if value is not None}, reverse=True
    )
    best = present[0] if present else None
    second = present[1] if len(present) > 1 else None
    return best, second


def latex_single_dataset(
    blocks: dict[str, Sequence[ModelSpec]],
    scores: dict[str, dict[str, float]],
    *,
    metrics: Sequence[str] = ("F1", "RL", "RLsum", "calls", "chars"),
    caption: str = "",
    label: str = "",
    bold_best_f1: bool = True,
    allowed_missing: Sequence[str] = (),
) -> str:
    """One dataset, one row per system, grouped into the paper's three blocks."""
    best = None
    if bold_best_f1:
        available = [
            scores[family][s.tag].get("F1")
            for family, rows in blocks.items()
            for s in rows
            if s.tag in scores.get(family, {})
        ]
        available = [value for value in available if value is not None]
        best = max(available) if available else None

    lines = [
        r"\begin{tabular}{l" + "c" * (len(metrics) + 1) + "}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Size} & "
        + " & ".join(METRIC_HEADERS[m] for m in metrics)
        + r" \\",
    ]
    for family, rows in blocks.items():
        lines += [r"\midrule", rf"\multicolumn{{{len(metrics) + 2}}}{{l}}{{\textit{{{BLOCK_TITLES[family]}}}}} \\"]
        for model in rows:
            entry = scores.get(family, {}).get(model.tag)
            _require_metrics(model.tag, entry, metrics, allowed_missing)
            cells = []
            for metric in metrics:
                value = None if entry is None else entry.get(metric)
                is_best = (
                    metric == "F1" and best is not None and value is not None
                    and abs(value - best) < 1e-9
                )
                cells.append(_format(value, metric, is_best))
            lines.append(
                f"{model.display} & {model.parameters} & " + " & ".join(cells) + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    if not caption:
        return body
    return _wrap(body, caption, label)


def latex_three_datasets(
    blocks: dict[str, Sequence[ModelSpec]],
    scores: dict[str, dict[str, dict[str, float]]],
    *,
    datasets: Sequence[str] = ("coqa", "pubmedqa", "clapnq"),
    display_names: Sequence[str] = ("CoQA", "PubMedQA", "CLAPNQ"),
    caption: str = "",
    label: str = "",
    allowed_missing: Sequence[tuple[str, str]] = (),
) -> str:
    """The main table: three metrics on each of three datasets, nine columns.

    `allowed_missing` holds (dataset, system) pairs that are deliberately not run
    and may therefore render as dashes. Anything else missing raises.
    """
    metrics = ("F1", "RL", "RLsum")
    for family, family_rows in blocks.items():
        for model in family_rows:
            for dataset in datasets:
                if (dataset, model.tag) in allowed_missing:
                    continue
                _require_metrics(
                    f"{dataset}/{family}/{model.tag}",
                    scores[dataset].get(family, {}).get(model.tag), metrics, (),
                )
    # Best and second-best per (dataset, metric) column -- nine columns, nine pairs.
    ranked = {
        (dataset, metric): _best_two(
            [
                scores[dataset][family][s.tag].get(metric)
                for family, rows in blocks.items()
                for s in rows
                if s.tag in scores[dataset].get(family, {})
            ]
        )
        for dataset in datasets
        for metric in metrics
    }
    header_groups = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{name}}}" for name in display_names
    )
    lines = [
        r"\begin{tabular}{ll" + "ccc" * len(datasets) + "}",
        r"\toprule",
        rf"& & {header_groups} \\",
        r"\textbf{Model} & \textbf{Size} & "
        + " & ".join([r"\textbf{F1}", r"\textbf{R-L}", r"\textbf{R-Ls}"] * len(datasets))
        + r" \\",
    ]
    for family, rows in blocks.items():
        lines += [
            r"\midrule",
            rf"\multicolumn{{{2 + 3 * len(datasets)}}}{{l}}"
            rf"{{\textit{{{MAIN_TABLE_BLOCK_TITLES[family]}}}}} \\",
        ]
        for model in rows:
            cells = []
            for dataset in datasets:
                entry = scores[dataset].get(family, {}).get(model.tag)
                for metric in metrics:
                    value = None if entry is None else entry.get(metric)
                    best, second = ranked[(dataset, metric)]
                    shown = None if value is None else round(value, 2)
                    is_best = shown is not None and shown == best
                    is_second = shown is not None and shown == second
                    cells.append(_format(value, metric, is_best, is_second))
            lines.append(
                f"{model.display} & {model.parameters} & " + " & ".join(cells) + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    if not caption:
        return body
    return _wrap(body, caption, label)


def latex_cohort_pair(
    blocks: dict[str, Sequence[ModelSpec]],
    cohort_scores: dict[str, dict[str, float]],
    full_scores: dict[str, dict[str, float]],
    *,
    cohort_metrics: Sequence[str] = ("F1", "RL", "RLsum", "calls", "chars"),
    full_metrics: Sequence[str] = ("F1", "RL", "RLsum"),
    cohort_heading: str = "Gold $\\geq 5$ tokens ($n = 1{,}814$)",
    full_heading: str = "All ($n = 7{,}983$)",
    caption: str = "",
    label: str = "",
    allowed_missing: Sequence[str] = (),
) -> str:
    """Table 7: one row per system, scored on two cohorts of the same benchmark.

    The left block is the reported cohort and carries the cost columns; the right
    block is every development turn and carries the three metrics only, because
    that is what the paper prints. Both blocks take their row list from the same
    place, so a system cannot appear in one and be forgotten in the other.

    No cell is bolded or underlined, because no cell of Table 7 is marked in the
    paper -- unlike Table 1, which marks best and second-best in all nine of its
    metric columns.
    """
    lines = [
        r"\begin{tabular}{l" + "c" * (len(cohort_metrics) + len(full_metrics) + 1) + "}",
        r"\toprule",
        rf"& & \multicolumn{{{len(cohort_metrics)}}}{{c}}{{{cohort_heading}}} "
        rf"& \multicolumn{{{len(full_metrics)}}}{{c}}{{{full_heading}}} \\",
        r"\textbf{Model} & \textbf{Size} & "
        + " & ".join(
            [METRIC_HEADERS[m] for m in cohort_metrics]
            + [METRIC_HEADERS[m] for m in full_metrics]
        )
        + r" \\",
    ]
    for family, rows in blocks.items():
        lines += [
            r"\midrule",
            rf"\multicolumn{{{2 + len(cohort_metrics) + len(full_metrics)}}}{{l}}"
            rf"{{\textit{{{BLOCK_TITLES[family]}}}}} \\",
        ]
        for model in rows:
            cohort_entry = cohort_scores.get(family, {}).get(model.tag)
            full_entry = full_scores.get(family, {}).get(model.tag)
            _require_metrics(model.tag, cohort_entry, cohort_metrics, allowed_missing)
            # Membership-test the plain tag, so `allowed_missing` can match; the
            # label only reaches the error message.
            _require_metrics(
                model.tag, full_entry, full_metrics, allowed_missing
            )
            cells = [
                _format(None if cohort_entry is None else cohort_entry.get(m), m)
                for m in cohort_metrics
            ] + [
                _format(None if full_entry is None else full_entry.get(m), m)
                for m in full_metrics
            ]
            lines.append(
                f"{model.display} & {model.parameters} & " + " & ".join(cells) + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    if not caption:
        return body
    return _wrap(body, caption, label)


def markdown_cohort_pair(
    blocks: dict[str, Sequence[ModelSpec]],
    cohort_scores: dict[str, dict[str, float]],
    full_scores: dict[str, dict[str, float]],
    *,
    cohort_metrics: Sequence[str] = ("F1", "RL", "RLsum", "calls", "chars"),
    full_metrics: Sequence[str] = ("F1", "RL", "RLsum"),
    title: str = "",
) -> str:
    """The same two-cohort table as readable text."""
    head = (
        ["Model", "Size"]
        + [f"{m} (1,814)" for m in cohort_metrics]
        + [f"{m} (7,983)" for m in full_metrics]
    )
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += [
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|",
    ]
    for family, rows in blocks.items():
        lines.append(f"| **{BLOCK_TITLES[family]}** |" + " |" * (len(head) - 1))
        for model in rows:
            cohort_entry = cohort_scores.get(family, {}).get(model.tag)
            full_entry = full_scores.get(family, {}).get(model.tag)
            cells = [
                _format(None if cohort_entry is None else cohort_entry.get(m), m)
                for m in cohort_metrics
            ] + [
                _format(None if full_entry is None else full_entry.get(m), m)
                for m in full_metrics
            ]
            lines.append(
                f"| {model.display} | {model.parameters} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)


def markdown_single_dataset(
    blocks: dict[str, Sequence[ModelSpec]],
    scores: dict[str, dict[str, float]],
    *,
    metrics: Sequence[str] = ("F1", "RL", "RLsum", "calls", "chars"),
    title: str = "",
) -> str:
    """The same table as readable text, for the results README."""
    head = ["Model", "Size", *metrics]
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += ["| " + " | ".join(head) + " |",
              "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for family, rows in blocks.items():
        lines.append(f"| **{BLOCK_TITLES[family]}** |" + " |" * (len(head) - 1))
        for model in rows:
            entry = scores.get(family, {}).get(model.tag)
            cells = [
                _format(None if entry is None else entry.get(m), m) for m in metrics
            ]
            lines.append(
                f"| {model.display} | {model.parameters} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)
