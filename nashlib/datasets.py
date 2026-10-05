"""The evaluation JSONL schema, shared by all three question-answering benchmarks.

One line per evaluated question:

    id                 stable example identifier
    prompt             the prompt the autoregressive systems receive
    modernbert_prompt  the prompt the masked encoders receive
    target             the gold answer that sets the oracle length budget
    references         every reference answer, for best-reference scoring
    source_text        the passage or abstract, for the dataset statistics
    metadata           per-dataset provenance (split, conversation id, ...)

The prompt is already assembled in the file rather than templated at run time, so
that "every system of a family receives the same string" is a property of the data
and not of a code path. The files are shipped in derived form; `data/README.md`
describes the derivation.

**The two prompt fields are equal on PubMedQA and CLAPNQ and deliberately unequal
on CoQA.** PubMedQA and CLAPNQ use the soft-instruction prompt, which every system
receives byte for byte. CoQA uses the dialogue prompt of Radford et al. (2019), and
Appendix E.1 specifies that the masked encoders receive it whitespace-normalised:
the `Q:`/`A:` line breaks become single spaces and a leading space is prepended.
This module pins that exact transformation (`masked_prompt_for_coqa`) and requires
every row to satisfy it; the passage's own paragraph breaks are kept.

CoQA is loaded from one file of all 7,983 development turns. The 1,814-turn cohort
the main tables report is *derived* from it by the length filter, so both cohorts
share one copy of every prompt and reference.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

REQUIRED_FIELDS = (
    "id",
    "prompt",
    "modernbert_prompt",
    "target",
    "references",
    "source_text",
)

# `coqa` is the >= 5-token cohort of Table 1 and Table 7's left block; `coqa_full`
# is every development turn, Table 7's right block. Both read the same file.
EXPECTED_ROWS = {"coqa": 1814, "coqa_full": 7983, "pubmedqa": 500, "clapnq": 300}

COQA_DATASETS = ("coqa", "coqa_full")

# Appendix D.3: the retained cohort is the turns whose gold answer occupies at
# least this many ModernBERT-Large tokens, counted as " " + target. The count is
# recorded per row at build time, so applying the filter needs no tokenizer.
COHORT_MIN_GOLD_TOKENS = 5

# What a Nash trajectory returns when it stops at a repeated state while some
# slots still hold [MASK]. The CLAPNQ runs fill those slots before returning --
# left to right, with [MASK] removed from the action set -- so every CLAPNQ answer
# has exactly T tokens; on CoQA and PubMedQA the answer is the written slots. The
# setting is recorded here so that every caller decodes a dataset the way its table
# was made.
COMPLETE_MASKED_SLOTS = {
    "coqa": False,
    "coqa_full": False,
    "pubmedqa": False,
    "clapnq": True,
}

INSTRUCTION = {
    "clapnq": "This question is answered completely with evidence from the passage.",
    "pubmedqa": "This question is answered completely with evidence from the abstract.",
    "coqa": "This question is answered completely with evidence from the passage.",
    "coqa_full": "This question is answered completely with evidence from the passage.",
}

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Which file each experiment reads, in one place.
#
#   evaluation   the corpus the result tables are computed on
#   grounded     the same questions with one clause appended to the instruction;
#                the coordinate-rule comparison of Appendix H, which is the one
#                experiment that keeps the soft-instruction wording on CoQA
#   qa_prompt    the instruction line deleted; the prompt ablation of Table 10
EVALUATION_FILE = {
    "coqa": "coqa_paperprompt.jsonl.gz",
    "coqa_full": "coqa_paperprompt.jsonl.gz",
    "pubmedqa": "pubmedqa_softinstr.jsonl",
    "clapnq": "clapnq_softinstr.jsonl",
}
GROUNDED_FILE = {
    "coqa": "coqa_grounded.jsonl.gz",
    "pubmedqa": "pubmedqa_grounded.jsonl",
    "clapnq": "clapnq_grounded.jsonl",
}
QA_PROMPT_FILE = {
    "pubmedqa": "pubmedqa_qa_prompt.jsonl",
    "clapnq": "clapnq_qa_prompt.jsonl",
}

# The prompt shape each file is required to have. `coqa_dialogue` is the
# Q:/A: format of Appendix E.1, the only one where the two families receive
# different strings; the other two give every system the same string and differ
# only in whether the instruction line is present.
PROMPT_STYLES = ("coqa_dialogue", "soft_instruction", "qa_prompt")
DEFAULT_PROMPT_STYLE = {
    "coqa": "coqa_dialogue",
    "coqa_full": "coqa_dialogue",
    "pubmedqa": "soft_instruction",
    "clapnq": "soft_instruction",
}


def masked_prompt_for_coqa(prompt: str) -> str:
    """The CoQA prompt as the masked encoders receive it (Appendix E.1).

    The dialogue delimiters are flattened to single spaces and a leading space is
    prepended, so the canvas is appended to a string that ends in `A:` with no
    newline. Everything else, including the passage's own paragraph breaks, is
    untouched.
    """
    return " " + prompt.replace("\n\nQ:", " Q:").replace("\nA:", " A:")


def current_question(row: dict, dataset: str) -> str:
    """The question being asked, pulled back out of the assembled prompt.

    The prompt is stored assembled rather than templated, so the dataset
    statistics have to recover its parts. CoQA's dialogue prompt puts the current
    question on the last `Q:` line and nothing after it but the bare `A:`; the
    soft-instruction prompt puts it on the `Question:` line.
    """
    prompt = row["prompt"]
    if dataset in COQA_DATASETS:
        return prompt.rsplit("\nQ: ", 1)[-1].split("\nA:")[0].strip()
    return prompt.split(" Question: ")[-1].split("\n")[0].strip()


def dialogue_history(row: dict, dataset: str) -> str:
    """The preceding turns of a CoQA conversation, as they appear in the prompt.

    Empty on the first turn of a conversation, and on the other two benchmarks,
    which have no dialogue.
    """
    if dataset not in COQA_DATASETS:
        return ""
    prompt = row["prompt"]
    if "\n\nQ: " not in prompt:
        return ""
    body = prompt.split("\n\nQ: ", 1)[1]
    turns = body.rsplit("\n\nQ: ", 1)
    return turns[0] if len(turns) == 2 else ""


def _open_text(path: Path):
    """Open a JSONL file whether or not it is gzipped.

    The CoQA corpus is 44 MB as text and 1.1 MB compressed, because a conversation's
    passage is repeated on every one of its turns. It ships compressed; the smaller
    files ship as text, since compressing them buys little and makes them harder to
    look at. Either extension loads the same way, so no caller needs to know which
    it is.
    """
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    with _open_text(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with _open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write atomically and with sorted keys, so the file hashes reproducibly.

    A `.gz` path is written with the timestamp zeroed, because gzip stores one by
    default and a file whose bytes change every time it is written cannot have a
    checksum recorded for it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    if path.suffix == ".gz":
        with gzip.GzipFile(temporary, "wb", compresslevel=9, mtime=0) as handle:
            handle.write(body.encode("utf-8"))
    else:
        temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def in_cohort(row: dict) -> bool:
    """Whether a CoQA turn belongs to the >= 5-token cohort."""
    return row["metadata"]["gold_mb_tokens"] >= COHORT_MIN_GOLD_TOKENS


def _check_prompt_shape(path, dataset: str, style: str, rows: list[dict]) -> None:
    """The prompt invariant, which is not the same one for every file.

    Three shapes, named rather than inferred, so that reading a CoQA file under
    the soft-instruction shape is a deliberate act by the caller and not an
    accident of which file it happened to open.
    """
    if style not in PROMPT_STYLES:
        raise KeyError(
            f"unknown prompt style {style!r}; expected one of {PROMPT_STYLES}"
        )

    if style == "coqa_dialogue":
        wrong_tail = [row["id"] for row in rows if not row["prompt"].endswith("\nA:")]
        if wrong_tail:
            raise ValueError(
                f"{path}: {len(wrong_tail)} CoQA prompts do not end at the answer "
                f"delimiter of the dialogue format (first: {wrong_tail[0]})"
            )
        offenders = [
            row["id"]
            for row in rows
            if row["modernbert_prompt"] != masked_prompt_for_coqa(row["prompt"])
        ]
        if offenders:
            raise ValueError(
                f"{path}: {len(offenders)} masked prompts are not the documented "
                f"whitespace normalisation of the causal prompt "
                f"(first: {offenders[0]}); see nashlib.datasets.masked_prompt_for_coqa"
            )
        return

    instruction = INSTRUCTION[dataset]
    if style == "soft_instruction":
        offenders = [row["id"] for row in rows if instruction not in row["prompt"]]
        if offenders:
            raise ValueError(
                f"{path}: {len(offenders)} prompts do not carry the soft "
                f"instruction (first: {offenders[0]})"
            )
    else:
        carriers = [row["id"] for row in rows if instruction in row["prompt"]]
        if carriers:
            raise ValueError(
                f"{path}: {len(carriers)} prompts still carry the instruction line "
                f"the ablation arm removes (first: {carriers[0]})"
            )
    unequal = [row["id"] for row in rows if row["prompt"] != row["modernbert_prompt"]]
    if unequal:
        raise ValueError(
            f"{path}: {len(unequal)} rows give the masked and causal families "
            f"different strings, so the comparison would not be prompt-matched "
            f"(first: {unequal[0]})"
        )


def load_dataset(
    path: str | Path,
    *,
    dataset: str | None = None,
    prompt_style: str | None = None,
) -> list[dict]:
    """Read an evaluation file and check the invariants the protocol relies on.

    Naming a dataset also selects its evaluated subset: `coqa` returns the 1,814
    cohort out of the 7,983 turns in the file, `coqa_full` returns all of them.

    `prompt_style` defaults to the shape that dataset's *evaluation* corpus has.
    A caller reading one of the prompt variants has to say so, which is how the
    coordinate-rule experiment declares that its CoQA file keeps the
    soft-instruction wording rather than the dialogue format the tables use.
    """
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path} line {index + 1} is missing {missing}")
        if not row["references"]:
            raise ValueError(f"{path} line {index + 1} has no references")
    identifiers = [row["id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path} has duplicate example ids")
    if dataset is None:
        return rows

    if dataset not in EXPECTED_ROWS:
        raise KeyError(
            f"unknown dataset {dataset!r}; expected one of {sorted(EXPECTED_ROWS)}"
        )
    if dataset == "coqa" and len(rows) != EXPECTED_ROWS["coqa"]:
        # The evaluation corpus holds all 7,983 turns and the cohort is derived
        # from it. The prompt variants are already stored at cohort size, so a
        # file that is the right length is taken as it stands.
        uncounted = [
            row["id"] for row in rows if "gold_mb_tokens" not in row["metadata"]
        ]
        if uncounted:
            raise ValueError(
                f"{path}: {len(uncounted)} rows carry no metadata.gold_mb_tokens, so "
                f"the reported cohort cannot be derived (first: {uncounted[0]})"
            )
        rows = [row for row in rows if in_cohort(row)]
    expected = EXPECTED_ROWS[dataset]
    if len(rows) != expected:
        raise ValueError(
            f"{path} gives {len(rows)} rows for {dataset}; "
            f"that evaluation set has {expected}"
        )
    style = prompt_style or DEFAULT_PROMPT_STYLE[dataset]
    _check_prompt_shape(path, dataset, style, rows)
    return rows


def evaluation_path(dataset: str) -> Path:
    """The corpus the result tables for this dataset are computed on."""
    return DATA_ROOT / EVALUATION_FILE[dataset]


def load_evaluation(dataset: str) -> list[dict]:
    """The evaluated questions of one dataset, checked and cut to its cohort."""
    return load_dataset(evaluation_path(dataset), dataset=dataset)


def references_by_id(rows: list[dict]) -> dict[str, list[str]]:
    return {row["id"]: list(row["references"]) for row in rows}
