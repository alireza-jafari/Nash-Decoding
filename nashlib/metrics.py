"""The three reported metrics, and the official CoQA accumulation used as a check.

Appendix E.7. Three numbers are reported per dataset:

F1        SQuAD-style token F1 (Rajpurkar et al., 2016): lowercase, drop ASCII
          punctuation, drop the articles `a`, `an`, `the`, collapse whitespace,
          then the harmonic mean of unigram precision and recall against the
          *most favorable* reference.
R-L       ROUGE-L F-measure from `rouge_score`, `use_stemmer=False`, accumulated
          over all references with `score_multi`.
R-Lsum    ROUGE-Lsum F-measure on sentence-split text. Sentence splitting changes
          only R-Lsum: ROUGE-L tokenization treats a newline like any other
          separator.

`coqa_f1` is the official leave-one-annotator-out accumulation CoQA defines for a
turn with several human answers: average over i of max over j != i of F1 against
reference j. The paper's tables print the best-reference column; the official one
runs about 0.9 to 1.6 points lower on the systems of Table 7 and preserves their
ordering. Both are stored for every CoQA system.

This module has no torch dependency, so metrics can be recomputed on any machine.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

_PUNCTUATION = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])(?:["\')\]]*)\s+')


def normalize(text: str) -> str:
    """SQuAD normalization: lowercase, strip punctuation and articles, collapse spaces."""
    stripped = "".join(c for c in text.lower() if c not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", stripped).split())


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD token F1 between one prediction and one reference."""
    predicted = normalize(prediction).split()
    gold = normalize(reference).split()
    if not predicted or not gold:
        # Both empty is a match; one empty is not. This mirrors the SQuAD script.
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def best_reference_f1(prediction: str, references: Sequence[str]) -> float:
    """Token F1 against the most favorable reference. This is the printed column."""
    if not references:
        raise ValueError("an example with no reference answer cannot be scored")
    return max(token_f1(prediction, reference) for reference in references)


def official_coqa_f1(prediction: str, references: Sequence[str]) -> float:
    """Leave-one-annotator-out F1: average over i of max over j != i."""
    if not references:
        raise ValueError("an example with no reference answer cannot be scored")
    if len(references) <= 1:
        return token_f1(prediction, references[0])
    total = 0.0
    for held_out in range(len(references)):
        total += max(
            token_f1(prediction, reference)
            for index, reference in enumerate(references)
            if index != held_out
        )
    return total / len(references)


def sentence_split(text: str) -> str:
    """Newline-separate sentences, which is what distinguishes R-Lsum from R-L."""
    parts = (part.strip() for part in _SENTENCE_BOUNDARY.split(text.strip()))
    return "\n".join(part for part in parts if part)


class RougeScorer:
    """ROUGE-L / ROUGE-Lsum F-measure, multi-reference, no stemming.

    Wrapped in a class only so the underlying `rouge_score` scorer is built once;
    it is the single most expensive object in the scoring path.
    """

    def __init__(self) -> None:
        from rouge_score import rouge_scorer

        self._scorer = rouge_scorer.RougeScorer(
            ["rougeL", "rougeLsum"], use_stemmer=False
        )

    def score(self, prediction: str, references: Sequence[str]) -> tuple[float, float]:
        scores = self._scorer.score_multi(
            [sentence_split(reference) for reference in references],
            sentence_split(prediction),
        )
        return scores["rougeL"].fmeasure, scores["rougeLsum"].fmeasure


def score_predictions(
    predictions: dict[str, str],
    references: dict[str, Sequence[str]],
    *,
    official_coqa: bool = False,
) -> dict[str, float]:
    """Aggregate the reported metrics over a whole run, as percentages.

    `predictions` and `references` must cover exactly the same example ids; a run
    that is missing examples is a partial run and must not be scored, so this
    raises rather than silently averaging over fewer items.
    """
    if not references:
        raise ValueError("nothing to score: the evaluation set is empty")
    missing = set(references) - set(predictions)
    extra = set(predictions) - set(references)
    if missing or extra:
        raise ValueError(
            f"prediction set does not match the evaluation set: "
            f"{len(missing)} missing, {len(extra)} unexpected"
        )
    rouge = RougeScorer()
    f1_values, rouge_l, rouge_lsum, coqa = [], [], [], []
    for example_id, prediction in predictions.items():
        refs = references[example_id]
        f1_values.append(best_reference_f1(prediction, refs))
        left, right = rouge.score(prediction, refs)
        rouge_l.append(left)
        rouge_lsum.append(right)
        if official_coqa:
            coqa.append(official_coqa_f1(prediction, refs))
    n = len(f1_values)
    result = {
        "F1": 100 * sum(f1_values) / n,
        "RL": 100 * sum(rouge_l) / n,
        "RLsum": 100 * sum(rouge_lsum) / n,
        "n": n,
    }
    if official_coqa:
        result["coqa_f1"] = 100 * sum(coqa) / n
    return result
