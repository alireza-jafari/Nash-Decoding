#!/usr/bin/env python3
"""Tables 5 and 6: what the three benchmarks actually contain.

    python experiments/10_dataset_statistics/compute_statistics.py
    python experiments/10_dataset_statistics/compute_statistics.py --table 6 --reference both

Everything here is computed from `data/*.jsonl` and a tokenizer; nothing is read
from a summary. Table 5 is scale and length, Table 6 is how much of each reference
answer is already present in its context and how that material is arranged.

Three conventions that change the numbers:

*Token counts* use `answerdotai/ModernBERT-Large` at the pinned revision, with the
answer tokenized as `" " + target` -- with the leading space, matching how the
decoding budget is computed. A different tokenizer gives a different budget for the
same answer, which is exactly why the oracle budget is computed per model rather
than fixed; the GPT-2 row is included to show the size of that effect.

*Lexical statistics* use the evaluation normalizer -- lowercase, drop ASCII
punctuation and the articles, collapse whitespace -- so "appears in the context"
means the same thing here as it does inside the F1 metric.

*Multiple references* need a convention. `--reference best-overlap`, the default,
picks one reference per question -- the one with the highest token overlap with the
context -- and computes every statistic from it. `--reference per-measure` lets each
statistic pick its own most favourable reference independently, which is strictly
more generous. They agree on PubMedQA, which has one reference per question, and
diverge on CoQA, which has four: the contiguous-span rate is 78.8% under the first
and 92.8% under the second. `--reference both` prints the two side by side.

The picture that comes out is three genuinely different tasks. CoQA answers are
short and usually a contiguous span of the passage (78.8%); CLAPNQ answers reuse
passage wording heavily (97.6% of tokens) but almost never as one span (4.0%), so
they have to be assembled; PubMedQA answers are largely new wording (76.8% of bigrams
are absent from the abstract), so no copying strategy reaches them. The copying
baselines at the bottom of Table 6 put a number on that: copying the whole context
scores 50.7 F1 on CLAPNQ and 4.9 on CoQA, and the best single context sentence --
chosen using the reference, so this is an oracle -- scores 67.6 on CLAPNQ and 47.2 on
CoQA.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import datasets, metrics  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

TOKENIZER = "answerdotai/ModernBERT-large"
TOKENIZER_REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
GPT2_TOKENIZER = "gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

DISPLAY = {"clapnq": "CLAPNQ", "pubmedqa": "PubMedQA", "coqa": "CoQA"}
SENTENCE = re.compile(r'(?<=[.!?])(?:["\')\]]*)\s+')
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "he", "she", "they", "we", "you",
    "i", "his", "her", "their", "our", "your", "my", "not", "no", "do", "does", "did",
    "has", "have", "had", "will", "would", "can", "could", "may", "might", "should",
}


def content_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in STOPWORDS]


def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))


def contiguous(needle: list[str], haystack: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return True
    return False


def longest_shared_run(needle: list[str], haystack: list[str]) -> int:
    """Length of the longest contiguous block of the answer that occurs in the context."""
    if not needle or not haystack:
        return 0
    previous = [0] * (len(haystack) + 1)
    best = 0
    for i in range(1, len(needle) + 1):
        current = [0] * (len(haystack) + 1)
        for j in range(1, len(haystack) + 1):
            if needle[i - 1] == haystack[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def sentence_selection(
    answer: list[str], sentences: list[list[str]], coverage: float
) -> tuple[int, int] | None:
    """Greedily cover the answer's distinct content words with context sentences.

    Returns (sentences selected, distance between the first and last), or None when
    the diagnostic does not apply: an answer with no content words to cover, or one
    whose content words appear in no context sentence at all. Those two are not
    "zero sentences sufficed", and averaging them as zero would understate both
    statistics, so the caller drops them and reports how many it dropped.
    """
    target = set(content_words(answer))
    if not target:
        return None
    need = max(1, int(round(coverage * len(target))))
    remaining = set(target)
    chosen: list[int] = []
    while remaining and len(target) - len(remaining) < need:
        best_index, best_gain = None, 0
        for index, sentence in enumerate(sentences):
            if index in chosen:
                continue
            gain = len(remaining & set(sentence))
            if gain > best_gain:
                best_index, best_gain = index, gain
        if best_index is None:
            break
        chosen.append(best_index)
        remaining -= set(sentences[best_index])
    if not chosen:
        return None
    return len(chosen), max(chosen) - min(chosen)


def table_5(rows: list[dict], dataset: str, tokenize, gpt2_tokenize) -> dict:
    # `prompt`, not `modernbert_prompt`, under the ModernBERT tokenizer. On PubMedQA
    # and CLAPNQ those are the same string. On CoQA they are not: the dialogue prompt
    # is 510.9 tokens and its whitespace-normalised form, which is what the masked
    # encoders receive, is 482.8. Table 5 is a description of the dataset rather than
    # of a run, and the dialogue prompt is the dataset's own form, so that is the one
    # measured.
    context = [tokenize(row["source_text"]) for row in rows]
    answer = [tokenize(" " + row["target"].strip()) for row in rows]
    prompt = [tokenize(row["prompt"]) for row in rows]
    question = [
        tokenize(datasets.current_question(row, dataset)) for row in rows
    ]
    sorted_answers = sorted(answer)
    entry = {
        "questions": len(rows),
        "distinct_contexts": len({row["source_text"] for row in rows}),
        "references_per_question": statistics.mean(len(row["references"]) for row in rows),
        "distinct_references_per_question": statistics.mean(
            len({metrics.normalize(r) for r in row["references"]}) for row in rows
        ),
        "context_tokens": statistics.mean(context),
        "context_words": statistics.mean(len(row["source_text"].split()) for row in rows),
        "question_tokens": statistics.mean(question),
        "prompt_tokens": statistics.mean(prompt),
        "answer_tokens": statistics.mean(answer),
        "answer_words": statistics.mean(len(row["target"].split()) for row in rows),
        "answer_median": sorted_answers[len(sorted_answers) // 2],
        "answer_p90": sorted_answers[int(0.9 * len(sorted_answers))],
        "answer_max": max(answer),
        "answer_gpt2_tokens": statistics.mean(
            gpt2_tokenize(" " + row["target"].strip()) for row in rows
        ),
        "sentences_per_answer": statistics.mean(
            len([s for s in SENTENCE.split(row["target"].strip()) if s.strip()])
            for row in rows
        ),
    }
    if dataset == "coqa":
        entry["conversations"] = len(
            {row["metadata"]["conversation_id"] for row in rows}
        )
        # The row Table 5 reports for CoQA only: the prompt minus the context, so it
        # counts the `Q:` / `A:` scaffolding as part of what the prompt adds.
        entry["history_and_question_tokens"] = (
            entry["prompt_tokens"] - entry["context_tokens"]
        )
        # The two parts measured directly, which is 10.7 tokens smaller because it
        # excludes that scaffolding.
        history = [
            tokenize(datasets.dialogue_history(row, dataset))
            if row["metadata"]["history_turns"] else 0
            for row in rows
        ]
        entry["history_plus_question_measured"] = (
            statistics.mean(history) + entry["question_tokens"]
        )
    return entry


def pick_reference(references: list[str], context_tokens: set[str]) -> str:
    """The reference with the highest token overlap with the context."""
    def overlap(reference: str) -> float:
        tokens = metrics.normalize(reference).split()
        if not tokens:
            return -1.0
        return sum(token in context_tokens for token in tokens) / len(tokens)

    return max(references, key=overlap)


def question_overlap(rows: list[dict], dataset: str) -> float:
    """Share of question tokens that occur in the context, after normalization.

    On PubMedQA this is 70.9% (Appendix D.6): the question overlaps its abstract
    heavily even where the answer does not.
    """
    shares = []
    for row in rows:
        context = set(metrics.normalize(row["source_text"]).split())
        question = metrics.normalize(
            datasets.current_question(row, dataset)
        ).split()
        if question:
            shares.append(sum(t in context for t in question) / len(question))
    return 100 * statistics.mean(shares)


def table_6(rows: list[dict], dataset: str, convention: str = "best-overlap") -> dict:
    token_overlap, content_overlap, novel, below_half = [], [], [], []
    spans, runs, selected, distances, ge2, ge3 = [], [], [], [], [], []
    full_context_f1, best_sentence_f1 = [], []
    undefined_selection = 0
    for row in rows:
        context_tokens = metrics.normalize(row["source_text"]).split()
        context_set = set(context_tokens)
        sentences = [
            metrics.normalize(s).split()
            for s in SENTENCE.split(row["source_text"].strip())
            if s.strip()
        ]
        context_bigrams = set(bigrams(context_tokens))

        candidates = (
            row["references"]
            if convention == "per-measure"
            else [pick_reference(row["references"], context_set)]
        )
        best = {}
        for reference in candidates:
            answer_tokens = metrics.normalize(reference).split()
            if not answer_tokens:
                continue
            answer_content = content_words(answer_tokens)
            answer_bigrams = bigrams(answer_tokens)
            measures = {
                "token_overlap": sum(t in context_set for t in answer_tokens) / len(answer_tokens),
                "content_overlap": (
                    sum(t in context_set for t in answer_content) / len(answer_content)
                    if answer_content else 1.0
                ),
                "novel_bigrams": (
                    sum(b not in context_bigrams for b in answer_bigrams) / len(answer_bigrams)
                    if answer_bigrams else 0.0
                ),
                "contiguous": float(contiguous(answer_tokens, context_tokens)),
                "run_ratio": longest_shared_run(answer_tokens, context_tokens) / len(answer_tokens),
            }
            selection = sentence_selection(answer_tokens, sentences, 0.9)
            if selection is not None:
                measures["selected"], measures["distance"] = selection
            # "Most favourable" means larger for the overlap measures and smaller
            # for novelty and for the sentence-selection pair. `selected` and
            # `distance` come from one call and describe one selection, so they are
            # chosen together by `selected` rather than minimized independently.
            for key, value in measures.items():
                if key == "distance":
                    continue
                smaller_is_better = key in ("novel_bigrams", "selected")
                if key not in best:
                    best[key] = value
                elif (value < best[key]) if smaller_is_better else (value > best[key]):
                    best[key] = value
                if key == "selected" and best[key] == value:
                    best["distance"] = measures["distance"]
        token_overlap.append(best["token_overlap"])
        content_overlap.append(best["content_overlap"])
        novel.append(best["novel_bigrams"])
        below_half.append(best["content_overlap"] < 0.5)
        spans.append(best["contiguous"])
        runs.append(best["run_ratio"])
        if "selected" in best:
            selected.append(best["selected"])
            distances.append(best["distance"])
            ge2.append(best["selected"] >= 2)
            ge3.append(best["selected"] >= 3)
        else:
            undefined_selection += 1

        full_context_f1.append(
            metrics.best_reference_f1(row["source_text"], row["references"])
        )
        raw_sentences = [s for s in SENTENCE.split(row["source_text"].strip()) if s.strip()]
        best_sentence_f1.append(
            max(
                (metrics.best_reference_f1(s, row["references"]) for s in raw_sentences),
                default=0.0,
            )
        )
    return {
        "gold_tokens_in_context": 100 * statistics.mean(token_overlap),
        "gold_content_words_in_context": 100 * statistics.mean(content_overlap),
        "novel_bigrams": 100 * statistics.mean(novel),
        "answers_below_half_overlap": 100 * statistics.mean(below_half),
        "contiguous_span": 100 * statistics.mean(spans),
        "longest_run_ratio": 100 * statistics.mean(runs),
        "sentences_selected": statistics.mean(selected),
        "answers_ge2_sentences": 100 * statistics.mean(ge2),
        "answers_ge3_sentences": 100 * statistics.mean(ge3),
        "first_last_distance": statistics.mean(distances),
        "question_tokens_in_context": question_overlap(rows, dataset),
        "copy_full_context_f1": 100 * statistics.mean(full_context_f1),
        "copy_best_sentence_f1": 100 * statistics.mean(best_sentence_f1),
        # Answers the sentence-selection diagnostic does not apply to: no content
        # words, or none of them in any context sentence. Excluded above, counted
        # here, never averaged in as a zero.
        "sentence_selection_undefined": undefined_selection,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=int, choices=(5, 6), default=None)
    parser.add_argument(
        "--reference",
        choices=("best-overlap", "per-measure", "both"),
        default="best-overlap",
        help="how to handle questions with several reference answers",
    )
    arguments = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    gpt2 = AutoTokenizer.from_pretrained(GPT2_TOKENIZER, revision=GPT2_REVISION)
    tokenize = lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
    gpt2_tokenize = lambda text: len(gpt2(text, add_special_tokens=False)["input_ids"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    # With `--table`, keep the other table as it is in the results file.
    destination = RESULTS / "dataset_statistics.json"
    report: dict = {"table_5": {}, "table_6": {}}
    if arguments.table is not None and destination.exists():
        stored = json.loads(destination.read_text())
        report.update({key: stored[key] for key in report if key in stored})
    order = ("clapnq", "pubmedqa", "coqa")
    loaded = {
        dataset: datasets.load_dataset(
            datasets.evaluation_path(dataset), dataset=dataset
        )
        for dataset in order
    }

    if arguments.table in (None, 5):
        for dataset in order:
            report["table_5"][dataset] = table_5(
                loaded[dataset], dataset, tokenize, gpt2_tokenize
            )
        print("Table 5: scale and length of the evaluated subsets")
        fields = [
            ("questions", "Questions scored", "{:.0f}"),
            ("distinct_contexts", "Distinct contexts", "{:.0f}"),
            ("references_per_question", "References per question", "{:.2f}"),
            ("context_tokens", "Context tokens", "{:.1f}"),
            ("context_words", "Context words", "{:.1f}"),
            ("question_tokens", "Question tokens", "{:.1f}"),
            ("prompt_tokens", "Full prompt tokens", "{:.1f}"),
            ("answer_tokens", "Answer tokens", "{:.1f}"),
            ("answer_words", "Answer words", "{:.1f}"),
            ("answer_median", "Answer median", "{:.0f}"),
            ("answer_p90", "Answer 90th percentile", "{:.0f}"),
            ("answer_max", "Answer maximum", "{:.0f}"),
            ("answer_gpt2_tokens", "Answer GPT-2 BPE tokens", "{:.1f}"),
            ("sentences_per_answer", "Sentences per answer", "{:.2f}"),
        ]
        print(f"  {'':28s}" + "".join(f"{DISPLAY[d]:>12s}" for d in order))
        for key, label, form in fields:
            print(f"  {label:28s}" + "".join(
                form.format(report["table_5"][d][key]).rjust(12) for d in order
            ))
        print(f"  {'Dialogue history + question':28s}"
              + "".join(
                  ("---" if "history_and_question_tokens" not in report["table_5"][d]
                   else f"{report['table_5'][d]['history_and_question_tokens']:.1f}").rjust(12)
                  for d in order
              ))

    if arguments.table in (None, 6):
        conventions = (
            ("best-overlap", "per-measure")
            if arguments.reference == "both"
            else (arguments.reference,)
        )
        for convention in conventions:
            report["table_6"][convention] = {
                dataset: table_6(loaded[dataset], dataset, convention) for dataset in order
            }
        primary = conventions[0]
        print(f"\nTable 6: lexical overlap and arrangement of passage material "
              f"(reference convention: {primary})")
        fields = [
            ("gold_tokens_in_context", "Gold tokens in context (%)"),
            ("gold_content_words_in_context", "Gold content words in ctx (%)"),
            ("novel_bigrams", "Novel bigrams (%)"),
            ("question_tokens_in_context", "Question tokens in context (%)"),
            ("answers_below_half_overlap", "Answers < 50% overlap (%)"),
            ("contiguous_span", "Answer is a contiguous span (%)"),
            ("longest_run_ratio", "Longest run / answer length (%)"),
            ("sentences_selected", "Sentences selected (90% target)"),
            ("answers_ge2_sentences", "Answers with >= 2 sentences (%)"),
            ("answers_ge3_sentences", "Answers with >= 3 sentences (%)"),
            ("first_last_distance", "First-to-last sentence distance"),
            ("copy_full_context_f1", "Copy the whole context (F1)"),
            ("copy_best_sentence_f1", "Copy the best sentence (F1)"),
        ]
        print(f"  {'':32s}" + "".join(f"{DISPLAY[d]:>12s}" for d in order))
        for key, label in fields:
            print(f"  {label:32s}" + "".join(
                f"{report['table_6'][primary][d][key]:12.1f}" for d in order
            ))
        if len(conventions) > 1:
            print(f"\n  the same table under the per-measure convention")
            print(f"  {'':32s}" + "".join(f"{DISPLAY[d]:>12s}" for d in order))
            for key, label in fields:
                print(f"  {label:32s}" + "".join(
                    f"{report['table_6']['per-measure'][d][key]:12.1f}" for d in order
                ))

    destination.write_text(json.dumps(report, indent=1) + "\n")
    print("\n-> results/dataset_statistics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
