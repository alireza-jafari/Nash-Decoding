#!/usr/bin/env python3
"""Run the three coordinate rules on the shared ModernBERT-chat checkpoint.

    python experiments/08_coordinate_rules/run_coordinate_rules.py --dataset clapnq --rule nash
    python experiments/08_coordinate_rules/run_coordinate_rules.py --dataset coqa --all-rules

Appendix H. Everything except the coordinate rule is held fixed, and the things
held fixed are not defaults -- they are choices that would otherwise differ between
the two families, so each is stated here and asserted at run time.

**Prompt.** The instruction of Appendix E with one clause appended:

    {evidence} Question: {question}
    This question is answered completely with evidence from the {passage|abstract}.
    Answer using the {passage|abstract}'s own words.
    Answer:

presented as raw text -- no chat template, no system message, no added special
tokens -- and byte-identical across the three rules. `data/*_grounded.jsonl` holds
these prompts; the three rules read the same file.

**Canvas.** `[CLS] prompt [MASK]xT [SEP]`, with T the oracle answer-length budget in
this checkpoint's own tokenizer (mean 7.5 on CoQA, 48.3 on PubMedQA, 63.8 on
CLAPNQ). The longest prompt-plus-canvas across the three datasets is 1,288 tokens,
well inside the checkpoint's 8,192-token context, so nothing is truncated.

**Action set.** One set for every dataset and every rule: the checkpoint's four
non-mask special ids, plus eleven more -- the three pieces of its answer terminator
(`[/`, `Answer`, `]`), four chat-template markers (`[`, `SYS`, `Response`,
`Question`) and the four newline tokens. That leaves 50,352 actions common to all
three rules, plus [MASK], which only Nash decoding may play.

**Decoding.** Greedy in float32 with TF32 disabled and SDPA attention. The
diffusion rules perform exactly T updates and halt; Nash decoding uses tolerance 0,
halts at the first repeated state, and is capped at max(64, 10T) updates.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import datasets, numerics, registry  # noqa: E402
from nashlib.diffusion import diffusion_decode  # noqa: E402
from nashlib.masked import MaskedBackbone  # noqa: E402
from nashlib.metrics import best_reference_f1  # noqa: E402
from nashlib.nash import nash_decode  # noqa: E402
from nashlib.workqueue import WorkQueue, drain, longest_first  # noqa: E402

GROUNDED = {
    "coqa": "coqa_grounded.jsonl.gz",
    "pubmedqa": "pubmedqa_grounded.jsonl",
    "clapnq": "clapnq_grounded.jsonl",
}
RULES = ("uniform", "confidence", "nash")
ARM_NAME = {"uniform": "random", "confidence": "low_confidence", "nash": "nash"}

# The eleven content-adjacent tokens removed on top of the special ids (Appendix H.3):
# three pieces of the answer terminator, four chat-template markers, and the four
# newline tokens, each with its id in this checkpoint's vocabulary. The fourth newline
# token is " \n" (a space, then a newline).
EXTRA_BANNED = {
    "[/": 32871, "Answer": 32869, "]": 62,                       # answer terminator
    "[": 60, "SYS": 9316, "Response": 9604, "Question": 23433,   # chat-template markers
    "\n": 187, "\n\n": 535, " \n": 2490, "\n\n\n": 2756,          # newline tokens
}
EXPECTED_ACTION_COUNT = 50352


def extra_banned_ids(tokenizer) -> tuple[int, ...]:
    """The eleven ids, requiring each string to tokenize to exactly the id listed."""
    for text, token_id in EXTRA_BANNED.items():
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if encoded != [token_id]:
            raise RuntimeError(
                f"{text!r} tokenizes to {encoded}, not [{token_id}]: this is not the "
                f"checkpoint's tokenizer, and the banned set would differ"
            )
    return tuple(sorted(EXTRA_BANNED.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=registry.DATASETS)
    parser.add_argument("--rule", choices=RULES)
    parser.add_argument("--all-rules", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-root", default=None)
    arguments = parser.parse_args()
    if not arguments.all_rules and not arguments.rule:
        parser.error("give --rule {uniform,confidence,nash} or --all-rules")

    device = f"cuda:{arguments.gpu}"
    numerics.configure(device)
    model = registry.MODERNBERT_CHAT
    examples = datasets.read_jsonl(
        REPOSITORY_ROOT / "data" / GROUNDED[arguments.dataset]
    )
    if arguments.limit:
        examples = examples[: arguments.limit]

    # The ban set is a property of the tokenizer, so read it from the tokenizer
    # rather than loading the model once to reach it and again to use it.
    from transformers import AutoTokenizer

    extra = extra_banned_ids(
        AutoTokenizer.from_pretrained(model.hf_id, revision=model.revision)
    )
    backbone = MaskedBackbone(
        model.hf_id,
        device=device,
        revision=model.revision,
        context_limit=8192,
        allow_mask_action=True,
        extra_banned_ids=extra,
    )
    actions = backbone.model.config.vocab_size - len(backbone.banned_ids) - 1
    print(f"{model.hf_id}: {actions} actions shared by all three rules, "
          f"plus [MASK] for Nash decoding only")
    if actions != EXPECTED_ACTION_COUNT:
        print(f"  NOTE: expected {EXPECTED_ACTION_COUNT} actions (Appendix H.3); this "
              f"tokenizer gives {actions}. The eleven extra banned ids are {extra}.")

    output_root = Path(
        arguments.output_root
        or Path(__file__).resolve().parent / "runs" / arguments.dataset
    )
    slot_counts = {example["id"]: len(backbone.encode_answer(example["target"]))
                   for example in examples}
    ordered = longest_first(examples, lambda e: slot_counts[e["id"]])

    rules = RULES if arguments.all_rules else (arguments.rule,)
    for rule in rules:
        queue = WorkQueue(output_root / ARM_NAME[rule])
        started = time.time()
        for index, example in drain(queue, ordered):
            slots = slot_counts[example["id"]]
            if rule == "nash":
                trajectory = nash_decode(
                    backbone,
                    example["modernbert_prompt"],
                    slots,
                    example_id=example["id"],
                    construction="all_mask",
                    # On, so that Nash decoding always emits T real tokens, like the
                    # two samplers it is compared against.
                    complete_masked_slots=True,
                )
                record = trajectory.to_dict()
            else:
                run = diffusion_decode(
                    backbone,
                    example["modernbert_prompt"],
                    slots,
                    rule=rule,
                    example_id=example["id"],
                )
                record = {
                    "id": run.example_id,
                    "slots": run.slots,
                    # A diffusion rule always performs exactly T updates and stops;
                    # "done" is the status the stored arms use for that.
                    "status": "done",
                    "text": run.text,
                    "token_ids": run.token_ids,
                    "calls": run.forwards,
                    "seconds": run.seconds,
                    "order": run.order,
                }
            record["f_ref"] = best_reference_f1(record["text"], example["references"])
            queue.publish(index, record)
        print(f"  {rule:11s} {queue.completed()}/{len(examples)} "
              f"({time.time() - started:.0f}s) -> {queue.root}")
    print("\nrun `score_coordinate_rules.py` to turn these into Tables 3 and 11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
