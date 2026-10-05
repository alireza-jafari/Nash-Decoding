"""The two decoding drivers every dataset's run script calls.

Each dataset has its own run scripts under `experiments/04_qa_benchmark/<dataset>/`,
because the model set, the evidence field and the prompt wording differ between
them. What must *not* differ is the decoding rule, the length budget or the action
set, so those live here and are called by all of them. A dataset-specific file that
wanted to change one of these would have to change this module, where the change is
visible to every dataset at once.

Both drivers write one JSON per example through `nashlib.workqueue`, so a run
resumes after an interruption and several GPUs can drain it together.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import torch

from . import numerics
from .autoregressive import AutoregressiveDecoder
from .masked import MaskedBackbone
from .metrics import best_reference_f1
from .nash import Construction, nash_decode
from .registry import ModelSpec
from .workqueue import WorkQueue, drain, longest_first


def run_autoregressive(
    model: ModelSpec,
    examples: list[dict],
    output_directory: str | Path,
    *,
    device: str = "cuda:0",
    context_limit: int = 1024,
    warmup: int = 3,
    progress_every: int = 100,
    log: Callable[[str], None] = print,
) -> None:
    """Greedy decoding at the oracle budget, one forward pass per emitted token."""
    numerics.configure(device)
    decoder = AutoregressiveDecoder(
        model.hf_id, device=device, revision=model.revision,
        context_limit=context_limit, attn_implementation=model.attention,
    )
    log(
        f"[{model.tag}] {model.hf_id}@{model.revision[:8]} | context {decoder.context_limit} "
        f"| {model.attention} attention | {len(decoder.banned_ids)} banned ids "
        f"| end-of-sequence re-banned every step"
    )
    queue = WorkQueue(output_directory)
    budgets = {example["id"]: decoder.answer_budget(example["target"])
               for example in examples}
    ordered = longest_first(examples, lambda e: budgets[e["id"]])

    # Warm up on the longest examples, which is where the kernels and the allocator
    # are actually exercised; a prefix of the file warms the wrong shapes.
    for _, example in ordered[:warmup]:
        if budgets[example["id"]]:
            decoder.generate(example["prompt"], budgets[example["id"]])

    started = time.time()
    done = 0
    for index, example in drain(queue, ordered):
        budget = budgets[example["id"]]
        generation = decoder.generate(example["prompt"], budget)
        queue.publish(
            index,
            {
                "id": example["id"],
                "text": generation.text.strip(),
                "token_ids": generation.token_ids,
                "budget": budget,
                "calls": generation.forwards,
                "seconds": generation.seconds,
                "prompt_tokens": generation.prompt_tokens,
                "f_ref": best_reference_f1(generation.text.strip(), example["references"]),
            },
        )
        done += 1
        if progress_every and done % progress_every == 0:
            log(f"  {model.tag} {queue.completed()}/{len(examples)} "
                f"({time.time() - started:.0f}s)")
    log(f"[{model.tag}] done: {queue.completed()}/{len(examples)} "
        f"in {time.time() - started:.0f}s -> {output_directory}")


def run_nash(
    model: ModelSpec,
    examples: list[dict],
    output_directory: str | Path,
    *,
    construction: Construction = "all_mask",
    device: str = "cuda:0",
    context_limit: int = 1024,
    batch_size: int = 64,
    complete_masked_slots: bool = False,
    progress_every: int = 25,
    log: Callable[[str], None] = print,
) -> None:
    """Nash decoding to an exact equilibrium, from all-[MASK] or from L2R.

    `complete_masked_slots` is a property of the dataset, not a free choice: pass
    `datasets.COMPLETE_MASKED_SLOTS[dataset]`, which says why.
    """
    numerics.configure(device)
    backbone = MaskedBackbone(
        model.hf_id,
        device=device,
        revision=model.revision,
        context_limit=context_limit,
        batch_size=batch_size,
        allow_mask_action=True,
        attn_implementation=model.attention,
    )
    log(
        f"[{model.tag}] {model.hf_id}@{model.revision[:8]} | construction {construction} "
        f"| context {backbone.max_length} | mask id {backbone.mask_id} playable "
        f"| {len(backbone.banned_ids)} banned ids"
    )
    queue = WorkQueue(output_directory)
    # Tokenize each target once, not once per sort comparison and again per example.
    slot_counts = {example["id"]: len(backbone.encode_answer(example["target"]))
                   for example in examples}
    ordered = longest_first(examples, lambda e: slot_counts[e["id"]])

    started = time.time()
    done = 0
    first = True
    for index, example in drain(queue, ordered):
        slots = slot_counts[example["id"]]
        trajectory = nash_decode(
            backbone,
            example["modernbert_prompt"],
            slots,
            example_id=example["id"],
            construction=construction,
            check_identity=first,
            complete_masked_slots=complete_masked_slots,
        )
        first = False
        record = trajectory.to_dict()
        record["f_ref"] = best_reference_f1(trajectory.text, example["references"])
        queue.publish(index, record)
        done += 1
        if progress_every and done % progress_every == 0:
            log(f"  {model.tag}/{construction} {queue.completed()}/{len(examples)} "
                f"({time.time() - started:.0f}s)")
    log(f"[{model.tag}/{construction}] done: {queue.completed()}/{len(examples)} "
        f"in {time.time() - started:.0f}s -> {output_directory}")


def run_one_shot(
    model: ModelSpec,
    examples: list[dict],
    output_directory: str | Path,
    *,
    device: str = "cuda:0",
    context_limit: int = 1024,
    progress_every: int = 100,
    log: Callable[[str], None] = print,
) -> None:
    """The one-shot masked-LM baseline: every slot from a single forward pass.

    Same protocol as the Nash rows in every other respect -- the dataset's masked
    prompt field, the oracle budget T in the model's own tokenizer, the
    `[CLS] prompt [MASK]*T [SEP]` canvas, structural-token bans, greedy argmax,
    float32 with TF32 off, context 1024 (512 for RoBERTa). The only difference is
    the decoding rule: one call, argmax at all T positions at once, no iteration.
    [MASK] is banned here because every slot must commit a real token.
    """
    numerics.configure(device)
    backbone = MaskedBackbone(
        model.hf_id,
        device=device,
        revision=model.revision,
        context_limit=context_limit,
        allow_mask_action=False,
        attn_implementation=model.attention,
    )
    log(
        f"[{model.tag}] {model.hf_id}@{model.revision[:8]} | one-shot "
        f"| context {backbone.max_length} | mask id {backbone.mask_id} banned "
        f"| {len(backbone.banned_ids)} banned ids"
    )
    queue = WorkQueue(output_directory)
    slot_counts = {example["id"]: len(backbone.encode_answer(example["target"]))
                   for example in examples}
    ordered = longest_first(examples, lambda e: slot_counts[e["id"]])

    started = time.time()
    done = 0
    for index, example in drain(queue, ordered):
        slots = slot_counts[example["id"]]
        before = backbone.forwards
        tick = time.perf_counter()
        if slots:
            prompt_ids = backbone.encode_prompt(example["modernbert_prompt"], slots)
            token_ids = backbone.decode_one_shot(prompt_ids, slots)
            text = backbone.decode_answer(token_ids)
        else:
            token_ids, text = [], ""
        if backbone.device.type == "cuda":
            torch.cuda.synchronize(backbone.device)
        queue.publish(
            index,
            {
                "id": example["id"],
                "text": text,
                "token_ids": token_ids,
                "slots": slots,
                "calls": backbone.forwards - before,
                "seconds": time.perf_counter() - tick,
                "f_ref": best_reference_f1(text, example["references"]),
            },
        )
        done += 1
        if progress_every and done % progress_every == 0:
            log(f"  {model.tag}/one-shot {queue.completed()}/{len(examples)} "
                f"({time.time() - started:.0f}s)")
    log(f"[{model.tag}/one-shot] done: {queue.completed()}/{len(examples)} "
        f"in {time.time() - started:.0f}s -> {output_directory}")
