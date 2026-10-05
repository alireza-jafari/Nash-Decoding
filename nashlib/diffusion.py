"""Single-token masked-diffusion coordinate rules, for the controlled comparison.

Section 4.3 and Appendix H. Uniform order and confidence order are the two
position-selection rules of the dLLM recipe, run in their single-token setting
(steps = block length = T, so one position is committed per step and no block
structure remains). Both start from the same all-[MASK] canvas as Nash decoding
and write one token per step; the only difference from Nash decoding is which
position is written and whether a written position can be revised:

    uniform      pick a masked position uniformly at random
    confidence   pick the masked position whose predicted token has the highest
                 conditional probability
    nash gap     pick the position with the largest token gap, evaluated at all T
                 positions rather than only the masked ones  -> nashlib.nash

The two diffusion rules freeze a position once it is written, so each performs
exactly T updates and halts. Sampling is replaced by argmax in both, so no method
draws a random token: the uniform rule is random only in which position it fills
next. That randomness is the one place a seed enters any experiment in the paper,
and it is derived from the example id so a re-run reproduces it.

Both rules need one forward pass per step, because every masked position shares a
single canvas: re-masking an already-masked position is the identity.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Literal

import torch

from .masked import MaskedBackbone

CoordinateRule = Literal["uniform", "confidence"]


@dataclass(frozen=True)
class DiffusionRun:
    example_id: str
    slots: int
    text: str
    token_ids: list[int]
    prompt_ids: list[int]
    forwards: int
    seconds: float
    order: list[int]


def seed_for(example_id: str) -> int:
    """The uniform rule's per-example seed: the first 32 bits of sha256(example id).

    The uniform-ordered row of Table 3 is the one place a random number enters a
    reported result, and it is reproducible because the seed is a function of the
    example and of nothing else -- not of the worker, the run or the order examples
    are taken in.
    """
    return int(hashlib.sha256(str(example_id).encode()).hexdigest()[:8], 16)


@torch.inference_mode()
def diffusion_decode(
    backbone: MaskedBackbone,
    prompt: str,
    slots: int,
    *,
    rule: CoordinateRule,
    example_id: str = "",
) -> DiffusionRun:
    """Fill a T-slot canvas one position per step, never revising a written slot."""
    mask = backbone.mask_id
    prompt_ids = backbone.encode_prompt(prompt, slots)
    answer = [mask] * slots
    if slots == 0:
        return DiffusionRun(example_id, 0, "", [], prompt_ids, 0, 0.0, [])

    # The uniform rule is the reference sampler's `remasking="random"`: at every step
    # it draws one uniform number per sequence position on the model's device and
    # fills the masked slot that holds the largest draw. The seed, the generator and
    # the tensor shape together define the order. A device generator seeded with s
    # yields the same stream as `torch.manual_seed(s)` does on that device, which is
    # how the reported runs seeded it.
    generator = torch.Generator(device=backbone.device)
    generator.manual_seed(seed_for(example_id))
    before = backbone.forwards
    _synchronize(backbone.device)
    started = time.perf_counter()
    order: list[int] = []

    for _ in range(slots):
        remaining = [i for i in range(slots) if answer[i] == mask]
        sequence, answer_start = backbone.build_canvas(prompt_ids, answer)
        logits = backbone.forward_logits([sequence])[0].float()
        rows = torch.tensor(
            [answer_start + i for i in remaining], device=backbone.device
        )
        selected = logits[rows]
        backbone.apply_bans(selected)
        # A committed slot must hold a real token, so [MASK] is not a legal write.
        selected[:, mask] = float("-inf")
        probabilities = selected.softmax(-1)
        confidence, best = probabilities.max(-1)
        if rule == "confidence":
            choice = int(confidence.argmax())
        elif rule == "uniform":
            draws = torch.rand(
                (1, len(sequence)), device=backbone.device, generator=generator
            )
            choice = int(draws[0, rows].argmax())
        else:
            raise ValueError(f"unknown coordinate rule {rule!r}")
        position = remaining[choice]
        answer[position] = int(best[choice])
        order.append(position)

    _synchronize(backbone.device)
    elapsed = time.perf_counter() - started
    return DiffusionRun(
        example_id=example_id,
        slots=slots,
        text=backbone.decode_answer(answer),
        token_ids=[int(t) for t in answer],
        prompt_ids=[int(t) for t in prompt_ids],
        forwards=backbone.forwards - before,
        seconds=elapsed,
        order=order,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
