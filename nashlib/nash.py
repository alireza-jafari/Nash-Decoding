"""Algorithm 1: Nash decoding as maximum-gap best response.

    while G(x) > eps:
        compute g_i(x) at every position i
        i* = argmax_i g_i(x)
        x_{i*} <- argmax_v p(v | x_-i*; prompt)

with eps = 0, so the loop stops only at an exact equilibrium: no position can
raise its full-context conditional by playing a different token.

Two things the paper is explicit about and this module therefore is too.

**There are no separate construction and refinement phases under all-mask.** The
gap is recomputed at all T positions after every committed token, so a position
that is already filled competes with an unwritten one and can be revised before
the canvas is complete. "Construction" is just the prefix of updates that happen
to unmask a slot. Left-to-right is a *baseline* that differs only in how the
initial completed sequence is produced: T model calls in index order, no revision
while construction is in progress, and then the identical max-gap refinement.

**The two histories are indexed differently under the two constructions.**
`gap_history` records one entry per *scan*, and left-to-right construction performs
no scans: it writes `slots` entries into `updates_log` before the first scan happens.
So `gap_history[k]` pairs with `updates_log[k]` under all-mask and with
`updates_log[slots + k]` under left-to-right. `Trajectory.to_dict` emits
`gap_history_offset` so a consumer does not have to know which construction produced
a record.

**Two rules bound the loop.** Theorem 1 bounds the number of updates using the joint
probability as an ordinal potential. While Theorem 1 rules out cycles, they may arise
when the true conditional probabilities are replaced by a transformer's estimates,
since these need not be compatible with a common joint distribution (Section 2.2).
The loop therefore halts at the first repeated state, recorded as a limit cycle, and
after max(64, 10T) updates; `status` records which termination applied. Convergence
is nearly universal in the reported runs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

import torch

from .masked import MaskedBackbone

Construction = Literal["all_mask", "l2r"]
# "empty" is a zero-token budget, which is not an equilibrium that was reached: no
# game was played. It is kept distinct so it cannot inflate an equilibrium rate.
# It does not occur in any shipped evaluation set, where every gold answer is
# non-empty in every tokenizer.
Status = Literal["equilibrium", "cycle", "cap", "empty"]


def update_cap(slots: int) -> int:
    """max(64, 10T), the loop bound of Appendix E.6."""
    return max(64, 10 * slots)


@dataclass
class Trajectory:
    """Everything one example's run produced, enough to replay it exactly."""

    example_id: str
    slots: int
    status: Status
    text: str
    token_ids: list[int]
    prompt_ids: list[int]
    updates: int
    unmask_updates: int
    refine_updates: int
    step_canvas_full: int | None
    interleaved: bool
    forwards: int
    forwards_to_full: int | None
    seconds: float
    seconds_to_full: float | None
    final_gap: float
    gap_history: list[float] = field(default_factory=list)
    pseudo_loglik_history: list[float] = field(default_factory=list)
    unmask_flags: list[bool] = field(default_factory=list)
    updates_log: list[list[int]] = field(default_factory=list)
    completed_slots: list[list[int]] = field(default_factory=list)

    @property
    def gap_history_offset(self) -> int:
        """Index into `updates_log` that `gap_history[0]` corresponds to.

        `gap_history` records one entry per scan, and left-to-right construction
        performs no scans: it writes `slots` entries into `updates_log` before the
        first scan happens. So this is 0 under all-mask and `slots` under
        left-to-right. On the equilibrium path there is one more scan than update,
        because the loop scans, sees a zero gap and stops.
        """
        if not self.gap_history:
            return 0
        scans_without_update = 1 if self.status == "equilibrium" else 0
        return len(self.updates_log) - len(self.gap_history) + scans_without_update

    def to_dict(self) -> dict:
        return {
            "id": self.example_id,
            "slots": self.slots,
            "status": self.status,
            "text": self.text,
            "token_ids": self.token_ids,
            "prompt_ids": self.prompt_ids,
            "updates": self.updates,
            "unmask_updates": self.unmask_updates,
            "refine_updates": self.refine_updates,
            # `step_last_mask_filled` is the name the stored artifacts use; the
            # alias is emitted too so a consumer written against either one works.
            "step_last_mask_filled": self.step_canvas_full,
            "step_canvas_full": self.step_canvas_full,
            "interleaved": self.interleaved,
            "gap_history_offset": self.gap_history_offset,
            "calls": self.forwards,
            "calls_to_full": self.forwards_to_full,
            "seconds": self.seconds,
            "seconds_to_full": self.seconds_to_full,
            "final_gap": self.final_gap,
            "traj_gap": self.gap_history,
            "traj_pll": self.pseudo_loglik_history,
            "traj_unmask": self.unmask_flags,
            "traj_upd": self.updates_log,
            "completed_slots": self.completed_slots,
        }


def nash_decode(
    backbone: MaskedBackbone,
    prompt: str,
    slots: int,
    *,
    example_id: str = "",
    construction: Construction = "all_mask",
    gap_tolerance: float = 0.0,
    check_identity: bool = False,
    complete_masked_slots: bool = False,
) -> Trajectory:
    """Decode `slots` tokens by running the token game to an equilibrium.

    `gap_tolerance` is the eps of Algorithm 1 and is 0 everywhere in the paper.
    `complete_masked_slots` fills any slot still holding [MASK] when the loop stops
    early, left to right and with [MASK] removed from the action set, so the system
    emits exactly `slots` real tokens; the passes it costs are counted in `forwards`.
    Without it a slot left masked is dropped by the detokenizer and the answer is
    shorter than its budget.

    Which one applies is a property of the dataset: the CLAPNQ runs use it and the
    CoQA and PubMedQA runs do not. Callers read it from
    `nashlib.datasets.COMPLETE_MASKED_SLOTS`. The coordinate-rule experiment of
    Section 4.3 also turns it on, because the comparison there is against samplers
    that always emit T real tokens.
    """
    if slots == 0:
        return Trajectory(
            example_id=example_id, slots=0, status="empty", text="",
            token_ids=[], prompt_ids=[], updates=0, unmask_updates=0,
            refine_updates=0, step_canvas_full=None, interleaved=False,
            forwards=0, forwards_to_full=None, seconds=0.0, seconds_to_full=None,
            final_gap=0.0,
        )

    mask = backbone.mask_id
    prompt_ids = backbone.encode_prompt(prompt, slots)
    answer = [mask] * slots
    cap = update_cap(slots)

    forwards_before = backbone.forwards
    gap_history: list[float] = []
    pseudo_history: list[float] = []
    unmask_flags: list[bool] = []
    updates_log: list[list[int]] = []
    updates = 0
    status: Status = "cap"
    step_canvas_full: int | None = None
    forwards_to_full: int | None = None
    seconds_to_full: float | None = None
    started = time.perf_counter()

    if construction == "l2r":
        # Left-to-right construction: position i is written from the tokens already
        # committed to its left; positions to its right are still [MASK], so the
        # bidirectional backbone sees how many slots remain but no content there.
        # Nothing is revised while construction is in progress.
        with torch.inference_mode():
            for position in range(slots):
                token, _ = backbone.predict_position(
                    prompt_ids, answer, position, forbid_mask=True
                )
                answer[position] = token
                updates_log.append([position, token])
                unmask_flags.append(True)
                updates += 1
        step_canvas_full = updates
        forwards_to_full = backbone.forwards - forwards_before
        _synchronize(backbone.device)
        seconds_to_full = time.perf_counter() - started

    seen: dict[tuple[int, ...], int] = {tuple(answer): 0}

    with torch.inference_mode():
        while updates < cap:
            scan = backbone.scan_gaps(
                prompt_ids,
                answer,
                check_identity=check_identity and updates == 0,
            )
            gap = scan.nash_gap
            gap_history.append(gap)
            pseudo_history.append(_pseudo_loglikelihood(scan.current_probability))
            if gap <= gap_tolerance:
                status = "equilibrium"
                break
            position = scan.argmax_position()
            was_masked = answer[position] == mask
            token = scan.best_token[position]
            updates_log.append([int(position), int(token)])
            unmask_flags.append(bool(was_masked))
            answer[position] = token
            updates += 1
            if step_canvas_full is None and mask not in answer:
                step_canvas_full = updates
                forwards_to_full = backbone.forwards - forwards_before
                _synchronize(backbone.device)
                seconds_to_full = time.perf_counter() - started
            signature = tuple(answer)
            if signature in seen:
                status = "cycle"
                break
            seen[signature] = updates

    # The gap of the state the loop stopped in. On the equilibrium path the last
    # scan describes the emitted sequence. On a cycle or a cap the loop applied an
    # update after its last scan, so gap_history[-1] describes the state *before*
    # that update; certifying the emitted sequence costs one more scan.
    #
    # Those forward passes are deliberately NOT added to `forwards`. `Calls` is a
    # printed column and means the cost of decoding; charging a diagnostic to it
    # would change the number the tables report.
    final_gap, certification_forwards = _certify(backbone, prompt_ids, answer,
                                                 status, gap_history)

    completed: list[list[int]] = []
    if complete_masked_slots and mask in answer:
        with torch.inference_mode():
            for position in range(slots):
                if answer[position] != mask:
                    continue
                token, _ = backbone.predict_position(
                    prompt_ids, answer, position, forbid_mask=True
                )
                answer[position] = token
                completed.append([position, token])
    if complete_masked_slots and mask in answer:
        raise AssertionError("canvas still holds [MASK] after completion")

    elapsed = time.perf_counter() - started
    unmasked = sum(unmask_flags)
    return Trajectory(
        example_id=example_id,
        slots=slots,
        status=status,
        text=backbone.decode_answer(answer),
        token_ids=[int(t) for t in answer],
        prompt_ids=[int(t) for t in prompt_ids],
        updates=updates,
        unmask_updates=unmasked,
        refine_updates=updates - unmasked,
        step_canvas_full=step_canvas_full,
        interleaved=bool(
            step_canvas_full is not None
            and (updates - unmasked) > 0
            and any(not flag for flag in unmask_flags[:step_canvas_full])
        ),
        forwards=backbone.forwards - forwards_before - certification_forwards,
        forwards_to_full=forwards_to_full,
        seconds=elapsed,
        seconds_to_full=seconds_to_full,
        final_gap=final_gap,
        gap_history=gap_history,
        pseudo_loglik_history=pseudo_history,
        unmask_flags=unmask_flags,
        updates_log=updates_log,
        completed_slots=completed,
    )


def refine_to_equilibrium(
    backbone: MaskedBackbone,
    prompt: str | None,
    initial_answer_ids: list[int],
    *,
    example_id: str = "",
    gap_tolerance: float = 0.0,
    prompt_ids: list[int] | None = None,
) -> Trajectory:
    """Run max-gap refinement from a completed sequence supplied by someone else.

    Used by the initialization study, which starts refinement from sequences that
    Nash decoding did not build (random tokens, a one-shot parallel decode, a
    confidence-ordered construction), and by the WikiText-103 dynamics experiment,
    where the measured trajectory begins at the completed sequence x^(1).

    Pass `prompt_ids` instead of `prompt` when the prompt has already been
    tokenized and truncated -- the WikiText prompts are stored as ids, because
    "exactly 512 tokens" is a property of the ids and not of the text.
    """
    slots = len(initial_answer_ids)
    if slots == 0:
        return Trajectory(
            example_id=example_id, slots=0, status="empty", text="",
            token_ids=[], prompt_ids=[], updates=0, unmask_updates=0,
            refine_updates=0, step_canvas_full=0, interleaved=False,
            forwards=0, forwards_to_full=0, seconds=0.0, seconds_to_full=0.0,
            final_gap=0.0,
        )
    if prompt_ids is None:
        if prompt is None:
            raise ValueError("give either prompt or prompt_ids")
        prompt_ids = backbone.encode_prompt(prompt, slots)
    prompt_ids = list(prompt_ids)
    answer = list(initial_answer_ids)
    cap = update_cap(slots)
    forwards_before = backbone.forwards
    started = time.perf_counter()

    gap_history: list[float] = []
    pseudo_history: list[float] = []
    updates_log: list[list[int]] = []
    seen: dict[tuple[int, ...], int] = {tuple(answer): 0}
    updates = 0
    status: Status = "cap"

    with torch.inference_mode():
        while updates < cap:
            scan = backbone.scan_gaps(prompt_ids, answer)
            gap = scan.nash_gap
            gap_history.append(gap)
            pseudo_history.append(_pseudo_loglikelihood(scan.current_probability))
            if gap <= gap_tolerance:
                status = "equilibrium"
                break
            position = scan.argmax_position()
            token = scan.best_token[position]
            updates_log.append([int(position), int(token)])
            answer[position] = token
            updates += 1
            signature = tuple(answer)
            if signature in seen:
                status = "cycle"
                break
            seen[signature] = updates

    final_gap, certification_forwards = _certify(backbone, prompt_ids, answer,
                                                 status, gap_history)
    elapsed = time.perf_counter() - started
    return Trajectory(
        example_id=example_id,
        slots=slots,
        status=status,
        text=backbone.decode_answer(answer),
        token_ids=[int(t) for t in answer],
        prompt_ids=[int(t) for t in prompt_ids],
        updates=updates,
        unmask_updates=0,
        refine_updates=updates,
        step_canvas_full=0,
        interleaved=False,
        forwards=backbone.forwards - forwards_before - certification_forwards,
        forwards_to_full=0,
        seconds=elapsed,
        seconds_to_full=0.0,
        final_gap=final_gap,
        gap_history=gap_history,
        pseudo_loglik_history=pseudo_history,
        updates_log=updates_log,
    )


def _certify(
    backbone: MaskedBackbone,
    prompt_ids: list[int],
    answer: list[int],
    status: Status,
    gap_history: list[float],
) -> tuple[float, int]:
    """G(x) of the sequence actually emitted, and what it cost to find out.

    Returns the gap and the number of forward passes the check consumed, so the
    caller can exclude them from the reported decoding cost.
    """
    if status == "equilibrium":
        return (gap_history[-1] if gap_history else 0.0), 0
    before = backbone.forwards
    with torch.inference_mode():
        gap = backbone.scan_gaps(prompt_ids, answer).nash_gap
    return gap, backbone.forwards - before


def _pseudo_loglikelihood(current_probability: list[float]) -> float:
    """sum_i log q(x_i | x_-i), the masked model's pseudo-log-likelihood of x.

    This is the quantity Appendix C.5 substitutes for the joint probability when
    checking Equation 5. It is a diagnostic, not a likelihood: the conditionals of
    a masked model need not be the marginals of any common joint distribution.
    """
    return float(sum(math.log(max(p, 1e-45)) for p in current_probability))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
