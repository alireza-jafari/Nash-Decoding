#!/usr/bin/env python3
"""Produce the WikiText-103 trajectories that `analyse_dynamics.py` reads.

    python experiments/03_wikitext_dynamics/run_wikitext.py --gpu 0
    python experiments/03_wikitext_dynamics/run_wikitext.py --gpu 0 --limit 5

Appendix C. The task is continuation, not question answering, so that what is
measured is the decoding dynamics rather than the difficulty of a downstream task.
500 prompts of exactly 512 tokens from the WikiText-103 test split; ModernBERT-Large
generates a 64-token continuation of each.

Each trajectory has two stages, and only the second is the measured one.

**Construction** fills the 64-slot canvas one slot per model call, never revising: at
each of 64 calls, the masked position whose predicted token carries the highest
conditional probability is committed, with [MASK] excluded so every commitment writes
a real token. The completed sequence that pass produces is x^(1).

**Refinement** then runs Algorithm 1 with tolerance 0 from x^(1): recompute the gap
at all 64 positions, rewrite the largest-gap position, stop at G(x) <= 0. The
iteration index k counts refinement steps, which is why both panels of Figure 2
begin at the moment the last [MASK] is filled.

The action set is the full vocabulary minus the four non-mask special ids, leaving
50,364 actions, and [MASK] is one of them. Keeping it legal is what makes the token
gap well defined at a position that has not been written yet, and it lets a player
return its position to the masked state if that is its best response. Across all
1,300 recorded updates, no player ever does.

Two rules bound the loop: it stops after max(64, 10T) = 640 updates, and at the
first repeated state, which is recorded as a limit cycle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import numerics  # noqa: E402
from nashlib.masked import MaskedBackbone  # noqa: E402
from nashlib.nash import refine_to_equilibrium, update_cap  # noqa: E402

PROMPTS = REPOSITORY_ROOT / "data" / "wikitext512_500.json"
RESULTS = Path(__file__).resolve().parent / "results"

MODEL = "answerdotai/ModernBERT-large"
REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
SLOTS = 64
PROMPT_TOKENS = 512


def construct_by_confidence(
    backbone: MaskedBackbone, prompt_ids: list[int], slots: int
) -> tuple[list[int], int]:
    """Commit the highest-confidence masked position, one per call, until full.

    [MASK] is excluded from the action set here, so each of the `slots` calls writes
    a real token. This is the construction pass of Appendix C.2, and it is *not*
    part of the measured trajectory.
    """
    answer = [backbone.mask_id] * slots
    before = backbone.forwards
    with torch.inference_mode():
        for _ in range(slots):
            remaining = [i for i in range(slots) if answer[i] == backbone.mask_id]
            sequence, answer_start = backbone.build_canvas(prompt_ids, answer)
            logits = backbone.forward_logits([sequence])[0].float()
            rows = torch.tensor(
                [answer_start + i for i in remaining], device=backbone.device
            )
            selected = logits[rows]
            backbone.apply_bans(selected)
            selected[:, backbone.mask_id] = float("-inf")
            probabilities = selected.softmax(-1)
            confidence, best = probabilities.max(-1)
            choice = int(confidence.argmax())
            answer[remaining[choice]] = int(best[choice])
    return answer, backbone.forwards - before


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--slots", type=int, default=SLOTS)
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()
    device = arguments.device or f"cuda:{arguments.gpu}"

    payload = json.loads(PROMPTS.read_text())
    records = payload["records"]
    if arguments.limit:
        records = records[: arguments.limit]
    print(f"{payload['dataset']}: {len(records)} prompts of "
          f"{payload['prompt_tokens']} tokens, {arguments.slots} continuation slots")

    numerics.configure(device)
    backbone = MaskedBackbone(
        MODEL,
        device=device,
        revision=REVISION,
        context_limit=PROMPT_TOKENS + arguments.slots + 2,
        allow_mask_action=True,
    )
    print(f"action set: {backbone.model.config.vocab_size - len(backbone.banned_ids)} "
          f"actions ([MASK] playable, {len(backbone.banned_ids)} banned)")
    print(f"update cap: {update_cap(arguments.slots)}")

    output = []
    started = time.time()
    for index, record in enumerate(records):
        # The prompts are stored as token ids, because "exactly 512 tokens" is a
        # property of the ids; re-tokenizing the text would not guarantee it.
        prompt_ids = record["prompt_ids"]
        if len(prompt_ids) != payload["prompt_tokens"]:
            raise AssertionError(
                f"prompt {record['id']} has {len(prompt_ids)} ids, expected "
                f"{payload['prompt_tokens']}"
            )
        initial, construction_calls = construct_by_confidence(
            backbone, prompt_ids, arguments.slots
        )
        trajectory = refine_to_equilibrium(
            backbone,
            None,
            initial,
            # The id keeps the type the prompt file gives it. The referee shards are
            # keyed by the same integer ids, and a stringified id would make every
            # lookup in analyse_dynamics.py miss without raising.
            example_id=record.get("id", index),
            prompt_ids=prompt_ids,
        )
        entry = trajectory.to_dict()
        entry["init_ids"] = initial
        entry["init_text"] = backbone.decode_answer(initial)
        entry["init_calls"] = construction_calls
        entry["calls"] = entry["calls"] + construction_calls
        output.append(entry)
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(records)}  {time.time() - started:.0f}s")

    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = Path(arguments.output or (RESULTS / "refinement_trajectories.json"))
    destination.write_text(json.dumps(output))
    equilibria = sum(1 for e in output if e["status"] == "equilibrium")
    print(f"\n{len(output)} trajectories, {equilibria} reached an exact equilibrium, "
          f"{sum(1 for e in output if e['status'] == 'cycle')} stopped at a repeated state")
    print(f"-> {destination}")
    print("run `analyse_dynamics.py` to turn these into Table 4 and Figure 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
