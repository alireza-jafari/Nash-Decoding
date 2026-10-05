#!/usr/bin/env python3
"""Figure 1 and Appendix A: Nash decoding on top of GPT-2's own joint distribution.

    python experiments/01_motivating_example/run_motivating_example.py
    python experiments/01_motivating_example/run_motivating_example.py --example 2

This is the one experiment that uses an *autoregressive* model as the conditional
estimator, and it uses the exact conditional of Equation 8 rather than any
approximation:

    q(v | x_-i, prompt) = p(x_<i, v, x_>i | prompt) / sum_u p(x_<i, u, x_>i | prompt)

The numerator is the joint probability of the whole continuation with position i
replaced by v, so the suffix has to be re-scored for every candidate token. All
50,257 candidates are scored; there is no shortlist and no beam. That is what makes
the example expensive -- and it is exactly the cost argument of Section 2.2 for why
the question-answering experiments use masked encoders instead.

Because these conditionals come from a genuine joint distribution, the guarantees
of Theorem 1 apply here without qualification: every update must raise the sequence
probability, and the log-probabilities printed below do rise monotonically.

The two prompts behave differently. The first takes one update: greedy decoding
writes "Barcelona", and with the six tokens to its right in view the best response
becomes "Madrid". The second takes two, and shows the players influencing one
another: correcting position 4 more than doubles the gap at position 14, from 0.0435
to 0.0972, because the repaired first sentence makes the proper noun a better subject
for the third.

Generation and refinement use the same prompt and the same model. What follows the
question differs between the two examples.

*Example 1: the question, then a space.* With a trailing space the model emits the
sentence behind whitespace filler tokens, and starting with `"The"` rather than
`" The"`. Those are artifacts of where the space was put, and a game played on them
spends its first move replacing `"The"` with `" The"` for a gap of 0.96, a fact about
tokenization and not about Spain. So the generated text is decoded and re-tokenized
as `" " + text`, which is how an answer is tokenized everywhere else in this
repository, and the game is played on those ten tokens.

*Example 2: the question, then a blank line.* The prompt is the question followed by
two newline tokens. The twenty tokens the model emits are the players as emitted;
nothing is re-tokenized, so the first player is `"The"` with no leading space.

`--continuation greedy` decodes the sentence here and plays the game on it;
`--continuation paper` plays it on the sentence as printed in Appendix A. The two give
the same game.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import numerics  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

MODEL = "gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
HALT_BELOW = 1e-9  # the game halts when the Nash gap falls below this

# The conditional table of Appendix A for the first example: at player 4, the same
# five tokens under the autoregressive conditional p(x_4 | x_<4) and under the
# full-context conditional p(x_4 | x_-4).
CONDITIONAL_TABLE = {
    1: {
        "position": 4,
        "tokens": (" Madrid", " Barcelona", " Valencia", " Spain", " Catalonia"),
    }
}

# What follows the question differs between the two examples; see the module
# docstring. `prompt_pieces` are tokenized one by one and concatenated, because
# "\n\n" as a single string is one token and the second example's prompt ends in two
# single-newline tokens.
EXAMPLES = {
    1: {
        "question": "What famous city is the capital of Spain?",
        "prompt_pieces": ("What famous city is the capital of Spain? ",),
        # The generated text is re-tokenized as " " + text before the game is played.
        "retokenize_with_leading_space": True,
        "paper_continuation": " The city of Barcelona is the capital of Spain.",
        "continuation_tokens": 10,
    },
    2: {
        "question": "What city is the political capital of Spain?",
        "prompt_pieces": ("What city is the political capital of Spain?", "\n", "\n"),
        # The players are the tokens the model emits, as emitted.
        "retokenize_with_leading_space": False,
        "paper_continuation": (
            "The answer is Barcelona. The city is the capital of Spain. "
            "It is the capital of Spain."
        ),
        "continuation_tokens": 20,
    },
}


def choose_chunk(device: str, longest_continuation: int = 20) -> int:
    """How many candidate tokens to score per forward pass.

    The logits for one chunk are `chunk x continuation x |V|` floats, which at
    |V| = 50,257 and a 20-token continuation is 4 MB per candidate. Picking this
    blindly is how the run dies two minutes in with an out-of-memory error, so it
    is sized from the device rather than fixed.
    """
    if torch.device(device).type != "cuda":
        return 512
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    bytes_per_candidate = longest_continuation * 50257 * 4
    # Half the free memory, leaving room for the activations and the model itself.
    budget = max(256, int(0.5 * free / bytes_per_candidate))
    return min(4096, budget)


class ExactAutoregressiveGame:
    """GPT-2's full-context conditionals, computed over the entire vocabulary."""

    def __init__(self, device: str, chunk: int = 2048) -> None:
        numerics.configure(device)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )
        self.vocabulary_size = self.model.config.vocab_size
        self.chunk = chunk

    def encode_prompt(self, pieces: tuple[str, ...]) -> list[int]:
        """Token ids of the prompt: each piece tokenized on its own, in order."""
        ids: list[int] = []
        for piece in pieces:
            ids += self.tokenizer(piece, add_special_tokens=False)["input_ids"]
        return ids

    @torch.inference_mode()
    def greedy_continuation(self, prompt_ids: list[int], new_tokens: int) -> list[int]:
        generated: list[int] = []
        for _ in range(new_tokens):
            logits = self.model(
                input_ids=torch.tensor([prompt_ids + generated], device=self.device)
            ).logits[0, -1].float()
            generated.append(int(logits.argmax()))
        return generated

    @torch.inference_mode()
    def log_joint_over_vocabulary(
        self, prompt_ids: list[int], answer: list[int], position: int
    ) -> torch.Tensor:
        """log p(x_<i, v, x_>i | prompt) for every v in the vocabulary."""
        out = torch.empty(self.vocabulary_size, dtype=torch.float32, device=self.device)
        base = prompt_ids + answer
        offset, length = len(prompt_ids), len(answer)
        for start in range(0, self.vocabulary_size, self.chunk):
            end = min(start + self.chunk, self.vocabulary_size)
            sequences = (
                torch.tensor(base, device=self.device)
                .unsqueeze(0)
                .repeat(end - start, 1)
            )
            sequences[:, offset + position] = torch.arange(start, end, device=self.device)
            index = torch.arange(offset, offset + length, device=self.device)
            targets = sequences[:, offset : offset + length]
            logits = self.model(input_ids=sequences).logits[:, index - 1, :].float()
            # log p(target) = logit[target] - logsumexp(logits), which avoids
            # materializing a second tensor the size of the logits. At |V| = 50,257
            # and a chunk of a few thousand that is the difference between fitting
            # on a 48 GB card and not.
            chosen = logits.gather(2, targets.unsqueeze(2)).squeeze(2)
            out[start:end] = (chosen - torch.logsumexp(logits, dim=-1)).sum(1)
            del logits, chosen
        return out

    @torch.inference_mode()
    def sequence_log_probability(self, prompt_ids: list[int], answer: list[int]) -> float:
        sequence = prompt_ids + answer
        offset = len(prompt_ids)
        log_probabilities = torch.log_softmax(
            self.model(input_ids=torch.tensor([sequence], device=self.device))
            .logits[0]
            .float(),
            -1,
        )
        return float(
            sum(
                log_probabilities[offset + j - 1, sequence[offset + j]]
                for j in range(len(answer))
            )
        )

    def scan(self, prompt_ids: list[int], answer: list[int]):
        gaps, best, current = [], [], []
        for position in range(len(answer)):
            conditional = torch.softmax(
                self.log_joint_over_vocabulary(prompt_ids, answer, position), -1
            )
            top_probability, top_index = conditional.max(-1)
            held = float(conditional[answer[position]])
            gaps.append(float(top_probability) - held)
            best.append(int(top_index))
            current.append(held)
        return gaps, best, current


def conditional_table_at(
    game: ExactAutoregressiveGame, number: int, prompt_ids: list[int],
    answer: list[int],
) -> dict | None:
    """The autoregressive and the full-context conditional at one player."""
    specification = CONDITIONAL_TABLE.get(number)
    if specification is None:
        return None
    position = specification["position"] - 1
    full = torch.softmax(
        game.log_joint_over_vocabulary(prompt_ids, answer, position), -1
    )
    with torch.inference_mode():
        prefix = torch.tensor(
            [prompt_ids + answer[:position]], device=game.device
        )
        causal = torch.softmax(game.model(input_ids=prefix).logits[0, -1].float(), -1)

    rows = []
    print(f"\n    conditional table at player {specification['position']}")
    print(f"    {'token':12s} {'p(x|x<i) AR':>14s} {'p(x|x_-i) full':>16s}")
    for token in specification["tokens"]:
        token_id = game.tokenizer(token, add_special_tokens=False)["input_ids"][0]
        rows.append({"token": token, "autoregressive": float(causal[token_id]),
                     "full_context": float(full[token_id])})
        print(f"    {token.strip():12s} {rows[-1]['autoregressive']:14.4f} "
              f"{rows[-1]['full_context']:16.4f}")
    gap = float(full.max() - full[answer[position]])
    print(f"    gap at that player: {gap:.4f}")
    return {"rows": rows, "gap": gap}


def run_example(
    game: ExactAutoregressiveGame, number: int, max_steps: int,
    continuation: str = "greedy",
) -> dict:
    specification = EXAMPLES[number]
    prompt_ids = game.encode_prompt(specification["prompt_pieces"])
    prompt_text = "".join(specification["prompt_pieces"])
    slots = specification["continuation_tokens"]

    if continuation == "paper":
        answer = game.tokenizer(
            specification["paper_continuation"], add_special_tokens=False
        )["input_ids"]
    elif specification["retokenize_with_leading_space"]:
        # Generate a little past the budget, because the trailing-space prompt emits
        # whitespace filler tokens that carry no player.
        raw = game.greedy_continuation(prompt_ids, slots + 6)
        text = game.tokenizer.decode(raw)
        # Re-tokenize as " " + text, the way an answer is tokenized everywhere else
        # here, so the players are the sentence's tokens rather than the
        # generation's whitespace layout.
        answer = game.tokenizer(
            " " + " ".join(text.split()), add_special_tokens=False
        )["input_ids"][:slots]
    else:
        # The players are the emitted tokens themselves.
        answer = game.greedy_continuation(prompt_ids, slots)
    greedy_text = game.tokenizer.decode(answer)
    print(f"\n=== example {number} ({continuation} continuation) ===")
    print(f"question            {specification['question']!r}")
    print(f"prompt              {prompt_text!r}  ({len(prompt_ids)} tokens)")
    print(f"continuation        {greedy_text!r}  ({len(answer)} players)")

    conditional_table = conditional_table_at(game, number, prompt_ids, answer)
    log_probabilities = [game.sequence_log_probability(prompt_ids, answer)]
    steps = []
    scans = []
    seen = {tuple(answer)}
    status = "cap"
    started = time.time()

    for step in range(1, max_steps + 1):
        gaps, best, current = game.scan(prompt_ids, answer)
        gap = max(gaps)
        position = min(i for i, value in enumerate(gaps) if value == gap)
        # Every player's view at this scan: p(current), the best response and its
        # probability, and the gap.
        scans.append(
            {
                "step": step,
                "nash_gap": gap,
                "players": [
                    {
                        "position": i + 1,
                        "token": game.tokenizer.decode([answer[i]]),
                        "p_current": current[i],
                        "best": game.tokenizer.decode([best[i]]),
                        "p_best": current[i] + gaps[i],
                        "gap": gaps[i],
                    }
                    for i in range(len(answer))
                ],
            }
        )
        ranked = sorted(range(len(answer)), key=lambda i: -gaps[i])[:3]
        print(f"\n--- step {step} ---  G(x) = {gap:.4f}   ({time.time() - started:.0f}s)")
        for i in ranked:
            if gaps[i] <= 0:
                continue
            print(
                f"    player {i + 1:>2}: {game.tokenizer.decode([answer[i]])!r} "
                f"(p {current[i]:.4f}) -> {game.tokenizer.decode([best[i]])!r} "
                f"(p {current[i] + gaps[i]:.4f}), gap {gaps[i]:.4f}"
            )
        if gap <= HALT_BELOW:
            status = "equilibrium"
            print("    equilibrium: every player already plays its best response")
            break
        replaced = game.tokenizer.decode([answer[position]])
        answer[position] = best[position]
        log_probabilities.append(game.sequence_log_probability(prompt_ids, answer))
        steps.append(
            {
                "step": step,
                "nash_gap": gap,
                "position": position + 1,
                "from": replaced,
                "to": game.tokenizer.decode([best[position]]),
                "log_probability": log_probabilities[-1],
                "ratio": float(
                    __import__("math").exp(log_probabilities[-1] - log_probabilities[-2])
                ),
            }
        )
        print(f"    update player {position + 1}: {replaced!r} -> "
              f"{game.tokenizer.decode([best[position]])!r}")
        print(f"    text: {game.tokenizer.decode(answer)!r}")
        print(f"    log p: {log_probabilities[-2]:.4f} -> {log_probabilities[-1]:.4f} "
              f"(x{steps[-1]['ratio']:.4f})")
        if tuple(answer) in seen:
            status = "cycle"
            break
        seen.add(tuple(answer))

    final_text = game.tokenizer.decode(answer)
    total_ratio = float(
        __import__("math").exp(log_probabilities[-1] - log_probabilities[0])
    )
    print(f"\nfinal    {final_text!r}")
    print(f"status   {status}")
    print(f"log p    {log_probabilities[0]:.4f} -> {log_probabilities[-1]:.4f} "
          f"(x{total_ratio:.4f} overall)")
    monotone = all(
        log_probabilities[i + 1] > log_probabilities[i]
        for i in range(len(log_probabilities) - 1)
    )
    print(f"sequence probability rose at every update: {'yes' if monotone else 'NO'}")

    return {
        "example": number,
        "continuation_source": continuation,
        "question": specification["question"],
        "prompt": prompt_text,
        "prompt_ids": prompt_ids,
        "greedy": greedy_text,
        "equilibrium": final_text,
        "status": status,
        "log_probabilities": log_probabilities,
        "total_ratio": total_ratio,
        "steps": steps,
        "scans": scans,
        "monotone": monotone,
        "conditional_table": conditional_table,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=int, choices=(1, 2), help="run one example")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=0,
                        help="candidate tokens scored per forward pass; 0 picks a "
                             "size that fits the device")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--continuation", choices=("greedy", "paper", "both"),
                        default="greedy",
                        help="decode the continuation here (the default), play the "
                             "game on the sentence as printed in Appendix A, or do both")
    arguments = parser.parse_args()

    chunk = arguments.chunk or choose_chunk(arguments.device)
    game = ExactAutoregressiveGame(arguments.device, chunk=chunk)
    print(f"model {MODEL}@{REVISION[:8]}  |V| = {game.vocabulary_size}  "
          f"device {arguments.device}  float32, TF32 disabled  "
          f"chunk {chunk} candidates/pass")

    numbers = (arguments.example,) if arguments.example else (1, 2)
    sources = (
        ("greedy", "paper") if arguments.continuation == "both"
        else (arguments.continuation,)
    )
    records = [
        run_example(game, number, arguments.max_steps, source)
        for source in sources
        for number in numbers
    ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "motivating_example.json").write_text(
        json.dumps(records, indent=1) + "\n"
    )
    print(f"\n-> results/motivating_example.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
