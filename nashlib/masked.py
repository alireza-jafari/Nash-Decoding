"""A masked encoder exposing the full-context conditionals Nash decoding needs.

Section 2.2 of the paper. For a position i we replace x_i with [MASK], keep the
prompt and every other answer token, and read the softmax over the unbanned logits
at that position as an estimate of p(v | x_-i; prompt). One forward pass yields
both the best response and the probability of the token currently played there,
which together are the token gap of Equation 3.

Canvas layout, identical for every masked system and every dataset:

    [CLS]  prompt tokens  answer slots  [SEP]

The prompt is truncated from the left to fit `context_limit` with the answer slots
and the two special tokens reserved, which preserves the question and the
instruction because they sit at the end of the prompt. `context_limit` is further
clipped to what the checkpoint actually accepts, so the ModernBERT-family encoders
run at 1024 and RoBERTa-Large at 512.

RoBERTa is the reason that clip is not simply `max_position_embeddings`. Its config
says 514, but a RoBERTa-family model numbers positions from `padding_idx + 1`, so the
usable context is `max_position_embeddings - (padding_idx + 1)` = 512; feeding the
advertised length overruns the position table and raises on exactly the longest
examples, the ones where prompt truncation makes the sequence land on the limit. On
CoQA that is not an edge case: 733 of the 1,814 prompts are long enough, 40% of
the dataset.

Two bounds are applied, because either alone can be wrong. The positional offset is
computed from the model type, which is what the architecture actually does; the
tokenizer's `model_max_length` is a second opinion, ignored when it holds the
"no limit" sentinel some tokenizers use.

One exactness note, because it is what makes the max-gap scan affordable. Every
position that currently holds [MASK] shares a single input tensor: masking a
position that is already [MASK] is the identity, so one forward pass gives the
conditional at all of them. Positions holding a real token each need their own
row. `scan_gaps` checks the shared-canvas identity rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .policy import structural_token_ids


@dataclass(frozen=True)
class Scan:
    """One leave-one-out scan of every answer position."""

    gaps: list[float]
    best_token: list[int]
    current_probability: list[float]
    forwards: int

    @property
    def nash_gap(self) -> float:
        """G(x) = max_i g_i(x)."""
        return max(self.gaps)

    def argmax_position(self) -> int:
        """The position to update. Ties break to the lowest index, deterministically."""
        target = self.nash_gap
        return min(i for i, gap in enumerate(self.gaps) if gap == target)


class MaskedBackbone:
    """A frozen masked language model used as a conditional estimator.

    Nothing here trains, fine-tunes or adapts the checkpoint. The only inputs are
    the checkpoint name and a pinned revision; the only outputs are conditionals.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        revision: str,
        context_limit: int = 1024,
        batch_size: int = 64,
        allow_mask_action: bool = True,
        extra_banned_ids: tuple[int, ...] = (),
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        # `output_loading_info=True` is not optional here. A checkpoint whose
        # masked-LM head does not match the architecture loads with a randomly
        # initialized head and goes on to produce fluent-looking text, which no
        # output check catches. Refusing to run is the only way to see it.
        self.model, self.loading_info = AutoModelForMaskedLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=torch.float32,
            attn_implementation=attn_implementation,
            output_loading_info=True,
        )
        problems = {
            key: self.loading_info.get(key, [])
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys",
                        "error_msgs")
            if self.loading_info.get(key)
        }
        if problems:
            raise RuntimeError(
                f"{model_name}@{revision} did not load exactly: {problems}. Some "
                f"weights would be randomly initialized, and the model would still "
                f"decode fluent text, so this refuses to run rather than report it."
            )
        self.model = self.model.to(self.device).eval()
        self.mask_id = _require(self.tokenizer.mask_token_id, "mask")
        self.cls_id = _require(self.tokenizer.cls_token_id, "CLS")
        self.sep_id = _require(self.tokenizer.sep_token_id, "SEP")

        positional_limit = getattr(self.model.config, "max_position_embeddings", None)
        if positional_limit is not None:
            positional_limit -= position_id_offset(self.model.config)
        self.max_length = min(
            context_limit,
            *(
                limit
                for limit in (positional_limit, tokenizer_limit(self.tokenizer))
                if limit is not None
            ),
        )
        self.batch_size = batch_size

        banned = structural_token_ids(self.tokenizer, self.model.config.vocab_size)
        banned |= set(extra_banned_ids)
        if allow_mask_action:
            # [MASK] must stay playable, or the gap is undefined at an unwritten slot.
            banned.discard(self.mask_id)
        elif self.mask_id in extra_banned_ids:
            raise ValueError("[MASK] cannot be both banned and required")
        self.banned_ids = sorted(banned)
        self._banned = torch.tensor(self.banned_ids, device=self.device)
        self.forwards = 0

    # ---- tokenization -----------------------------------------------------

    def encode_prompt(self, text: str, answer_slots: int) -> list[int]:
        """Prompt ids, truncated from the left so the canvas fits `max_length`."""
        available = self.max_length - answer_slots - 2
        if available < 1:
            raise ValueError(
                f"{answer_slots} answer slots leave no prompt space at "
                f"max_length {self.max_length}"
            )
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids[-available:]

    def encode_answer(self, text: str) -> list[int]:
        """The oracle budget: |tok(' ' + gold)| in this checkpoint's own tokenizer."""
        if not text.strip():
            return []
        return self.tokenizer(" " + text.strip(), add_special_tokens=False)["input_ids"]

    def decode_answer(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

    # ---- canvas -----------------------------------------------------------

    def build_canvas(
        self,
        prompt_ids: list[int],
        answer_ids: list[int],
        masked_position: int | None = None,
    ) -> tuple[list[int], int]:
        """`[CLS] prompt answer [SEP]`, optionally with one answer slot re-masked.

        Returns the sequence and the index at which the answer starts.
        """
        answer = list(answer_ids)
        if masked_position is not None:
            if not 0 <= masked_position < len(answer):
                raise IndexError("masked answer position out of range")
            answer[masked_position] = self.mask_id
        sequence = [self.cls_id] + list(prompt_ids) + answer + [self.sep_id]
        return sequence, 1 + len(prompt_ids)

    def apply_bans(self, logits: torch.Tensor) -> torch.Tensor:
        """Remove the unplayable actions, in place. Returns the same tensor.

        Callers pass a tensor they own. `scan_gaps` passes the result of advanced
        indexing, which is a copy; `predict_position` copies explicitly, because
        basic indexing would give a view into the model's output.
        """
        if len(self.banned_ids):
            logits[..., self._banned] = float("-inf")
        return logits

    def forward_logits(self, sequences: list[list[int]]) -> torch.Tensor:
        input_ids = torch.tensor(sequences, device=self.device)
        self.forwards += len(sequences)
        return self.model(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids)
        ).logits

    # ---- the conditionals -------------------------------------------------

    @torch.inference_mode()
    def scan_gaps(
        self, prompt_ids: list[int], answer_ids: list[int], *, check_identity: bool = False
    ) -> Scan:
        """g_i(x) at every answer position i, with the best response and p(x_i | x_-i).

        Masked positions are answered by one shared forward pass; filled positions
        are batched `batch_size` at a time.
        """
        slots = len(answer_ids)
        masked = [i for i in range(slots) if answer_ids[i] == self.mask_id]
        filled = [i for i in range(slots) if answer_ids[i] != self.mask_id]
        gaps = [0.0] * slots
        best = [0] * slots
        current = [0.0] * slots
        before = self.forwards

        if masked:
            sequence, answer_start = self.build_canvas(prompt_ids, answer_ids, masked[0])
            if check_identity and len(masked) > 1:
                # The claim being checked is that one forward pass gives the
                # conditional at *every* masked position. Comparing two canvases
                # built by build_canvas would be circular, since both re-mask a
                # position that already holds [MASK]. So compare the shared row
                # against an independently-run forward pass for the last masked
                # position, at the logits.
                #
                # The two passes it costs are a diagnostic, not decoding, so they
                # are not charged to `forwards`: `Calls` is a printed column, and
                # the stored runs did not count them either.
                counted = self.forwards
                separate = self.forward_logits([sequence])[0].float()
                reference, reference_start = self.build_canvas(
                    prompt_ids, answer_ids, masked[-1]
                )
                if sequence != reference or answer_start != reference_start:
                    raise AssertionError(
                        "shared-canvas identity violated: re-masking an already "
                        "masked position changed the input"
                    )
                again = self.forward_logits([reference])[0].float()
                difference = float((separate - again).abs().max())
                if difference != 0.0:
                    raise AssertionError(
                        f"the shared forward pass is not deterministic: two "
                        f"evaluations of the same canvas differ by {difference:.3e}"
                    )
                self.forwards = counted
            logits = self.forward_logits([sequence])[0].float()
            rows = torch.tensor([answer_start + i for i in masked], device=self.device)
            selected = logits[rows]
            self.apply_bans(selected)
            probabilities = selected.softmax(-1)
            top_probability, top_index = probabilities.max(-1)
            held = probabilities.gather(
                1, torch.tensor([[answer_ids[i]] for i in masked], device=self.device)
            ).squeeze(1)
            for row, position in enumerate(masked):
                gaps[position] = float(top_probability[row] - held[row])
                best[position] = int(top_index[row])
                current[position] = float(held[row])

        for offset in range(0, len(filled), self.batch_size):
            positions = filled[offset : offset + self.batch_size]
            sequences, mask_indices = [], []
            for position in positions:
                sequence, answer_start = self.build_canvas(
                    prompt_ids, answer_ids, position
                )
                sequences.append(sequence)
                mask_indices.append(answer_start + position)
            logits = self.forward_logits(sequences)
            rows = torch.arange(len(positions), device=self.device)
            selected = logits[
                rows, torch.tensor(mask_indices, device=self.device)
            ].float()
            self.apply_bans(selected)
            probabilities = selected.softmax(-1)
            top_probability, top_index = probabilities.max(-1)
            held = probabilities.gather(
                1, torch.tensor([[answer_ids[i]] for i in positions], device=self.device)
            ).squeeze(1)
            for row, position in enumerate(positions):
                gaps[position] = float(top_probability[row] - held[row])
                best[position] = int(top_index[row])
                current[position] = float(held[row])

        return Scan(gaps, best, current, self.forwards - before)

    @torch.inference_mode()
    def predict_position(
        self,
        prompt_ids: list[int],
        answer_ids: list[int],
        position: int,
        *,
        forbid_mask: bool = False,
    ) -> tuple[int, float]:
        """Best response at one position, and its probability."""
        sequence, answer_start = self.build_canvas(prompt_ids, answer_ids, position)
        logits = self.forward_logits([sequence])[0]
        # A copy, not a view: apply_bans writes -inf and must not reach the
        # model's own output tensor.
        selected = logits[answer_start + position].float().clone().unsqueeze(0)
        self.apply_bans(selected)
        if forbid_mask:
            selected[0, self.mask_id] = float("-inf")
        probabilities = selected.softmax(-1)
        probability, index = probabilities.max(-1)
        return int(index), float(probability)

    @torch.inference_mode()
    def decode_one_shot(self, prompt_ids: list[int], slots: int) -> list[int]:
        """Fill every answer slot from a single forward pass over the all-[MASK] canvas.

        The one-shot masked-LM baseline of Tables 1 and 7-9: one call, the argmax
        over the unbanned logits at all T positions at once, and no iteration. It is
        the same canvas, budget, bans, precision and context as Nash decoding; only
        the decoding rule differs. [MASK] has to be banned on this backbone
        (`allow_mask_action=False`), because every slot must commit a real token.
        """
        if self.mask_id not in self.banned_ids:
            raise ValueError(
                "one-shot decoding needs [MASK] banned; build the backbone with "
                "allow_mask_action=False"
            )
        if slots == 0:
            return []
        sequence, answer_start = self.build_canvas(prompt_ids, [self.mask_id] * slots)
        logits = self.forward_logits([sequence])[0]
        # A copy, not a view, for the same reason as in predict_position.
        selected = logits[answer_start : answer_start + slots].float().clone()
        self.apply_bans(selected)
        return [int(index) for index in selected.argmax(-1)]


# Model families whose position ids start at padding_idx + 1 rather than 0, so that
# the usable context is shorter than max_position_embeddings advertises.
_POSITION_OFFSET_MODEL_TYPES = frozenset({
    "roberta", "xlm-roberta", "xlm-roberta-xl", "camembert", "longformer",
})

# Some tokenizers use a sentinel in the billions to mean "no limit"; anything that
# large is not a real bound and would only mask the checkpoint's own limit.
_NO_TOKENIZER_LIMIT = 1_000_000


def position_id_offset(config) -> int:
    """How many position-embedding rows a model reserves before position zero."""
    if getattr(config, "model_type", None) not in _POSITION_OFFSET_MODEL_TYPES:
        return 0
    padding_index = getattr(config, "pad_token_id", None)
    return 0 if padding_index is None else int(padding_index) + 1


def tokenizer_limit(tokenizer) -> int | None:
    limit = getattr(tokenizer, "model_max_length", None)
    if limit is None or limit >= _NO_TOKENIZER_LIMIT:
        return None
    return int(limit)


def _require(token_id: int | None, name: str) -> int:
    if token_id is None:
        raise ValueError(f"this checkpoint has no {name} token; it cannot be used here")
    return token_id
