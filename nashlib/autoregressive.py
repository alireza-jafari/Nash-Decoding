"""Greedy left-to-right decoding at the oracle token budget.

Equation 1 of the paper, run under the shared protocol of Appendix E:

* greedy argmax over the unbanned logits, no temperature, top-k or top-p;
* float32 with TF32 disabled and the SDPA attention implementation;
* context limited to 1024 tokens with one margin token reserved, truncated from
  the left so the question and the instruction survive;
* exactly B = |tok(' ' + gold)| tokens emitted, where B is measured in *this*
  model's tokenizer, with end-of-sequence re-banned at every step so nothing can
  stop early. That is what makes the length comparison against a masked decoder
  filling T = B slots exact rather than approximate.

One forward pass per generated token: a prefill over the prompt, then B-1 cached
single-token steps. The count is asserted, because it is the `Calls` column of
Tables 7-9 for this family.

Loading. `from_pretrained` materializes the float32 weights in host memory before
they are moved to the GPU, which for a 7B model is 28 GB of RAM however much memory
the GPU has. Setting `NASHLIB_LOAD_ON_DEVICE=1` loads each tensor straight onto the
device instead (it needs `accelerate`). The weights, and therefore the outputs, are
the same either way; the switch only decides where they sit while loading.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .masked import position_id_offset, tokenizer_limit
from .policy import structural_token_ids


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: list[int]
    prompt_tokens: int
    forwards: int
    seconds: float


class AutoregressiveDecoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        revision: str,
        context_limit: int = 1024,
        context_margin: int = 1,
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tokenizer.truncation_side = "left"
        config = AutoConfig.from_pretrained(model_name, revision=revision)
        # As in the masked path: a checkpoint that does not load exactly would run
        # with randomly initialized weights and still emit fluent text. Warn rather
        # than refuse here, because a causal checkpoint legitimately reports tied
        # weights as unexpected on some architectures.
        # Off by default: the plain load is what produced the stored outputs. See the
        # module docstring for when a node needs the other one.
        on_device = os.environ.get("NASHLIB_LOAD_ON_DEVICE") == "1"
        placement = (
            {"low_cpu_mem_usage": True, "device_map": {"": str(self.device)}}
            if on_device else {}
        )
        self.model, self.loading_info = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            revision=revision,
            dtype=torch.float32,
            attn_implementation=attn_implementation,
            output_loading_info=True,
            **placement,
        )
        problems = {
            key: self.loading_info.get(key, [])
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys",
                        "error_msgs")
            if self.loading_info.get(key)
        }
        if problems:
            warnings.warn(
                f"{model_name}@{revision} did not load exactly: {problems}. Some "
                f"weights may be randomly initialized; check this before trusting "
                f"the output.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not on_device:
            self.model = self.model.to(self.device)
        self.model = self.model.eval()
        stray = {
            str(parameter.device)
            for parameter in self.model.parameters()
            if parameter.device.type != self.device.type
            or (self.device.index is not None
                and parameter.device.index != self.device.index)
        }
        if stray:
            raise RuntimeError(
                f"{model_name} has weights on {sorted(stray)}; every weight has to "
                f"be on {self.device}"
            )

        text_config = getattr(self.model.config, "text_config", self.model.config)
        positional_limit = getattr(text_config, "max_position_embeddings", None)
        if positional_limit is not None:
            positional_limit -= position_id_offset(text_config)
        bounds = [
            limit
            for limit in (positional_limit, tokenizer_limit(self.tokenizer))
            if limit is not None
        ]
        self.context_limit = min(context_limit, *bounds) if bounds else context_limit
        self.context_margin = context_margin
        vocab_size = int(getattr(text_config, "vocab_size", len(self.tokenizer)))
        self.banned_ids = sorted(structural_token_ids(self.tokenizer, vocab_size))
        self._banned = torch.tensor(
            self.banned_ids, device=self.device, dtype=torch.long
        )
        self.forwards = 0

    def answer_budget(self, gold_answer: str) -> int:
        """B = |tok(' ' + gold)| in this model's own tokenizer."""
        return len(
            self.tokenizer(" " + gold_answer.strip(), add_special_tokens=False)[
                "input_ids"
            ]
        )

    def prompt_length(self, prompt: str) -> int:
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    @torch.inference_mode()
    def generate(self, prompt: str, new_tokens: int) -> Generation:
        if new_tokens <= 0:
            return Generation("", [], self.prompt_length(prompt), 0, 0.0)
        available = self.context_limit - new_tokens - self.context_margin
        if available < 1:
            raise ValueError(
                f"budget {new_tokens} leaves no prompt context at "
                f"context_limit {self.context_limit}"
            )
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=available
        )["input_ids"]

        _synchronize(self.device)
        started = time.perf_counter()
        before = self.forwards

        output = self.model(
            input_ids=torch.tensor([prompt_ids], device=self.device), use_cache=True
        )
        self.forwards += 1
        past = output.past_key_values
        logits = output.logits[:, -1, :].float()

        eos_id = self.tokenizer.eos_token_id
        generated: list[int] = []
        for step in range(new_tokens):
            selection = logits[0].clone()
            if self.banned_ids:
                selection[self._banned] = float("-inf")
            if eos_id is not None:
                # Re-banned every step: nothing may stop before the budget.
                selection[eos_id] = float("-inf")
            token = int(selection.argmax())
            generated.append(token)
            if step + 1 == new_tokens:
                break
            output = self.model(
                input_ids=torch.tensor([[token]], device=self.device),
                past_key_values=past,
                use_cache=True,
            )
            self.forwards += 1
            past = output.past_key_values
            logits = output.logits[:, -1, :].float()

        _synchronize(self.device)
        elapsed = time.perf_counter() - started
        forwards = self.forwards - before
        if len(generated) != new_tokens:
            raise AssertionError(f"emitted {len(generated)} tokens, expected {new_tokens}")
        if forwards != new_tokens:
            raise AssertionError(f"{forwards} forward passes for {new_tokens} tokens")
        if eos_id is not None and eos_id in generated:
            raise AssertionError("end-of-sequence emitted despite the ban")
        text = self.tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return Generation(text, generated, len(prompt_ids), forwards, elapsed)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
