"""The action set: which vocabulary items a player is allowed to play.

Appendix E.3 of the paper. Exactly one rule applies to every system in the main
tables, autoregressive and masked alike:

    banned = tokenizer.all_special_ids  U  [len(tokenizer), model.config.vocab_size)

that is, the tokenizer's own special ids, plus the untrained embedding rows a
checkpoint has when its embedding matrix is padded past the end of its vocabulary.
No content word, scaffold label or answer-like string is ever banned. The size of
the set is therefore a property of the tokenizer and not a tuning knob; it ranges
from 1 id for GPT-2 to a few hundred for the checkpoints that pad their embedding
matrix past the end of their vocabulary.

Two deliberate consequences:

* Banning end-of-sequence is what makes the length comparison exact. With EOS
  unavailable an autoregressive model cannot stop early, so it emits exactly the
  oracle budget B, matching a masked decoder that fills exactly T = B slots.
* For a masked encoder, [MASK] is *not* banned during max-gap decoding. It is a
  legal action at every position, which is what makes the token gap of Equation 3
  well defined at a position that has not been written yet. It is banned only
  during left-to-right construction, where each step must commit a real token.

`scaffold_label_ids` returns the ids of the prompt's own labels, for a control that
bans them as well. The protocol of the reported tables bans structural tokens only,
for every family.
"""

from __future__ import annotations

from collections.abc import Iterable

SCAFFOLD_LABELS: tuple[str, ...] = ("Question", "Answer", "Summary", "TL;DR")


def structural_token_ids(tokenizer, model_vocab_size: int) -> set[int]:
    """The protocol's action-set complement: specials plus untrained embedding slots."""
    return set(tokenizer.all_special_ids) | set(
        range(len(tokenizer), model_vocab_size)
    )


def scaffold_label_ids(
    tokenizer, labels: Iterable[str] = SCAFFOLD_LABELS
) -> set[int]:
    """Single-token spellings of the prompt's scaffold labels (control runs only)."""
    ids: set[int] = set()
    for label in labels:
        for variant in {label, label.lower(), label.upper()}:
            for form in (variant, " " + variant):
                encoded = tokenizer(form, add_special_tokens=False)["input_ids"]
                if len(encoded) == 1:
                    ids.add(encoded[0])
    return ids


def describe(banned: Iterable[int], tokenizer) -> str:
    """A one-line, reviewable description of an action set."""
    banned = sorted(banned)
    shown = ", ".join(
        f"{i}={tokenizer.convert_ids_to_tokens(i)!r}" for i in banned[:8]
    )
    tail = f", ... (+{len(banned) - 8} more)" if len(banned) > 8 else ""
    return f"{len(banned)} banned ids [{shown}{tail}]"
