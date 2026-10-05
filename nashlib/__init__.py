"""Shared implementation of Nash decoding and the shared evaluation protocol.

The package is deliberately small and has no state of its own: every experiment
under `experiments/` imports from here so that the decoding rule, the banned-token
policy, the length budget and the metrics are literally the same code in every run.

Modules
-------
numerics        float32 / TF32-off numerics, asserted rather than assumed
policy          the banned-token policy (structural tokens only)
masked          a masked encoder exposing leave-one-out full-context conditionals
nash            Algorithm 1: max-gap best response, and the two constructions
autoregressive  greedy left-to-right decoding at an oracle token budget
diffusion       single-token masked-diffusion coordinate rules (uniform, confidence)
runner          the decoding drivers: autoregressive, one-shot masked, Nash
metrics         token F1, best-reference F1, official CoQA F1, ROUGE-L / ROUGE-Lsum
datasets        the evaluation JSONL schema and its loader
registry        which models belong to which dataset, with pinned revisions
workqueue       an atomic claim-file queue so several GPUs can drain one run
tables          LaTeX and Markdown rendering for the paper's tables
"""

__version__ = "1.0.0"

PROTOCOL = {
    "decoding": "greedy",
    "dtype": "float32",
    "tf32": False,
    # SDPA everywhere except BLOOM-3B, which transformers has no SDPA kernel for;
    # each model's own `ModelSpec.attention` is authoritative, and this is the default
    # it almost always holds.
    "attn_implementation": "sdpa, except where ModelSpec.attention says otherwise",
    "context_limit": 1024,
    "context_margin": 1,
    "gap_tolerance": 0.0,
    "length_policy": "oracle",
    "banned_tokens": "structural only (all_special_ids | [len(tokenizer), vocab_size))",
}
