"""Which models belong to which dataset, and at which pinned revision.

Two separate concerns, kept separate on purpose.

**Identity.** `AUTOREGRESSIVE` and `MASKED_ENCODERS` map an internal tag to a
checkpoint and a commit SHA. Checkpoints are pinned to explicit revisions rather
than to branch names, so a re-run downloads the same weights; combined with greedy
decoding this is what makes a run reproduce character for character.

**Membership.** The set of models evaluated on a dataset is *not* the same on the
three datasets: Table 7 (CoQA) reports 14 autoregressive baselines, Table 8
(PubMedQA) 18 and Table 9 (CLAPNQ) 16, and the subsets are not nested. Membership
is read from `paper/paper_values.json`, which holds the paper's tables in
machine-readable form. A tag that appears in a table but has no entry below is a
hard error at import time.

The three masked encoders of the result tables appear twice each: once in the
one-shot block, where every canvas slot is filled from a single forward pass, and
once in the Nash block, where the same frozen checkpoint is decoded to an
equilibrium. The two families share `MASKED_ENCODERS` and differ only in the
decoding rule.

CoQA appears twice. `coqa` is the 1,814-turn cohort of Table 1 and the left block
of Table 7; `coqa_full` is all 7,983 development turns, its right block. Both
blocks print the same rows, so both dataset names read the same table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PAPER_VALUES_PATH = REPOSITORY_ROOT / "paper" / "paper_values.json"

# The three benchmarks, as the tables name them. `coqa_full` is the second cohort
# of the same benchmark rather than a fourth benchmark, so it is kept out of this
# tuple and listed separately; anything that loops over "the datasets" means these
# three, and anything that means "every scored evaluation set" uses SCORED_SETS.
DATASETS = ("coqa", "pubmedqa", "clapnq")
SCORED_SETS = ("coqa", "coqa_full", "pubmedqa", "clapnq")
FAMILIES = ("oneshot", "autoregressive", "nash")


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    hf_id: str
    revision: str
    display: str
    parameters: str
    family: str
    notes: str = ""
    # Every checkpoint runs under SDPA except the ones transformers has no SDPA
    # kernel for. Carrying it on the spec rather than at the call site means a
    # caller cannot forget it for one runner and remember it for another.
    attention: str = "sdpa"


def _spec(tag, hf_id, revision, display, parameters, family, notes="", attention="sdpa"):
    return ModelSpec(tag, hf_id, revision, display, parameters, family, notes, attention)


# --------------------------------------------------------------------------
# Autoregressive decoders. The three tables report 23 distinct models between them;
# the ten others defined here (falcon3_7b, llama2_7b, llama3_8b, qwen15_7b, qwen25_3b,
# qwen25_7b, qwen2_7b, redpajama_7b, yi15_9b, yi_6b) appear in no table and have no
# stored predictions. Which models a given dataset reports is decided by the paper
# tables, not by this dictionary.
# --------------------------------------------------------------------------
AUTOREGRESSIVE: dict[str, ModelSpec] = {
    s.tag: s
    for s in [
        _spec("gpt2_small",     "gpt2",                                     "607a30d783dfa663caf39e06633721c8d4cfcd7e", "GPT-2 Small",      "124M", "autoregressive"),
        _spec("smollm2_135m",   "HuggingFaceTB/SmolLM2-135M",               "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", "SmolLM2-135M",     "135M", "autoregressive"),
        _spec("pythia_160m",    "EleutherAI/pythia-160m",                   "50f5173d932e8e61f858120bcb800b97af589f46", "Pythia-160M",      "160M", "autoregressive"),
        _spec("opt_350m",       "facebook/opt-350m",                        "08ab08cc4b72ff5593870b5d527cf4230323703c", "OPT-350M",         "331M", "autoregressive"),
        _spec("gpt2_medium",    "gpt2-medium",                              "6dcaa7a952f72f9298047fd5137cd6e4f05f41da", "GPT-2 Medium",     "355M", "autoregressive"),
        _spec("smollm2_360m",   "HuggingFaceTB/SmolLM2-360M",               "f8027fd0eaeea54caa13c31d31b9fdc459c38b49", "SmolLM2-360M",     "362M", "autoregressive"),
        _spec("pythia_410m",    "EleutherAI/pythia-410m",                   "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f", "Pythia-410M",      "410M", "autoregressive"),
        _spec("qwen25_0p5b",    "Qwen/Qwen2.5-0.5B",                        "060db6499f32faf8b98477b0a26969ef7d8b9987", "Qwen2.5",          "0.5B", "autoregressive"),
        _spec("gpt2_large",     "gpt2-large",                               "32b71b12589c2f8d625668d2335a01cac3249519", "GPT-2 Large",      "774M", "autoregressive"),
        _spec("gpt2_xl",        "gpt2-xl",                                  "15ea56dee5df4983c59b2538573817e1667135e2", "GPT-2 XL",         "1.5B", "autoregressive"),
        _spec("bloom_3b",       "bigscience/bloom-3b",                      "52bc5b43010b4844513826b8be3f78c7344c37d7", "BLOOM-3B",         "3B",   "autoregressive",
              "transformers ships no SDPA kernel for this architecture, so it runs "
              "under the eager attention implementation; every other model uses SDPA",
              attention="eager"),
        _spec("qwen25_3b",      "Qwen/Qwen2.5-3B",                          "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b", "Qwen2.5",          "3B",   "autoregressive"),
        _spec("danube3_4b",     "h2oai/h2o-danube3-4b-base",                "6bdf2f1e317143c998b88d9e9d72facc621a863f", "Danube3-4B",       "4B",   "autoregressive"),
        _spec("yi_6b",          "01-ai/Yi-6B",                              "80080be87ec5a0103f643195f2d9003b8068941b", "Yi",               "6B",   "autoregressive"),
        _spec("pythia_69b",     "EleutherAI/pythia-6.9b",                   "c0e3eee36dc47af0c49f361c74cfe459c09f7f23", "Pythia",           "6.9B", "autoregressive"),
        _spec("olmo_17_7b",     "allenai/OLMo-1.7-7B-hf",                   "e3c82a0871580b424d74e68acee866bb007df7df", "OLMo-1.7",         "7B",   "autoregressive"),
        _spec("llama1_7b",      "huggyllama/llama-7b",                      "4782ad278652c7c71b72204d462d6d01eaaf7549", "Llama 1",          "7B",   "autoregressive"),
        _spec("llama2_7b",      "NousResearch/Llama-2-7b-hf",               "8efe6c9b93655b934e27bd9981e3ec13e55aee9d", "Llama 2",          "7B",   "autoregressive"),
        _spec("openllama_7b",   "openlm-research/open_llama_7b",            "6fb184ff23774c25bf84b3628e49c8b78372c7be", "OpenLLaMA",        "7B",   "autoregressive"),
        _spec("redpajama_7b",   "togethercomputer/RedPajama-INCITE-7B-Base","78f7e482443971f4873ba3239f0ac810a367833b", "RedPajama-INCITE", "7B",   "autoregressive"),
        _spec("amber_7b",       "LLM360/Amber",                             "2d35937a32b99aec0d59dd772b54e1bb08719a7f", "Amber",            "7B",   "autoregressive"),
        _spec("granite_7b",     "ibm-granite/granite-7b-base",              "3694001fa74a2aa9a6eff07774cbcb68b73e78f2", "Granite",          "7B",   "autoregressive"),
        _spec("qwen15_7b",      "Qwen/Qwen1.5-7B",                          "831096e3a59a0789a541415da25ef195ceb802fe", "Qwen1.5",          "7B",   "autoregressive"),
        _spec("qwen2_7b",       "Qwen/Qwen2-7B",                            "453ed1575b739b5b03ce3758b23befdb0967f40e", "Qwen2",            "7B",   "autoregressive"),
        _spec("qwen25_7b",      "Qwen/Qwen2.5-7B",                          "d149729398750b98c0af14eb82c78cfe92750796", "Qwen2.5",          "7B",   "autoregressive"),
        _spec("mistral_7b",     "mistralai/Mistral-7B-v0.1",                "27d67f1b5f57dc0953326b2601d68371d40ea8da", "Mistral v0.1",     "7.2B", "autoregressive"),
        _spec("mistral_7b_v03", "mistralai/Mistral-7B-v0.3",                "caa1feb0e54d415e2df31207e5f4e273e33509b1", "Mistral v0.3",     "7.2B", "autoregressive"),
        _spec("falcon_7b",      "tiiuae/falcon-7b",                         "ec89142b67d748a1865ea4451372db8313ada0d8", "Falcon",           "7.2B", "autoregressive"),
        _spec("olmo2_7b",       "allenai/OLMo-2-1124-7B",                   "7df9a82518afdecae4e8c026b27adccc8c1f0032", "OLMo-2",           "7.3B", "autoregressive"),
        _spec("falcon3_7b",     "tiiuae/Falcon3-7B-Base",                   "bf3d7ed586cb22a921520e2d681a9d3d7642cde8", "Falcon3",          "7.5B", "autoregressive"),
        _spec("llama3_8b",      "NousResearch/Meta-Llama-3-8B",             "315b20096dc791d381d514deb5f8bd9c8d6d3061", "Llama 3",          "8B",   "autoregressive"),
        _spec("llama31_8b",     "NousResearch/Meta-Llama-3.1-8B",           "1f47e50cdbe801ad8a5174156ec3a0655108fb9f", "Llama 3.1",        "8B",   "autoregressive"),
        _spec("yi15_9b",        "01-ai/Yi-1.5-9B",                          "80d5471b1eae28beae33e06eadbd4b48e74d4ce1", "Yi-1.5",           "9B",   "autoregressive"),
    ]
}

# --------------------------------------------------------------------------
# Masked encoder backbones for Nash decoding. Frozen: no training, no fine-tuning.
# --------------------------------------------------------------------------
MASKED_ENCODERS: dict[str, ModelSpec] = {
    s.tag: s
    for s in [
        _spec("modernbert_base",  "answerdotai/ModernBERT-base",   "8949b909ec900327062f0ebf497f51aef5e6f0c8", "ModernBERT-Base",  "149M", "nash"),
        _spec("ettin_150m",       "jhu-clsp/ettin-encoder-150m",   "45d08642849e5c5701b162671ac811b7654bfd9f", "Ettin-150m",       "150M", "nash"),
        _spec("mmbert_base",      "jhu-clsp/mmBERT-base",          "c5955035435e2bf121cde7f3c8863ef52ff35d82", "mmBERT-base",      "308M", "nash"),
        _spec("roberta_large",    "FacebookAI/roberta-large",      "722cf37b1afa9454edce342e7895e588b6ff1d59", "RoBERTa-Large",    "355M", "nash",
              "accepts 512 tokens, not the 514 its config advertises, so this "
              "backbone runs at context 512 rather than 1024"),
        _spec("modernbert_large", "answerdotai/ModernBERT-large",  "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13", "ModernBERT-Large", "395M", "nash"),
        _spec("ettin_400m",       "jhu-clsp/ettin-encoder-400m",   "7662476d60abb071a5bd319c9f3074f3072c062d", "Ettin-400m",       "396M", "nash"),
    ]
}

# The instruction-tuned masked-diffusion checkpoint of Section 4.3. It keeps the
# ModernBERT-Large architecture but is supervised fine-tuned for masked-diffusion
# generation, so it is deliberately NOT one of the frozen backbones above.
#
# The hub repository publishes no tagged release, but it has commits like any other,
# and this is the one the reported runs loaded: it is the only snapshot in the cache
# those runs used, and what `main` resolved to when they were made.
MODERNBERT_CHAT = _spec(
    "modernbert_large_chat",
    "dllm-hub/ModernBERT-Large-chat-v0.1",
    "b4ca339e8d572cfa74433b65578fce3ef4904567",
    "ModernBERT-Large-chat",
    "395M",
    "nash",
    "supervised fine-tuned for masked-diffusion generation; used only in the "
    "coordinate-rule comparison of Section 4.3",
)

ALL_SPECS: dict[str, ModelSpec] = {
    **AUTOREGRESSIVE,
    **MASKED_ENCODERS,
    MODERNBERT_CHAT.tag: MODERNBERT_CHAT,
}


def spec(tag: str) -> ModelSpec:
    if tag not in ALL_SPECS:
        raise KeyError(f"unknown model tag {tag!r}")
    return ALL_SPECS[tag]


# --------------------------------------------------------------------------
# Membership, read from the paper's tables.
# --------------------------------------------------------------------------

_TABLE_FOR_DATASET = {
    "coqa": "table_7_coqa",
    # Table 7's two blocks print one row list, so the cohorts share a table.
    "coqa_full": "table_7_coqa",
    "pubmedqa": "table_8_pubmedqa",
    "clapnq": "table_9_clapnq",
}


def _load_paper_values() -> dict:
    with PAPER_VALUES_PATH.open() as handle:
        return json.load(handle)


def _build_membership() -> dict[str, dict[str, tuple[str, ...]]]:
    values = _load_paper_values()
    membership: dict[str, dict[str, tuple[str, ...]]] = {}
    unknown: list[str] = []
    for dataset, table in _TABLE_FOR_DATASET.items():
        blocks = values[table]["blocks"]
        membership[dataset] = {}
        for family in FAMILIES:
            tags = tuple(blocks[family].keys())
            for tag in tags:
                if tag not in ALL_SPECS:
                    unknown.append(f"{dataset}/{family}/{tag}")
            membership[dataset][family] = tags
    if unknown:
        raise RuntimeError(
            "paper_values.json names models with no entry in nashlib.registry: "
            + ", ".join(unknown)
        )
    return membership


DATASET_MODELS: dict[str, dict[str, tuple[str, ...]]] = _build_membership()


def models_for(dataset: str, family: str) -> tuple[ModelSpec, ...]:
    """The models the paper reports for one dataset and one decoding family."""
    if dataset not in DATASET_MODELS:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; expected one of {FAMILIES}")
    return tuple(spec(tag) for tag in DATASET_MODELS[dataset][family])


def tags_for(dataset: str, family: str) -> tuple[str, ...]:
    return DATASET_MODELS[dataset][family]


# Backbones of the initialization study. Table 2 in the body is the four largest of
# these, on PubMedQA only; Appendix G.4 extends the comparison to all six on all
# three datasets, which experiment 07 computes but the paper does not tabulate.
BASIN_BACKBONES: tuple[str, ...] = (
    "modernbert_base",
    "ettin_150m",
    "mmbert_base",
    "roberta_large",
    "modernbert_large",
    "ettin_400m",
)
BODY_BASIN_BACKBONES: tuple[str, ...] = (
    "mmbert_base",
    "roberta_large",
    "modernbert_large",
    "ettin_400m",
)

# The five systems measured on one device for the cost figures (Figures 4 and 6),
# in the order the figures draw them. `experiments/09_cost/measure_cost.py` reads this
# rather than keeping its own copy, and the family names are the ones
# `nashlib.artifacts` uses, so a tag here resolves to a stored prediction file.
COST_SYSTEMS: tuple[tuple[str, str, str], ...] = (
    ("modernbert_large", "nash_maxgap", "ModernBERT-L, all-mask"),
    ("modernbert_large", "nash_l2r", "ModernBERT-L, L2R"),
    ("falcon_7b", "autoregressive", "Falcon 7.2B"),
    ("gpt2_large", "autoregressive", "GPT-2 Large 774M"),
    ("opt_350m", "autoregressive", "OPT-350M 331M"),
)


def summary() -> str:
    """A human-readable census of the models each evaluation set reports."""
    lines = ["model census by evaluation set (family: count)"]
    for dataset in SCORED_SETS:
        parts = [
            f"{family} {len(DATASET_MODELS[dataset][family])}" for family in FAMILIES
        ]
        total = sum(len(DATASET_MODELS[dataset][family]) for family in FAMILIES)
        lines.append(f"  {dataset:10s} " + "  ".join(parts) + f"  = {total} rows")
    return "\n".join(lines)
