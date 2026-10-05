"""The exact set of models PubMedQA reports, and why it is this set.

Table 8 of the paper lists 3 one-shot masked, 18 autoregressive and 3 Nash rows for
PubMedQA. The three benchmarks do **not** share one model list: each was run with the
systems that table reports, and the autoregressive subsets in particular are neither
equal nor nested. Printing this module is the quickest way to see which models a
PubMedQA run will touch.

Eighteen autoregressive baselines, the **widest** of the three sets and not a superset
of either other one. Against CoQA's it adds eight 7-8B models -- Amber, Falcon,
Granite, Llama 1, Llama 3.1, both Mistral versions and OpenLLaMA -- and drops four:
GPT-2 Small, GPT-2 Medium, Pythia-410M and Qwen2.5-0.5B.

The membership itself is not written here. It is derived from
`paper/paper_values.json` by `nashlib.registry`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import registry  # noqa: E402

DATASET = "pubmedqa"

ONESHOT = registry.models_for(DATASET, "oneshot")
AUTOREGRESSIVE = registry.models_for(DATASET, "autoregressive")
NASH_BACKBONES = registry.models_for(DATASET, "nash")

# The initialization study of experiment 07 runs six backbones rather than the three
# the main tables report, on all three datasets and both constructions.
BASIN_BACKBONES = tuple(registry.spec(tag) for tag in registry.BASIN_BACKBONES)


def main() -> int:
    for family, models in (
        ("one-shot masked LMs", ONESHOT),
        ("autoregressive", AUTOREGRESSIVE),
        ("Nash decoding backbones", NASH_BACKBONES),
    ):
        print(f"\n{family} ({len(models)})")
        for model in models:
            print(f"  {model.tag:18s} {model.parameters:>5s}  "
                  f"{model.hf_id}@{model.revision[:8]}")
    print(f"\ntotal rows in Table 8: "
          f"{len(ONESHOT) + len(AUTOREGRESSIVE) + len(NASH_BACKBONES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
