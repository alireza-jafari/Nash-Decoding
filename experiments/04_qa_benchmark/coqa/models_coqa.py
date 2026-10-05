"""The exact set of models CoQA reports, and why it is this set.

Table 7 of the paper lists 3 one-shot masked, 14 autoregressive and 3 Nash rows for
CoQA, and it scores every one of them twice: on the 1,814-turn cohort and on all 7,983
development turns. The three benchmarks do **not** share one model list: each was run
with the systems that table reports, and the autoregressive subsets in particular are
neither equal nor nested. Printing this module is the quickest way to see which models
a CoQA run will touch.

Fourteen autoregressive baselines, from GPT-2 Small at 124M to OLMo-1.7 at 7B. It is
the **narrowest** of the three autoregressive sets, not the widest: the 7B band that
PubMedQA and CLAPNQ carry is largely absent here, and what CoQA has instead is the
small end -- GPT-2 Small and Medium, Pythia-410M, Qwen2.5-0.5B -- which the other two
drop. CoQA answers are short, so a run is cheap, and it is the shortest-answer
benchmark that the cost comparison of Figure 4 is drawn on.

The membership itself is not written here. It is derived from
`paper/paper_values.json` by `nashlib.registry`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nashlib import registry  # noqa: E402

DATASET = "coqa"

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
    print(f"\ntotal rows in Table 7: "
          f"{len(ONESHOT) + len(AUTOREGRESSIVE) + len(NASH_BACKBONES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
