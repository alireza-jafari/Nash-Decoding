# 11. Example generations

**Paper:** Appendix G.2. **Needs a GPU:** no.

```bash
python experiments/11_qualitative_examples/select_examples.py
python experiments/11_qualitative_examples/select_examples.py --percentiles
python experiments/11_qualitative_examples/select_examples.py --percentiles --dataset coqa -n 6
```

## What it shows

Ten CLAPNQ generations from ModernBERT-Large under Nash decoding, verbatim, with F1
from 98.5 down to 23.1. The first command writes the ten examples of Appendix G.2; the
one discussed in Section 4.2 is marked.

`--percentiles` selects from any run instead: one example at each evenly spaced
percentile of the run's own F1 distribution, so the list spans the run from its best
example to its worst. That selection is deterministic -- sort by F1, index by
percentile, no sampling and no seed.

The best is a 48-slot canvas filled to 98.5 F1:

> **Question** where do they shoot guy's grocery games
>
> **Nash decoding** Guy's Grocery Games Season 1 was shot inside of an actual grocery
> store, Field's Market in West Hills, California. For Season 2, the market was built
> in a 15,500 square foot warehouse in Santa Rosa.

It contains two well-formed sentences with proper nouns and numbers, showing that a
Nash equilibrium can be a coherent text even though the masked model is not fine-tuned.
All text is verbatim output from ModernBERT-Large under Nash decoding.

## Data

`data/clapnq_softinstr.jsonl` (300 questions) and
`artifacts/predictions/clapnq__nash_maxgap__modernbert_large.jsonl.gz`. The examples
are selected from outputs that already exist; nothing is decoded here. See
`data/README.md` in this folder.

The question shown for each example is recovered from the prompt the system received
(`nashlib.datasets.current_question`): the last `Q:` line of the CoQA dialogue prompt,
and the `Question:` line of the other two.

## Results

| file | written by |
|---|---|
| `clapnq_modernbert_large_paper_examples.md` | the default run: the ten examples of Appendix G.2, readable |
| `clapnq_modernbert_large_paper_examples.json` | the same, with each example's rank within the run and its telemetry: slots, forward passes, termination status |
| `<dataset>_modernbert_large_examples.md`, `.json` | `--percentiles`: the evenly spaced selection from that dataset's run |

With `--percentiles`, CoQA is selected from `data/coqa_paperprompt.jsonl.gz`, read as
the 1,814-turn cohort.
