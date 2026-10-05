# Data for experiment 3

`../../../data/wikitext512_500.json` — 500 prompts from the WikiText-103 test split.

| | |
|---|---|
| records | 500 |
| prompt length | exactly 512 tokens each |
| continuation slots | 64 |
| tokenizer | `answerdotai/ModernBERT-large` @ `45bb4654` |
| sha256 | see `../../../data/checksums.json` |

Each record holds `id`, `split`, `prompt_ids` and `prompt_text`.

**The prompts are stored as token ids, not text.** "Exactly 512 tokens" is a property
of the ids; re-tokenizing `prompt_text` would not guarantee it, and a controlled
continuation length is the whole point of the experiment. `run_wikitext.py` asserts
the length of every record before using it.

The trajectories this experiment analyses are in `../../../artifacts/wikitext/`:
`refinement_trajectories.json` (500 runs, update by update) and `shard*.json` (the
GPT-2 XL referee's continuation log-perplexity along each one).
