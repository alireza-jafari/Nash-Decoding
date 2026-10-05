# 5. Decoding dynamics

**Paper:** Figure 3 (body), Figure 5 and Appendix G.1. **Model:** ModernBERT-Large,
all-mask construction. **Needs a GPU:** no.

```bash
python experiments/05_decoding_dynamics/analyse_trajectories.py
python experiments/05_decoding_dynamics/analyse_trajectories.py --figures
```

## What it shows

Two curves against the update index k, on all three datasets. F1 measures movement
toward the reference answer; the Nash gap measures the remaining incentive for any
player to change its token. Neither is a function of the other, which is the point of
plotting them together.

| | CoQA | PubMedQA | CLAPNQ |
|---|---:|---:|---:|
| mean gap at k=0 | 0.47 | 0.59 | 0.45 |
| peak mean gap | 0.71 | 0.81 | 0.76 |
| at update k | 3 | 10 | 21 |
| rise | +49% | +36% | +71% |
| answer written at the peak | ~45% | ~25% | ~43% |
| F1 within 1% of final at k | 10 | 55 | 76 |
| plotting window | 50 | 175 | 250 |
| longest trajectory | 40 | 156 | 240 |
| final F1 | 58.12 | 24.28 | 38.04 |
| converged / limit cycles | 1806 / 8 | 500 / 0 | 300 / 0 |

**The gap rises before it falls.** Generation starts from an entirely masked
sequence, while ModernBERT was pretrained at a 30% masking rate. With little context
the model is uncertain and every conditional is flat, so the maximum available
improvement is small; as early tokens are filled, the emerging context sharpens the
full-context conditionals and the largest available improvement grows. The peak
occurs later for longer-answer datasets, and at the peak a substantial fraction of the
answer is still masked. Once sufficient context is placed, Nash updates increasingly
resolve unstable positions and drive the gap toward zero.

**Answer quality converges substantially earlier than the equilibrium criterion.** F1
comes within 1% of its final value at k = 10, 55 and 76, while the longest trajectories
run to 40, 156 and 240 updates, and over the second half of each trajectory F1 changes
by at most 0.09 points. Most answer quality is obtained well before exact convergence;
the additional computation primarily removes residual token-level instability and
certifies equilibrium. Revisions -- updates that rewrite an already-written position --
account for 4.3%, 2.5% and 4.4% of all updates.

**Convergence is nearly universal.** All 500 PubMedQA and all 300 CLAPNQ examples reach
an exact equilibrium, and 1,806 of 1,814 on CoQA; the remaining trajectories stop at
the first repeated state.

## Conventions

* **k = 0 is the all-[MASK] canvas**, matching the paper, so k counts updates already
  applied.
* A partially-decoded sequence is scored as it stands, with unwritten slots
  contributing nothing, so the F1 curve starts near zero rather than being undefined.
* Once a trajectory terminates it holds its final value; the "questions still running"
  fraction is reported alongside.
* The "answer written at the peak" row counts one slot filled per update.

## Where the trajectory files come from

The trajectory files are built from decoding runs by `tools/build_trajectories.py`:
given a run's raw per-example records it replays the recorded updates, scores each
partial canvas and writes the file, using a tokenizer and no model weights and no GPU.

## Data

`artifacts/trajectories/{coqa,pubmedqa,clapnq}_all_mask.json` -- per example, the gap
and the F1 of the partially-decoded answer at every update. `clapnq_l2r.json` is the
same for left-to-right construction. The F1 in those files is scored against the
references in the corresponding evaluation set -- `data/coqa_paperprompt.jsonl.gz` for
CoQA, `data/{pubmedqa,clapnq}_softinstr.jsonl` for the other two; the revision share is
read from `artifacts/predictions/<dataset>__nash_maxgap__modernbert_large.jsonl.gz`.
See `data/README.md` in this folder.

## Results

`results/dynamics_summary.json`, and with `--figures`, the three panels of Figure 5.
