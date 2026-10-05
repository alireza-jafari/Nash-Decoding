# 3. Equilibrium dynamics on WikiText-103

**Paper:** Section 2.3, Figure 2, Table 4, Appendix C. **Model:** ModernBERT-Large
(395M), frozen. **Needs a GPU:** only to regenerate the trajectories.

```bash
python experiments/03_wikitext_dynamics/analyse_dynamics.py            # Table 4, Figure 2
python experiments/03_wikitext_dynamics/analyse_dynamics.py --figures  # and draw them
python experiments/03_wikitext_dynamics/run_wikitext.py --gpu 0        # regenerate
```

## The question

Section 2.3 replaces the exact conditionals assumed by Theorem 1 with estimates from a
pretrained masked model and asks whether the dynamics still behave as the analysis
predicts. What is tested is the convergence rate and the likelihood increase of
Theorem 1, not question-answering performance.

500 prompts of exactly 512 tokens from the WikiText-103 test split, 64-token
continuations with ModernBERT-Large, tolerance zero.

## What is measured

**Convergence test (Figure 2a).** The mean over the 500 texts of the running minimum
`min_{j≤k} G(x^(j))` falls from 0.342 at k=1 to 6.9e-5 at k=36 and stays below the 1/k
reference throughout. On log axes a 1/k rate is a straight line of slope −1, so the
reference curve fixes the rate implied by Theorem 1 and not its constant.

**A numerical analogue of the bound.** Writing p̃₀ for the product of ModernBERT's
full-context conditionals at x⁽¹⁾, the running minimum remains below 2 log(1/p̃₀)/k for
**all 500 texts and all k**, with a median ratio of 71×; the mean value of 2 log(1/p̃₀)
is 21.4. Because ModernBERT's conditionals need not be induced by a common joint
distribution, this comparison is an empirical test.

**Monotonicity test (Figure 2b).** The likelihood is evaluated with an external
autoregressive referee, GPT-2 XL, alongside ModernBERT's pseudo-perplexity. Both scores
decrease with iterations, indicating an increase in likelihood: GPT-2 XL perplexity
from 3.54 to 3.48 and pseudo-perplexity from 1.182 to 1.123.

**The step inequality.** Equation 5 requires the sequence probability to rise by a
factor of at least 1 + G(x^(k)) at every update. Substituting the masked model's
pseudo-likelihood for p, that inequality holds on 71.5% of updates, and the
pseudo-likelihood increases on 75.2%. This is the expected consequence of Section 2.2:
an update raises the estimated conditional at the position it rewrites, but it also
perturbs the conditionals at every other position. Monotone improvement is a property
of the idealized game; the Nash gap depends only on conditional probabilities rather
than the joint probability, so the equilibrium certificate G(x) = 0 is computed
directly and holds exactly.

## Table 4

| | |
|---|---|
| Prompts, tokens per prompt, continuation slots | 500, 512, 64 |
| Action set (vocabulary minus four special ids) | 50,364 |
| Reaching an exact equilibrium | 494 (98.8%) |
| Stopped at a repeated state | 6 (1.2%) |
| Already at equilibrium when construction ends | 118 (23.6%) |
| Refinement updates per text (mean / median / max) | 2.60 / 2 / 36 |
| Tokens changed by refinement (mean) | 2.29 |
| Updates that write [MASK] | **0 of 1,300** |
| Model calls per text (construction + refinement) | 64 + 230 |
| Seconds per text | 3.3 |

`[MASK]` is a legal action at every position. Keeping it legal is what makes the token
gap well defined at a position that has not been written yet, and it lets a player
return its position to the masked state if that is its best response. Across all 1,300
updates, no player ever does.

An equilibrium is certified when a full scan of all positions finds G(x) = 0. For
cycle detection, the complete token sequence of every visited state is retained; if a
replacement returns the sequence to an earlier state, refinement stops there and the
termination is recorded as a cycle. While Theorem 1 rules out cycles, they may arise
when the true conditional probabilities are replaced by a transformer's estimates.

## Conventions

**Where the trajectory starts.** Each trajectory has two stages. Construction is a
separate pass of 64 model calls: at each call, the masked position whose predicted
token has the highest conditional probability is written, with [MASK] excluded so that
a real token is written, and nothing is revised. Refinement then runs Algorithm 1 from
the completed sequence x⁽¹⁾. The measured trajectory begins at x⁽¹⁾, so k=1 is the
first fully instantiated sequence; sequence-level quantities such as likelihood and
perplexity are compared from the point where no [MASK] positions remain.

**How a terminated trajectory is counted.** It contributes a gap of zero thereafter.

## Data

`data/wikitext512_500.json`, stored as token ids, so every prompt is exactly 512
tokens.

## Results

`results/wikitext_dynamics.json` — Table 4, both diagnostics, the running-minimum gap
curve and both perplexity curves. With `--figures`, the two panels of Figure 2 as PDF.
