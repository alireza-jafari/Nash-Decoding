# 1. The motivating example

**Paper:** Figure 1, Appendix A. **Model:** GPT-2 Small (124M). **Needs a GPU:** in
practice yes. Every scan scores all 50,257 candidate tokens at every position, so one
scan of the ten-token example is about 130 forward passes over a few thousand
sequences; that is a couple of minutes on a GPU and the better part of an hour on a
CPU.

```bash
python experiments/01_motivating_example/run_motivating_example.py
python experiments/01_motivating_example/run_motivating_example.py --example 2
```

## What it shows

Greedy autoregressive decoding commits to a token using only the tokens to its left,
and cannot revisit that choice once the rest of the sentence exists. Prompted with
*"What famous city is the capital of Spain?"*, GPT-2 Small produces

> The city of **Barcelona** is the capital of Spain.

At the moment that token was emitted the decoder had seen only "The city of", and
under that left context alone it preferred "Barcelona" to "Madrid" by a factor of
1.15. Conditioning instead on *all* the other tokens, including the six to its right,
reverses the preference:

| token | p(x₄ \| x<₄) autoregressive | p(x₄ \| x₋₄) full context |
|---|---:|---:|
| Madrid | 0.0904 | **0.3343** |
| Barcelona | 0.1036 | 0.2655 |
| Valencia | 0.0519 | 0.1507 |
| Spain | 0.0160 | 0.0666 |
| Catalonia | 0.0116 | 0.0305 |

The best response is "Madrid", giving g₄ = 0.0688. Every one of the other nine players
already plays its best response, so G(x) = g₄ and player 4 is the unique improvable
position. One replacement reaches an equilibrium whose next scan reports zero gap at
all ten positions, and whose sequence probability is 1.2591× higher
(log p: −20.2943 → −20.0639).

## A second example, where the players influence one another

Prompted with *"What city is the political capital of Spain?"*, the greedy continuation
is

> The answer is **Barcelona**. The city is the capital of Spain. **It** is the capital
> of Spain.

Here all twenty tokens participate, with the erroneous city at player 4 and the pronoun
"It" opening the third sentence at player 14. The game takes two steps:

| scan | player | token | p(current) | best response | gap |
|---|---:|---|---:|---|---:|
| 1 | 4 | Barcelona | 0.3039 | Madrid (0.3977) | 0.0938 |
| 1 | 14 | It | 0.2709 | Madrid (0.3144) | 0.0435 |
| 2 | 14 | It | 0.3160 | Madrid (0.4132) | 0.0972 |

Player 4 moves first. Correcting it does not leave player 14 untouched: its gap more
than doubles, from 0.0435 to 0.0972, because the repaired first sentence makes the
proper noun a better subject for the third. Player 14 therefore moves second, and the
text reaches the equilibrium

> The answer is **Madrid**. The city is the capital of Spain. **Madrid** is the capital
> of Spain.

The sequence log-probabilities are −25.8888 for the greedy text, −25.6198 after the
first replacement and −25.3516 at equilibrium: each step raises the likelihood, by
1.309× and 1.308×, for a total of 1.711×.

## Why this experiment uses an autoregressive model

It is the only one that does. An autoregressive model's full-context conditional is
exact and compatible with its joint distribution by construction:

```
q(v | x₋ᵢ, prompt) = p(x<ᵢ, v, x>ᵢ | prompt) / Σ_u p(x<ᵢ, u, x>ᵢ | prompt)
```

so Theorem 1 applies here without qualification, and the sequence probability rises at
every update. Everywhere else in the paper the conditionals come from a masked encoder
and need not belong to any common joint distribution.

It is also why nothing else uses it. The numerator is the joint probability of the
*whole* continuation with position i replaced, so the suffix has to be re-scored for
every candidate token: all 50,257 of them, in chunks of 2,048 per forward pass, at
every position, at every step. That cost is the argument of Section 2.2 for using
masked encoders in the question-answering experiments.

## How the two prompts are laid out

Generation and refinement use the same prompt and the same model. What follows the
question differs between the two examples:

* **Example 1: the question, then a space.** The generated sentence is re-tokenized as
  `" " + text`, the way an answer is tokenized everywhere else here, and the game is
  played on those ten tokens.
* **Example 2: the question, then a blank line** (two newline tokens). The twenty
  tokens the model emits are the players exactly as emitted; the first one is `"The"`,
  with no leading space.

## Details that matter

* The continuation is produced by greedy decoding **inside the script** rather than
  pasted in. `--continuation paper` plays the game on the sentence as printed in
  Appendix A instead; the two give the same game.
* Conditionals are computed exactly over the full vocabulary. No shortlist, no beam,
  no approximation.
* float32 with TF32 disabled, asserted against a float64 reference. Two of the gaps
  here are around 0.04, and TF32 changes which token wins an argmax when two logits
  are close.
* Refinement halts when the Nash gap falls below 1e-9, which is "exactly zero" at
  float32 resolution.

## Data

None. The two prompts are the experiment, and they are in the script.

## Results

`results/motivating_example.json` — for each example, the prompt and its token ids;
per step, the Nash gap, the position updated, the token replaced, the new sequence
log-probability and the ratio to the previous one; and per scan, every player's
current probability, best response and gap.
