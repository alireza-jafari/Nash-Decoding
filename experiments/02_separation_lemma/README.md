# 2. The autoregressive–Nash separation

**Paper:** Lemma 1, Appendix B. **Model:** none. **Needs a GPU:** no. **Runtime:**
about a second.

```bash
python experiments/02_separation_lemma/compute_separation.py
```

## What it shows

Lemma 1 says the gap between greedy autoregressive decoding and Nash equilibrium is
not a small constant but grows exponentially with sequence length. For every even n
there is a strictly positive autoregressive distribution on {0,1}ⁿ with

```
p(x_nash) / p(x_AR) = (539/289)^(n/2) = exp(γn),   γ = ½ log(539/289) ≈ 0.31164
```

and the two sequences differ at exactly n/2 positions.

The construction is a single block (a, b) ∈ {0,1}², repeated independently:

```
p₁(0) = 0.51   p₁(1) = 0.49
p₂(1|0) = 0.51   p₂(1|1) = 0.99
```

Greedy takes a = 0, because 0.51 > 0.49, and then b = 1. But p(0,1) = 0.2601 against
p(1,1) = 0.4851, so once b = 1 is on the table the full-context best response at the
first position is a = 1. Every block pays the same multiplicative price, and that is
where the exponential comes from. The Nash gap at the greedy output is
125/414 ≈ 0.30193, independent of n.

## What the script computes

The lemma is a statement about a distribution that is written down explicitly. The
script builds the distribution, enumerates {0,1}ⁿ, and computes:

* the greedy output, which is (0,1,…,0,1);
* the Nash gap at all 2ⁿ sequences, which shows the equilibrium is **unique**;
* that unique equilibrium, which is all-ones;
* the number of positions at which the two differ, exactly n/2;
* the probability ratio, equal to (539/289)^(n/2) **exactly**, in rational arithmetic
  with `fractions.Fraction`;
* the same ratio as exp(γn), to within 1e-9 relative;
* the Nash gap at the greedy output, exactly 125/414, which exceeds the 0.30 of the
  lemma.

Enumeration is exact for the small n it is run at; the closed form is then evaluated
on its own for every even n up to 2,000.

## Output

```
   n         greedy    equilibrium  d_H  p(nash)/p(ar)   exp(gamma n)   G(x_ar)  ok
   2             01             11    1         1.8651         1.8651   0.30193  yes
   4           0101           1111    2         3.4784         3.4784   0.30193  yes
   ...
  12   010101010101   111111111111    6        42.0868        42.0868   0.30193  yes
```

## Data

None.

## Results

`results/separation.json` — every enumerated length, with its equilibria and each of
the quantities above.
