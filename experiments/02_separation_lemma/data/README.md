# Data for experiment 2

**None, and none is possible.** Lemma 1 is a statement about a distribution written
down in closed form:

```
p1(0) = 0.51   p1(1) = 0.49        p2(1|0) = 0.51   p2(1|1) = 0.99
```

repeated over n/2 independent blocks. `compute_separation.py` builds it in exact
rational arithmetic and enumerates {0,1}^n; there is no corpus and no model.
