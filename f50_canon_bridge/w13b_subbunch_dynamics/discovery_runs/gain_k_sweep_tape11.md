Manual sweep log (tape_0_1357_11, small samples, horizons 5/50 steps only, for speed) used to pick
the frozen scalar gain before the 3-tape discovery validation and freeze.

k in {2,4,8,16}, gain_scale in {0.05 .. 1.3} (multiplies eng.config.attract_gain=0.3):

| k | g | skill@5 | skill@50 |
|---|---|---|---|
| 2 | 0.05 | -1.06 | -0.98 |
| 4 | 0.05 | -1.04 | -0.97 |
| 8 | 0.05 | -1.16 | -1.10 |
| 16 | 0.05 | -1.78 | -1.12 |
| 8 | 0.08 | -1.06 | -1.02 |
| 8 | 0.12 | -0.94 | -0.91 |
| 8 | 0.15 | -0.86 | -0.83 |
| 8 | 0.18 | -0.78 | -0.76 |
| 8 | 0.20 | -0.74 | -0.71 |
| 8 | 0.25 | -0.66 | -0.61 |
| 8 | 0.30 | -0.61 | -0.54 |
| 8 | 0.40 | -0.60 | -0.44 |
| 8 | 0.50 | -0.60 | -0.39 |
| 8 | 0.60 | -0.60 | -0.38 |
| 8 | 0.70 | -0.63 | -0.38 |
| 8 | 0.85 | -0.71 | -0.31 |
| 8 | 1.00 | -0.91 | -0.09 |
| 8 | 1.30 | -1.69 | -0.80 |

Best error is a shallow minimum around g=0.4-0.5 at the 5-step horizon and g~1.0 at the 50-step
horizon; increasing k from 2 to 16 gives no improvement at any gain tested (k=8 and k=16 are
statistically indistinguishable, both worse than universal AR3 by roughly an order of magnitude in
error). g=0.5, k=8 was frozen as a representative middle-ground choice, not because it is
meaningfully better than the neighbouring grid points -- none of them come close to AR3.

AR3 baseline on the same block/cores, for reference: skill@5 ~0.93-0.96, skill@50 ~0.76-0.86.
