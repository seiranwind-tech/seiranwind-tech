# W9 protocol — full L1.5-state gate closure (transport, birth, merge and centre channels)
This file was frozen before any formal W9 run. Only two smoke tests were run, of 300 and 160 steps, both on the discovery block.

- **Engine:** sha 4a71aeb4…, unmodified.
- **Vectors:** GloVe compact20k, sha a0997bb3…. T = 6000.
- **Discovery:** permutation 12345, block 0, engine seed 1.
- **Prospective (new):** (perm 3131, b0, seed 9) and (perm 5757, b1, seed 11).

## What the shadow simulator does
It is an independent re-implementation of the engine's L1.5-level operators: basin-centre prediction, observation mixing, relabel/compaction, birth, the basin neighbour graph, and merge.

Its initial state is copied once from the engine: labels, centres, tracks, ages and the basin graph. After that it **never reads engine labels, centres, tracks, ages or the graph**.

It receives only:
- **H_S(t):** the observed basin centres, normalize(Σ_members zin / n). Sent at every centre-update step (cs mod 5 = 0) and inside each reassign.
- **H_T(t_r):** at reassign steps, the base-scan result computed on the shadow's own centres and labels plus the current zin: per-well (best_label, best ≥ thr, orphan), and the zin of each orphan.
- **Static structure:** the well kNN graph (fixed at init), config constants and the engine bandwidth.

## Checks after every engine step
- K, labels, track ids, ages and the basin graph must be exactly equal.
- The maximum absolute centre error is recorded, along with the number of steps where the centres are bitwise equal.
- **Thinned variant (sh2):** H_S only at reassign gates, prediction only in between. Reports divergence (track disagreement, K difference, spurious births).

## Verdicts
- **EXACT_L15_STATE_CLOSURE_WITH_AGGREGATE_HANDOFF:** every mismatch count is 0 and the centre error is 0 on all prospective runs.
- **HS_THINNING_ADMISSIBLE:** the thinned variant stays exact. Otherwise HS_EVERY_UPDATE_REQUIRED.
- Autonomy of H_S itself (the member-motion channel) is not tested here, and is not claimed.
