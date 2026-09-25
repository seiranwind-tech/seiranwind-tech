# W13 — Sub-bunch (次生葡萄串) attack on the last gap A_a

**Line:** exploratory Claude line. This is not a Research OS / Science Supervisor ruling and not an F50 closure claim.

**Protocol:** freeze first, then evaluate once on 2 NEW prospective tapes, (4, 5151, 41) and (3, 6262, 42). The engine is frozen.

| stage | question | frozen law | commits | prospective verdict |
|---|---|---|---|---|
| W13A | Can A_a be computed from a fixed-size sub-bunch description? | 32 k-means sub-bunches (centroid + count) + closed environment (point-mass cores + halo) | 5f3dcca → 32a9188 | **FAIL**. Core-force cos 0.81 / 0.82 (n ≥ 30) and 0.74 / 0.81 (n ≥ 100); the bar is 0.999 / 0.99 |
| W13B | Do sub-bunches evolve autonomously (a core-centre rollout)? | 8 sub-bunches, each with salt_MF + 0.5·gain·nr(mean-field attraction) | b9a12e3 → ef30906 | **FAIL**. 250-step skill −0.51 / −0.73 vs AR3 0.46 / 0.42 |

## Conclusion (W11 + W12 + W13)

The core's internal pull A_a is **not closable from any fixed-size, low-dimensional description tested**:
- a single centroid
- top-k principal components (needs k ≈ 32)
- 2–32 sub-bunches, as forces or as dynamics

It carries O(n) information. The per-member unit normalisation of the attraction makes it sensitive to fine member offsets.

## State of the L1.5 macro equation
- **Closed:**
  - salt mean-field
  - core–environment coupling (point-mass cores + halo)
  - halo dynamics (W12C, prospective PASS)
  - boundary-transfer flux, i.e. the residual H_T (W12B, prospective PASS)
- **Open:** A_a. The best available description is the universal AR3 on centre increments, with 250-step skill about 0.40–0.46 prospectively. It is empirical, not closed.

## Next options
1. Accept A_a as a **stochastic / AR latent force** and close the statistics instead of the trajectory. That gives an effective SDE for M_a whose drift is salt_MF + environment and whose noise is AR3-coloured. Test the distributional agreement, e.g. spread of the 250-step displacement.
2. Keep A_a as a low-rate **handoff** and measure how often it must be refreshed to stay below a target error. This replaces H_S at 5 steps with a much sparser A_a handoff.
