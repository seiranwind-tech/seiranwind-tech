# W2 — L1.5 Observable Bridge Report (PU-BRIDGE-R1)

Engine sha256 (`w1/pu_engine.py`): `4ace90d04b44a9efa873242a4e3751d73dff02a09edb3a501de1b11603f6fc10`
DEV table sha256 (`w1/DEV_table.npz`): `082669e153917becad7d507910c79c7fd0a9c6da3d6ddcfb356c05e21f9ae414`
BRIDGE_FREEZE.json sha256: `53306ca269607e9e0aa2400e51be0353ef566464b2ee15c76905ced553a40582`
bridge_eval.py sha256: `eff3b1eb07a2f5ca4a899e2b4ad9971055507696c2cd053d45e7ef471ebdb92d`

All fitting/evaluation used phase==49 rows in full plus a seeded (rng seed 12345) 20% uniform
random sample of other-phase rows, to keep compute modest (≈360k of 1.66M DEV rows). Target is
`G_birth` (signed margin), threshold fixed at 0 (`event_hat = F_hat > 0`). Preregistration
(`w2/PREREG.json`) was written before any fitting.

## Equations (frozen, OLS, all DEV seeds 1-6)

**F0_LOW** (own-only ansatz, reference):
```
G_F0 = 1.08824034 - 1.20904850*yo + 0.12544417*sqrt(2*log(n))*sqrt(1-R**2)
```

**F1_LEGAL (PRIMARY)** (joint own+rival, LEGAL columns only):
```
G_F1 = 0.24941893 - 0.36701130*yo - 0.04010088*yb
       + 0.18789254*sqrt(2*log(n))*sqrt(1-R**2)
       + 1.83305038*skew + 0.02603607*cob + 0.09085124*dmu
```

**F2_AUG** (F1_LEGAL + proposed second-moment augmentation so, sb, rho):
```
G_F2 = 0.45307925 - 0.54018042*yo - 0.02521551*yb
       + 0.03044083*sqrt(2*log(n))*sqrt(1-R**2)
       + 0.33540365*skew + 0.00767191*cob + 0.03193147*dmu
       + 0.94381818*sqrt(2*log(n))*so - 0.00133795*sqrt(2*log(n))*sb
       + 0.00093390*rho
```

## Internal DEV validation (fit on seeds 1-4, evaluated on held-out seeds 5-6)

| Family | Split | n | Sign agree. | Precision | Recall | Corr | RMSE | False-early (FP) | False-late (FN) |
|---|---|---|---|---|---|---|---|---|---|
| F0_LOW | all rows | 118,670 | 0.9377 | 0.6432 | 0.3749 | 0.8851 | 0.01590 | 1846 | 5550 |
| F0_LOW | phase==49 | 10,980 | 0.9140 | 0.6272 | 0.3875 | 0.8742 | 0.01685 | 258 | 686 |
| F1_LEGAL (PRIMARY) | all rows | 118,670 | 0.9440 | 0.6955 | 0.4469 | 0.9034 | 0.01465 | 1737 | 4910 |
| F1_LEGAL (PRIMARY) | phase==49 | 10,980 | 0.9189 | 0.6392 | 0.4714 | 0.8819 | 0.01634 | 298 | 592 |
| F2_AUG | all rows | 118,670 | 0.9653 | 0.7839 | 0.7407 | 0.9635 | 0.00915 | 1813 | 2302 |
| F2_AUG | phase==49 | 10,980 | 0.9446 | 0.7109 | 0.7705 | 0.9378 | 0.01225 | 351 | 257 |

"False-early" = model predicts a birth (F_hat>0) that does not occur (G_birth≤0). "False-late" =
model fails to flag a birth that does occur (G_birth>0). Note both families are noticeably worse
on phase==49 (the actual redraw/decision step) than on the full row population — sign agreement
drops ~2-3 points and recall/precision degrade — because that is exactly where the max-over-members
extreme-value approximation is under most stress.

## Fiber-ambiguity (DEV, all seeds, 4 highest-importance coords per family, 5 quantile bins each)

Importance = |OLS coefficient| × std(feature), computed on the seeds 1-4 fit.

**F1_LEGAL** top-4 coords: `sqrt(2*log(n))*sqrt(1-R**2)`, `skew`, `yo`, `yb`
- 474 non-empty cells (of 5^4=625 possible), 359,889 rows
- Fraction of rows in cells containing **both signs** of G_birth: **0.3602**
- Row-weighted mean conditional sign entropy: **0.2116** bits (of max 1.0)

**F2_AUG** top-4 coords: `sqrt(2*log(n))*so`, `yo`, `sqrt(2*log(n))*sqrt(1-R**2)`, `skew`
- 385 non-empty cells, 359,889 rows
- Fraction of rows in ambiguous (mixed-sign) cells: **0.2487**
- Row-weighted mean conditional sign entropy: **0.1719** bits

Adding the second-moment augmentation shrinks the ambiguous-cell mass by roughly a third (0.360 →
0.249) and the mean conditional entropy by about the same proportion, consistent with `so`
(own-axis projected spread) carrying real information about the birth margin beyond what R, yo,
skew capture at first-moment order alone.

## Interpretation (honest, 5 lines)

1. The own-only extreme-value ansatz (F0_LOW) already recovers most of the signal (corr≈0.88,
   sign agreement≈0.94) — G_birth is dominated by the own-axis extremum, as the engine's rule
   structurally implies, so even a 3-parameter model is a decent zeroth approximation.
2. Adding rival-aware LEGAL terms (F1_LEGAL, the PRIMARY) gives a modest but real improvement
   (corr 0.885→0.903, recall of true births 0.375→0.447 on held-out seeds), confirming that
   `max(s_own, s_best)` genuinely depends on the rival centroid geometry (yb, cob) and not just
   the own axis, though the marginal gain from LEGAL-only rival information is limited.
3. The proposed second-moment augmentation (F2_AUG) is the single largest lever: RMSE nearly
   halves (0.0147→0.0092) and event recall on held-out seeds nearly doubles (0.447→0.741),
   because `so` (own-axis projected spread) is a much better proxy for "how far the true extreme
   member sits from the mean" than the aggregate circular-spread proxy `sqrt(1-R**2)` alone — this
   is expected since `so` is literally the relevant per-member scale the max-of-n argument needs.
4. All families degrade at the actual redraw step (phase==49) relative to the full row
   population; this is the operationally relevant regime and the honest number to trust when
   deciding whether the bridge is fit for controlling births — F1_LEGAL recall there is only
   0.47, and even F2_AUG's phase-49 sign agreement (0.945) trails its all-row figure (0.965).
5. The fiber-ambiguity analysis shows real irreducible overlap even for the best available legal
   inputs — roughly a quarter to over a third of DEV rows sit in bins where both signs of
   G_birth occur — meaning no low-dimensional first/second-moment linear bridge fully resolves
   the birth event from basin aggregates alone; residual member-level heterogeneity that the L1.5
   layer discards is doing real work, and the gap between F1_LEGAL and F2_AUG quantifies part of
   what LEGAL-only observables are structurally leaving on the table.
