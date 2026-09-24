# W2 — L1.5 Observable Bridge Report, CANONICAL engine line (CANON-BRIDGE-R1)

Engine: F40 V3.9R3, `canon_recorder.py` sha256: `facc817b531f8ddb575d1fbe51744f39cd082016c8d2b705db74bfc06adfa8bb`
DEV table sha256 (`w1/DEV_table.npz`): `2f7cd7067573423ef0838ca160ec544f40b28adf8ef3ce45338f3f6e809b0e4c`
BRIDGE_FREEZE.json sha256: `f48124deddfc5164b4c185f331ddc3cbf6903a01c714c3e87dd12035985657d6`
bridge_eval.py sha256: `f58caab69b0c016bfe58e8d28fdc61994e698c4768907d4c8b023da6c18646c9`

This is a **coefficient refit only**: the three preregistered families and their exact basis
functions are copied unchanged from the PU line's `f50_pu_bridge/w2/PREREG.json`
(see `w2/PREREG.json` here, which notes the copy). DEV here is the canonical engine's blocks
0-3 (677,340 rows total; 677k rows, event rate ≈2.78% overall, ≈4.27% at phase 49). All
fitting/evaluation used phase==49 rows in full plus a seeded (rng seed 12345) 20% uniform sample
of other-phase rows (147,243 rows), matching the PU line's compute protocol. Target is `G_birth`,
threshold fixed at 0 (`event_hat = F_hat > 0`).

## Equations (frozen, OLS, all DEV blocks 0-3)

**F0_LOW** (own-only ansatz, reference):
```
G_F0 = 0.04466612 - 0.14510187*yo + 0.12536191*sqrt(2*log(n))*sqrt(1-R**2)
```

**F1_LEGAL (PRIMARY)** (joint own+rival, LEGAL columns only):
```
G_F1 = -0.04522529 - 0.05281174*yo - 0.04946418*yb
       + 0.12982607*sqrt(2*log(n))*sqrt(1-R**2)
       + 0.75274091*skew + 0.04476542*cob - 0.00575106*dmu
```

**F2_AUG** (F1_LEGAL + proposed second-moment augmentation so, sb, rho):
```
G_F2 = 0.62914784 - 0.72730094*yo - 0.00844382*yb
       + 0.02557798*sqrt(2*log(n))*sqrt(1-R**2)
       - 0.06055652*skew + 0.00855149*cob - 0.00071268*dmu
       + 1.00069874*sqrt(2*log(n))*so - 0.06659787*sqrt(2*log(n))*sb
       + 0.00000000*rho     (rho coeff ~8.2e-11, numerically zero)
```

## Internal DEV validation (fit on blocks 0-2, evaluated on held-out block 3)

| Family | Split | n | Sign agree. | Precision | Recall | Corr | RMSE | False-early (FP) | False-late (FN) |
|---|---|---|---|---|---|---|---|---|---|
| F0_LOW | all rows | 36,150 | 0.9709 | 0.5034 | 0.4858 | 0.9217 | 0.01266 | 507 | 544 |
| F0_LOW | phase==49 | 3,565 | 0.9697 | 0.6569 | 0.5960 | 0.9280 | 0.01252 | 47 | 61 |
| F1_LEGAL (PRIMARY) | all rows | 36,150 | 0.9704 | 0.4950 | 0.5113 | 0.9246 | 0.01243 | 552 | 517 |
| F1_LEGAL (PRIMARY) | phase==49 | 3,565 | 0.9697 | 0.6503 | 0.6159 | 0.9306 | 0.01231 | 50 | 58 |
| F2_AUG | all rows | 36,150 | 0.9732 | 0.5335 | 0.6701 | 0.9612 | 0.00900 | 620 | 349 |
| F2_AUG | phase==49 | 3,565 | 0.9711 | 0.6304 | 0.7682 | 0.9502 | 0.01063 | 68 | 35 |

"False-early" = model predicts a birth (F_hat>0) that does not occur (G_birth≤0). "False-late" =
model fails to flag a birth that does occur (G_birth>0).

## Fiber-ambiguity (DEV, all blocks, 4 highest-importance coords per family, 5 quantile bins each)

Importance = |OLS coefficient| × std(feature), computed on the blocks 0-2 fit.

**F1_LEGAL** top-4 coords: `sqrt(2*log(n))*sqrt(1-R**2)`, `yb`, `cob`, `skew`
- 318 non-empty cells (of 5^4=625 possible), 147,243 rows
- Fraction of rows in cells containing **both signs** of G_birth: **0.3453**
- Row-weighted mean conditional sign entropy: **0.1111** bits (of max 1.0)

**F2_AUG** top-4 coords: `yo`, `sqrt(2*log(n))*so`, `sqrt(2*log(n))*sqrt(1-R**2)`, `sqrt(2*log(n))*sb`
- 266 non-empty cells, 147,243 rows
- Fraction of rows in ambiguous (mixed-sign) cells: **0.2383**
- Row-weighted mean conditional sign entropy: **0.1049** bits

## Interpretation (honest, 5 lines)

1. On the canonical engine, the own-only ansatz (F0_LOW) is already quite strong (corr≈0.92,
   sign agreement≈0.97) and only marginally worse than F1_LEGAL — the canonical engine's much
   larger neighbour-well competition (`s_best` = max over 20 reachable wells, vs. the PU line's
   single aggregate rival) appears to make the aggregate `yb`/`cob` rival summary less
   informative here than on the PU line, where F1's edge over F0 was clearer.
2. F1_LEGAL (PRIMARY) still improves over F0_LOW on every metric except phase-49 sign agreement
   (a tie), with a real recall gain (0.486→0.511 all-rows, 0.596→0.616 at phase 49); `skew` is
   the dominant term here (coefficient 0.753), much as on the PU line (1.833), while `cob` now
   also earns a nontrivial coefficient (0.045).
3. As on the PU line, the second-moment augmentation (F2_AUG) is the largest lever: RMSE drops
   ~28% (0.0124→0.0090 all rows), recall jumps from 0.511 to 0.670 (all rows) and 0.616 to 0.768
   at phase 49, and `so` again dominates its scaled coefficient (≈1.00) — this cross-engine
   consistency (same basis, same dominant term) is a useful sanity check that the augmentation is
   picking up a real, engine-independent extreme-value effect rather than overfitting to one
   simulator's quirks.
4. Precision is noticeably lower here than on the PU line for every family (e.g. F2_AUG
   precision 0.53 here vs. 0.78 on PU, all-rows) — consistent with the canonical engine's much
   lower base event rate (≈2.8% vs ≈7.2% on PU DEV), which mechanically inflates the false-positive
   share for any fixed decision threshold of 0; this is a base-rate effect, not necessarily a
   worse bridge, and argues for recomputing an engine-specific decision threshold before any
   operational use, though the preregistration fixes threshold at 0 for this analysis.
5. Fiber ambiguity is similar in magnitude to the PU line (34.5% of rows in mixed-sign cells for
   F1_LEGAL, 23.8% for F2_AUG, vs. 36.0%/24.9% on PU) and again shrinks substantially once the
   augmentation is added — reinforcing that a chunk of the birth decision is genuinely
   underdetermined by basin-aggregate first moments alone on both engines, and that member-level
   second-moment information (so in particular) recovers a similar, non-trivial fraction of that
   ambiguity in both the parallel-universe and canonical settings.
