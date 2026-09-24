# W6 — GloVe L1.5 closure audit: final report

**Line:** Claude Cloud exploratory analysis. This is **not** a Research OS or Science Supervisor ruling.
**Verdict:** **NOT DETERMINISTICALLY CLOSED — Level 0 (phenomenological).**

## 0. Inputs and protocol
- **Engine:** f40_v39r3_engine.py, sha 4a71aeb4…, unmodified.
- **Vectors:** GloVe 6B 300d compact20k, sha a0997bb3…. The engine's own loader reads it as WORD2VEC_BINARY, 20000×300, with no NaN or Inf (PRECHECK.json).
- **Protocol:** PROTOCOL.md (sha 76f18384…). Runs are T = 6000, recorded from step 1500, fixed in advance. Blocks: DEV 0–3, FRESH 4–7, FRESH2 8–9.
- **Recorder fidelity:** births reproduced from the replicated gate equal the engine's lineage births: DEV 424, FRESH 420, FRESH2 292.
- **DEV size:** 273 basin-level birth events at redraw steps. This is at least 200, so the replicate-DEV rule did not trigger.
- **Freeze:** FREEZE_LOCK.txt (CANDIDATE 5dd467af…, TIMING 5fbcc769…, TAIL_LAW 321c8e69…) was committed in ce14255 before FRESH or FRESH2 was read.
- **Declared DEV-only addendum:** B1c. Its β is calibrated at threshold 0 on DEV redraw rows.

## A. Tail law (DEV)

| | spaCy W5 | GloVe W6 |
|---|---|---|
| γ in L_tail = a·σ·n^γ | 0.345 | **0.456** (free fit 0.446; σ exponent 1.05) |
| γ per block | — | 0.48 / 0.41 / 0.51 / 0.43 |
| γ early / middle / late | — | 0.49 / 0.44 / 0.46 |
| γ on held-out (not refit, check only) | — | FRESH 0.461, FRESH2 0.432 |
| collapse CV across n-bins, U_γ vs U_√(2 ln n) | 0.21 vs 0.22 | **0.15 vs 0.33** |

- **γ ≈ 1/3 is NOT reproduced.** The exponent depends on the embedding: 0.35 for spaCy, 0.46 for GloVe.
- Within one embedding, γ is stable across blocks, time, and held-out splits.
- The power-law form n^γ beats sqrt(2 ln n).
- Per-size-stratum γ is noisy (−0.03 to 0.61) because each bin spans only a narrow range of n.
- U_γ falls for n > 200 (from about 0.73 to 0.45), so the law weakens in large basins.

## B. Frozen laws (DEV)

- B0, the spaCy law transferred unchanged: Γ = 0.9 − ȳ + 0.7065·σ·n^0.3447
- B1, GloVe refit: Γ = 0.9 − ȳ + 0.5375·σ·n^0.4559
- B2, free regression: Γ = 0.8770 − 0.9729·ȳ + 0.5034·σ·n^0.4559. **It recovers 0.9 − 1·ȳ again**, as it did on spaCy (0.892 − 0.989·ȳ).
- B3, B1 plus one rescue scalar: Γ = 0.9 − ȳ + 0.5017·σ·n^0.4559 + 0.00473·cob. It was selected on DEV block-3 validation, where it gave 194 errors against 209 without it.
- B1c, threshold-calibrated: Γ = 0.9 − ȳ + 0.32·σ·n^0.4559

## C. Held-out results (redraw-step classification, then window crossing)

| law | FRESH P / R | FRESH2 P / R | corr (F / F2) | window median abs Δt, P90 (F) | abs Δt ≤ 10 (F / F2) |
|---|---|---|---|---|---|
| B0 direct transfer | 0.559 / 0.504 | 0.490 / 0.580 | 0.946 / 0.952 | 4, 21 | 0.71 / 0.77 |
| B1 refit | 0.305 / 0.602 | 0.335 / 0.682 | 0.951 / 0.945 | 2, 19 | 0.79 / 0.77 |
| B2 regression | 0.323 / 0.610 | 0.349 / 0.699 | 0.951 / 0.946 | 3, 19 | 0.77 / 0.77 |
| B3 with rescue | 0.325 / 0.663 | 0.349 / 0.739 | 0.949 / 0.946 | 3, 20 | 0.77 / 0.76 |
| B1c calibrated | 0.732 / 0.155 | 0.477 / 0.239 | 0.915 / 0.925 | 9, 34 | 0.51 / 0.56 |

- **Direct transfer is not worse than the refit.** The spaCy-frozen B0 gives the best precision/recall balance on GloVe held-out data, so the functional form is not just memorising one embedding. Its coefficients, however, are not universal.
- **Per basin size:** for n ≤ 50, B0 and B1 reach precision 0.63–0.93 and recall 0.36–0.86. For n > 50, precision is 0.06–0.33 in both splits. **91–94% of B1's false positives come from basins with n > 50.** This is a systematic family of residual errors.
- Per block, the results are consistent (B0 precision 0.44–0.62, recall 0.43–0.61).

## D. Timing, from the analytic derivative
Γ̇ = −dȳ/dt + β·n^γ·dσ/dt, using a past-only 5-step difference. The window was selected on DEV from {1, 3, 5, 10}, and n is held constant within an interval.
- **FRESH:** median T̂/T = 1.28, median error 18 steps, P90 error 230 steps, 62% within a factor of 2.
- **FRESH2:** median T̂/T = 1.14, median error 16 steps, P90 error 88 steps, 70% within a factor of 2.
- **Conclusion:** the earlier late bias of 2.15 (W5) was mainly a velocity-estimator problem. The analytic Γ̇ lowers it to 1.1–1.3. The tails are still wide.

## E. Residual audit: is it rival rescue?
**No.** Only 3–5% of own-tail crossings are rescued by the best rival centre; B1's rescued false positives are 8 of 363 on FRESH and 2 of 238 on FRESH2. The single rescue scalar (cob) adds recall but no precision.

The dominant residual is **own-tail estimation error**: σ·n^γ predicts the tail length with about 33% multiplicative scatter near the threshold (0.45 overall).

**Upper-bound test, diagnostic only:** give L1.5 the basin's actual extreme member y_min, a single-scalar tail sketch. It then reaches
- DEV: precision 0.968, recall 1.000, corr 0.9997
- FRESH: precision 0.950, recall 1.000
- FRESH2: precision 0.967, recall 1.000

Its remaining false positives are exactly the rival rescues. Evaluating the extreme member's joint q = max(s_own, s_best) would therefore reproduce the exact gate.

## F. Closure level
- **Level 1 is not met.** Precision and recall of at least 0.95 each were never reached with (ȳ, σ, n[, r]).
- **Level 2 is not met.** Mixed-sign fiber fractions at redraw on held-out are:
  - (ȳ, σ, n): 0.227 on FRESH, 0.307 on FRESH2
  - adding cob: 0.164 on FRESH, 0.187 on FRESH2
- **→ NOT DETERMINISTICALLY CLOSED. Level 0.**

## Minimal-state answer supported by the data
- (ȳ_own, σ_own, n) is **not sufficient** on GloVe. The second moment explains most of the margin (corr 0.95) but not the sign at the wall.
- The evidence points to one additional scalar per basin: the **extreme-member own projection y_min**. This is a tail sketch, far cheaper than a 300×300 M₂. It could be paired with that member's rival projection s_best, so that rival rescue (3–5%) is resolved as well:

  X_L1.5^min = (ȳ_own, σ_own, n, y_min [, s_best at the extreme member])

  Γ_L1.5 = 0.9 − max(y_min, s_best,extreme)   (reduces to the exact gate if the tracked member really is the extreme one)

  Γ̇_L1.5 = −dȳ/dt + β·n^γ·dσ/dt + β·γ·σ·n^(γ−1)·dn/dt   (smooth-law surrogate for timing)

  B = 1[Γ_L1.5 > 0]

- **Open question for the next round:** can L1.5 maintain the extreme member causally, as a runtime "min tracker", without re-reading every member?
  If it can, this becomes a Level-1 candidate. It needs a new preregistration.
  If it cannot, the smooth law Γ = 0.9 − ȳ + β·σ·n^γ remains a Level-0 phenomenological surface, and its coefficients depend on the embedding.

## spaCy W5 vs GloVe W6

| | spaCy W5 | GloVe W6 |
|---|---|---|
| γ | 0.345 | 0.456 |
| β (one-parameter law) | 0.71 | 0.54 (0.32 threshold-calibrated) |
| Regression recovers 0.9 − ȳ | yes | yes |
| FRESH2 window recall / precision (refit law) | 0.84 / 0.80 | 0.73 / 0.42 |
| Is σ_own sufficient | approximately | no (fails for n > 50) |
| Rescue scalar needed | — | no gain in precision |
| Closure level | Level 0 | Level 0 |
