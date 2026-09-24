# PU-BRIDGE-R1 — Final bridge equation report (parallel-universe line)

**Claim ceiling:** these results come from a parallel-universe simulator (`w1/pu_engine.py`, sha 4ace90d0…) built from the documented rule text. They are NOT the canonical F50 engine and NOT F50 tape. They carry no Research OS authority. They are not an F50 closure claim.

Freeze lock: the hashes were recorded before FRESH was generated (`sealed/FREEZE_LOCK_BEFORE_FRESH.txt`).
BRIDGE_FREEZE 53306ca2… · CONTROLLER_FREEZE ce3612f7… · FRESH table d085bb1a… (seeds 101–106, 1,645,450 rows, 4,045 births, 32,909 redraw windows).

## 1. Exact engine surface
Γ_birth = 0.9 − max(s_own, s_best); basin margin G = max_j Γ_birth(j).

## 2. Claude diagnostic surface
Γ_shape = 0.9 − s_min.
By construction Γ_shape ≥ Γ_birth, because max(s_own, s_best) ≥ s_own. So shape is a necessary-condition envelope, not the birth gate.
On FRESH, window recall = 1.00, precision = 0.85, median Δt = 0, and it is never late. It raises 1,163 false windows, where the rival rescues the member (reassignment rather than birth).

## 3. Bridge equations (frozen on DEV, a_n = sqrt(2 ln n))
F1_LEGAL (PRIMARY, runtime-legal):
Γ̂ = 0.24941893 − 0.36701130·yo − 0.04010088·yb + 0.18789254·a_n·sqrt(1−R²) + 1.83305038·skew + 0.02603607·cob + 0.09085124·dmu

F2_AUG (needs the proposed second-moment state σ_own = sqrt(C_ownᵀ Σ_z C_own)):
Γ̂ = 0.45307925 − 0.54018042·yo − 0.02521551·yb + 0.03044083·a_n·sqrt(1−R²) + 0.33540365·skew + 0.00767191·cob + 0.03193147·dmu + 0.94381818·a_n·σ_own − 0.00133795·a_n·σ_best + 0.00093390·ρ

EVENT = 1[Γ̂ > 0].

## 4. ChatGPT controller surface
Γ_G = P_SWITCH − 0.5, counted only when challenger ≠ champion. This is a frozen two-layer recurrent analog, trained on W-labels only (`w3/`).

## 5. FRESH crossing statistics (per 50-step redraw window)
| pair | recall | precision | median Δt | median abs Δt | P90 abs Δt | abs Δt ≤ 1 | abs Δt ≤ 10 |
|---|---|---|---|---|---|---|---|
| shape vs birth | 1.000 | 0.852 | 0 | 0 | 1 | 0.90 | 0.95 |
| F0_LOW vs birth | 0.476 | 0.795 | 0 | 2 | 23 | 0.46 | 0.74 |
| **F1_LEGAL vs birth** | **0.555** | **0.849** | 0 | 1 | 21 | 0.52 | 0.76 |
| **F2_AUG vs birth** | **0.826** | **0.879** | 0 | 1 | 14 | 0.56 | 0.86 |
| controller vs birth | 0.091 | 0.424 | +13 | 14 | 40 | 0.14 | 0.40 |
| controller vs F1 | 0.156 | 0.476 | +8 | 12 | 38 | 0.15 | 0.46 |

Row level, F2_AUG: sign agreement 0.968, corr 0.966. Row level, F1: sign agreement 0.945, corr 0.904.
Among challenger-valid rows, the correlation between Γ_G and Γ_birth is 0.08.
P_SWITCH averaged around birth redraws is 0.33–0.37 with only a small bump at the redraw (0.371), against a baseline of 0.361. It shows no crossing toward 0.5.

## 6. Which surfaces coincide
- Σ_bridge(F2) ≈ Σ_birth: supported on FRESH (86% of crossings fall within 10 steps).
- Σ_bridge(F1, legal-only) ≈ Σ_birth: partial. It is precise (0.85) but misses 45% of birth windows.
- Σ_shape ⊇ Σ_birth: it is an envelope. The gap between them is rival rescue.
- **Σ_G ≉ Σ_birth.** This is result type 2: the recurrent decision law is a different surface. It reacts late (median +13 steps) and rarely (recall 0.09).

## 7. Necessary observables
- yo, skew and the extreme-value spread term carry most of the signal.
- Rival terms (yb, cob) add little.
- The projected second moment σ_own is the single largest lever: recall rises from 0.55 to 0.83 and P90 abs Δt falls from 21 to 14.

## 8. Fiber ambiguity (FRESH)
- Legal coordinates: 36.2% of rows fall in mixed-sign cells, with 0.209 bits of conditional entropy.
- With σ_own: 30.7% of rows, 0.184 bits.
The ambiguity persists, so this is not a deterministic closure.

## 9. Claim ceiling
In this parallel universe, a low-dimensional explicit bridge exists whose zero set tracks the exact birth surface on FRESH seeds. This holds only when L1.5 is augmented with the projected second moment. The legal first-moment bridge is partial. The recurrent HOLD/SWITCH surface does not align with birth. None of this is transferred to canonical F50 until the same pipeline runs on the real engine and tape under a Science Supervisor contract.

## 10. Remaining gap
1. Decide whether L1.5 may maintain M2 = Σ z zᵀ; if it may, σ_own becomes a legal observable.
2. The residual 31% fiber ambiguity needs member-level tail information, meaning higher moments or the extreme member.
3. The controller has to learn an event channel (Γ̂ as an input) before it can share a surface with birth. That is a new experiment.
