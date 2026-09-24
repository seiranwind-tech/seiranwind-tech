# CANON-BRIDGE-R1 — Final bridge equation report (canonical engine line)

**Engine:** F40 V3.9R3 `f40_v39r3_engine.py`, sha256 4a71aeb4a66a8a07…. This is the canonical engine SHA cited in the Research Admin preflight. It was run unmodified, with frozen default config: N=2000, d=300, threshold 0.9, reassign every 50 steps, EXACT_GLOBAL interaction.

**Deviation:** the vectors are spaCy en_core_web_md 300d, not GloVe, because the GloVe download was blocked in this container.

**Recorder fidelity:** births reproduced from the replicated gate equal the engine's lineage births (DEV 1,502/1,502; FRESH 1,536/1,536).

**Freeze lock:** recorded before FRESH was generated. BRIDGE f48124de… · CONTROLLER 4c1038ec…
FRESH = vector blocks 4–7, disjoint from DEV blocks 0–3: 704,129 rows, 14,993 redraw evaluations, 807 birth windows.

**Claim ceiling:** a non-authoritative Claude-line analysis. It has no Science Supervisor contract, and it is not a formal F50 decision.

## 1. Exact engine surface (verified from source)
Γ_birth = 0.9 − max(s_own, s_best), where s_own = zin·C[label] and s_best = max over the 20 reachable neighbour wells' basin centres.

## 2. Claude diagnostic surface
Γ_shape = 0.9 − s_min. It is an envelope, since Γ_shape ≥ Γ_birth.
On FRESH, windows: recall 1.000, precision 0.871, median Δt = 0, never late.

## 3. Bridge equations (frozen, a_n = sqrt(2 ln n))
F1_LEGAL (runtime-legal):
Γ̂ = −0.04522529 − 0.05281174·yo − 0.04946418·yb + 0.12982607·a_n·sqrt(1−R²) + 0.75274091·skew + 0.04476542·cob − 0.00575106·dmu

F2_AUG (adds the projected second moment σ_own² = C_ownᵀ Σ_z C_own):
Γ̂ = 0.62914784 − 0.72730094·yo − 0.00844382·yb + 0.02557798·a_n·sqrt(1−R²) − 0.06055652·skew + 0.00855149·cob − 0.00071268·dmu + 1.00069874·a_n·σ_own − 0.06659787·a_n·σ_best (the ρ coefficient is about 0)

**Phenomenological reading:**
F2 ≈ 0.727·(0.865 − yo) + 1.00·a_n·σ_own, i.e. Γ̂ ≈ c·(q* − ȳ_own) + sqrt(2 ln n)·σ_own.
This is the extreme-value law s_min ≈ ȳ_own − sqrt(2 ln n)·σ_own, with a unit coefficient (κ = 1) on the spread term. The same coefficient appeared on the PU engine (0.94) and here (1.0007). The rival term is small.

## 4. ChatGPT controller surface
Γ_G = P_SWITCH − 0.5, counted only when challenger ≠ champion. This is a frozen analog controller retrained on canonical W-labels.

## 5. FRESH statistics
Redraw-step event classification (phase 49, the operational birth decision):
| family | sign agree | precision | recall | corr |
|---|---|---|---|---|
| F0_LOW | 0.971 | 0.655 | 0.605 | 0.925 |
| F1_LEGAL | 0.971 | 0.648 | 0.624 | 0.928 |
| **F2_AUG** | **0.976** | **0.670** | **0.839** | **0.950** |

First-crossing inside each 50-step window, measured against birth:
| pair | recall | precision | median Δt | median abs Δt | P90 abs Δt | abs Δt ≤ 10 |
|---|---|---|---|---|---|---|
| shape | 1.000 | 0.871 | 0 | 0 | 5 | 0.92 |
| F1_LEGAL | 0.537 | 0.665 | −6 | 13 | 33 | 0.44 |
| F2_AUG | 0.747 | 0.696 | 0 | 8 | 21 | 0.62 |
| controller | 0.105 | 0.316 | −24 | 25 | 44 | 0.24 |

Among challenger-valid rows, the correlation between Γ_G and Γ_birth is 0.05.
P_SWITCH averages about 0.08 and decays after births; it never approaches 0.5 around birth.

## 6. Coincidence verdict
- Σ_bridge(F2) ≈ Σ_birth: supported. Precision is limited at the fixed threshold 0, a base-rate effect (event rate 2.9%); the threshold was not tuned.
- Σ_bridge(F1, legal-only): partial. Recall is 0.62 at redraw.
- Σ_shape ⊇ Σ_birth: an envelope.
- **Σ_G ≠ Σ_birth.** This is result type 2, and it matches the PU line.

## 7. Necessary observable
The single key state is the projected second moment σ_own. Adding it lifts redraw recall from 0.62 to 0.84.

## 8. Fiber ambiguity (FRESH)
- Legal coordinates: 36.3% of rows fall in mixed-sign cells, with 0.106 bits of conditional entropy.
- With σ_own: 29.7% of rows, 0.103 bits.
The ambiguity persists, so this is not a deterministic closure.

## 9. Claim ceiling
On the canonical engine (with substituted vectors), a preregistered explicit bridge generalises to disjoint FRESH blocks. Its dominant term is the extreme-value spread a_n·σ_own with a unit coefficient. The recurrent HOLD/SWITCH surface is a different surface. This is **not** a full L1.5 closure, and not a result about GloVe or 20K runs.

## 10. Next gap
1. Admit M2 = Σ zzᵀ, or its projection σ_own, into L1.5 state.
2. The residual ambiguity needs tail or extreme-member information.
3. Rerun with GloVe and under a Supervisor contract.
4. Feeding Γ̂ into the controller would be a new experiment.
