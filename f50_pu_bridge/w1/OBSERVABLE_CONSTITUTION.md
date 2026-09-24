# W1 — Observable constitution (parallel-universe line PU-BRIDGE-R1)

Engine: `w1/pu_engine.py` (sha256 4ace90d0…). This is a parallel-universe simulator built from the
documented birth rule. It is NOT the canonical F50 engine and it uses no F50 tape.

Exact rule (per member j, at redraw, every 50 steps):
  Γ_birth(j) = 0.9 − max(z_j·C_own, max_{k≠own} z_j·C_k);  birth iff Γ_birth > 0.
Basin target per step: G_birth = max_j Γ_birth(j).  Diagnostic: G_shape = 0.9 − min_j z_j·C_own.

Table: one row = (seed, step, basin) with n ≥ 2.

| column | class | note |
|---|---|---|
| seed, step, basin, redraw_index, phase | key | phase = step mod 50 |
| n, age | LEGAL | member count, steps since basin birth |
| R = ‖μ‖, skew = 1−μ·C/R, yo = μ·C_own | LEGAL | first moment vs own centroid |
| yb = μ·C_rival, cob = C_own·C_rival | LEGAL | C_rival = nearest other centroid (aggregate) |
| dR, dskew, dmu | LEGAL | one-step deltas of aggregates |
| so, sb, rho | PROPOSED_AUGMENTATION | projected 2nd moment of members on (C_own, C_rival); needs L1.5 to maintain M2 = Σ zzᵀ |
| G_birth | HIDDEN_TARGET | DEV fitting target only; never an input |
| G_shape | HIDDEN_DIAGNOSTIC | never an input |

Splits: DEV seeds 1–6 (built now). FRESH seeds 101–106 are generated only after BRIDGE_FREEZE exists.
Events file: births (hidden) and per-interval realised centroid movement W (moves).
