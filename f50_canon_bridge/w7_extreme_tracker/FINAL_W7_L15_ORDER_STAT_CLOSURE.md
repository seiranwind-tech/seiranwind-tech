# W7 — Extreme order-statistic L1.5 tracker: final report

**Line:** Claude Cloud exploratory analysis. This is not a Research OS or Science Supervisor ruling, and it makes no formal "closed" claim.

**Verdict:** **EXACT_HYBRID_TRACKER_WITH_CERTIFIED_REFRESH.**
- It is not FULLY_AUTONOMOUS_FINITE_STATE: every reassign still needs a rebuild.
- The refresh cost is substantial.

## Inputs
- **Engine:** sha 4a71aeb4…, unmodified.
- **Vectors:** GloVe compact20k, sha a0997bb3….
- **Instruction file:** sha be94508d….
- **Protocol and code:** frozen before any W7 data existed. PROTOCOL_W7.md f9c26873…, w7_tracker.py d6e29577… (W7_FREEZE_LOCK.txt, commit 4d112cc). The lock was re-verified as OK after the runs.
- **Discovery:** W6 DEV blocks 0 and 1 (permutation seed 12345, engine seed 1).
- **Prospective:** a new word partition (permutation seed 777), blocks 0 and 1, engine seed 3. It yielded 10,033 basin readouts at redraw steps.

## A. Exact readout
q_i^(1) = min over w in basin i of max(s_own,w, s_best,w). Γ_i = 0.9 − q_i^(1). B_i = 1[Γ_i > 0].

The births predicted from this gate equal the engine's actual births at every recorded reassign step, in all four runs, and birth_budget (256) was never hit (EXACT_GATE_AUDIT.json).

**Claim (1) is confirmed: one scalar, q_min, is sufficient to decode birth.** This is a readout, not dynamics.

Correction to W6: W6's "oracle y_min" was the own-only approximation, 0.9 − min s_own. The exact object is q_min above, which W6's `G_birth` already equalled.

## B. Witness and rank dynamics (prospective)
- **Previous-step witness is still rank 1:** 98.3% of quiet steps, 98.9% of centre-update steps (cs mod 5 = 0), and 91.0% across reassign steps.
- **Across a reassign:** in 1.7% of cases the witness has left the basin, and in 0.1% it has fallen to rank 16 or worse.
- **Worst jump between reassigns:** from rank 1 into ranks 8–15. It never went past rank 16.
- **Witness lifetime (steps):** median 2–3, P90 125–187, P99 about 890–930.
- **Maximum new entrants into the bottom-k set in one step:** 1, 2, 4, 6, 9 for k = 1, 2, 4, 8, 16.
- **Median gap q_(k+1) − q_(1):** 0.002–0.003 at k=1 and 0.016–0.030 at k=16.

## C. Candidate states (prospective, redraw readouts)

| state | exact q_min | sign mismatches | refresh rate |
|---|---|---|---|
| AUTO_1 | 0.860 | 22 | 0 |
| AUTO_2 | 0.958 | 8 | 0 |
| AUTO_4 | 0.990 | 3 | 0 |
| AUTO_8 | 0.998 | 0 | 0 |
| AUTO_16 | 0.9996 | 0 | 0 |
| GUARD_k (any η), k=1 | 1.000 | 0 | 1.000 |
| k=2 | 1.000 | 0 | 0.789 |
| k=4 | 1.000 | 0 | 0.651 |
| k=8 | 1.000 | 0 | 0.555 |
| k=16 | 1.000 | 0 | 0.452 |

- Across all guarded modes, a readout that passed the guard but was wrong happened **0** times, so every unsafe case was caught as REFRESH_REQUIRED.
- Discovery gave the same picture: GUARD refresh rate 0.422 at k=16, and AUTO_16 exact 0.9997.
- AUTO_8 and AUTO_16 had no birth-sign errors on either run. They still miss the exact q_min value 0.2% and 0.04% of the time, and they carry **no certificate**.

## D. Certified guard
The bound |Δq_w| ≤ ‖Δz_w‖ + max_c ‖ΔC_c‖ holds for unit vectors while labels are fixed, which is the case between reassigns.

The three η variants (global, basin, local candidate set) produced **identical** refresh counts. So the ‖Δz‖ term dominates: η is set by the fastest well, and summing it over 49 steps overwhelms the median bottom-k gap. The certificate is exact but conservative. A tighter per-well drift bound would need per-well information and is the main target for improvement.

## E. Event-aware (reassign) cost
- **Dirty wells per reassign** (own track changed or a neighbour's track changed): mean 16–17%, max 72–74%.
- **k_event** (entrants into one basin at one reassign): median 5.5–8, max 205–352.
- **GLOBAL_REFRESH_MATHEMATICALLY_REQUIRED at reassign.** Labels change, neighbour candidate sets change, merges relabel whole basins, and entrants can number in the hundreds. No fixed k can absorb that without re-reading those wells, so every mode here rebuilds Q_k on the engine's own reassign scan.

## G. Minimal-state verdict
1. **Exact readout:** YES. Γ = 0.9 − q_min.
2. **Autonomous dynamics:** NO fully autonomous finite-state closure was found. Between reassigns, a bottom-16 buffer without a guard is almost always right (sign errors 0), but it is not certified. At reassign, a global refresh is required.
3. **Hybrid closure:** YES, as an EXACT_HYBRID_TRACKER_WITH_CERTIFIED_REFRESH.
   - State: Q_i^(k) = Ord_k{(q_w, w)} plus the boundary q_(k+1) plus the accumulated certificate Σ η_i.
   - Q_i^+ = J_Q(Q_i^−, reads of the k tracked wells, η_i), when q_track ≤ q_(k+1) − Σ η_i.
   - Otherwise REFRESH_L1, a rescan of basin i.
   - Rebuild at every reassign by piggybacking on the L1 scan.
   - With k = 16, 55% of redraw readouts are served locally and certified exact. 45% need a basin rescan.

## Open for the next round
- A tighter drift bound. For example, a per-basin second-moment velocity, or a per-well speed sketch for the bottom-k plus a global speed quantile. This would cut the 45% refresh rate.
- Whether an event-aware partial update (re-read only the dirty wells) can replace the full rebuild at reassign.
- All of this needs a Science Supervisor contract before any formal claim.
