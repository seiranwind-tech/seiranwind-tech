# W7 protocol — extreme order-statistic L1.5 tracker
This file was frozen before any W7 data was generated.

**Instruction:** CLAUDE_W7_EXTREME_ORDER_STAT_CLOSURE.md, sha be94508d….
**Engine:** f40_v39r3_engine.py, sha 4a71aeb4…, unmodified.
**Vectors:** GloVe compact20k, sha a0997bb3….
**Run length:** T = 6000; tracker statistics start at step 1500.

## Data
- **Discovery** reuses W6 DEV blocks 0 and 1 (permutation seed 12345, engine seed 1). It is descriptive only.
- **Prospective** uses a NEW word partition (permutation seed 777), blocks 0 and 1, with engine seed 3. These runs are generated only after this protocol and w7_tracker.py are hashed into W7_FREEZE_LOCK.txt.

## Exact target (read from the engine's `_reassign_sparse`)
- q_w = max(s_own,w, s_best,w), where s_best,w = max over centres of labels[neighbors[w]].
- q_min(i) = min over w in basin i of q_w.
- Γ_i = 0.9 − q_min(i). A birth needs Γ_i > 0 at a reassign step (completed_step mod 50 == 0).

## Certified bound
Between two reassign steps, labels and the neighbour graph are fixed (neighbor_rebuild_every = 0), so each well's candidate-centre set is fixed.

For unit vectors, |Δ(z·C)| ≤ ‖Δz‖ + ‖ΔC‖. Hence |Δq_w| ≤ ‖Δz_w‖ + max_c ‖ΔC_c‖. (Differences of maxima are bounded by the largest term difference.)

- η_global(t) = max over all wells ‖Δz‖ + max over all centres ‖ΔC‖. This needs one scalar broadcast from L1.
- η_basin_i(t) = max over w in i of ‖Δz_w‖ + max over all centres ‖ΔC‖. This needs one scalar per basin from L1.

- η_local_i(t) = max over w in i of ‖Δz_w‖ + max over w in i of max over c in cand(w) of ‖ΔC_c‖, where cand(w) = {label_w} ∪ labels(neighbors[w]). This is certified by the same inequality, and needs one scalar per basin from L1.

## Candidate states (fixed in advance, no fitted parameters)
k ∈ {1, 2, 4, 8, 16}. Q_k = bottom-k (witness id, q), plus the boundary b = q_(k+1) (+∞ if n ≤ k).

Modes:
- **AUTO_k:** rebuild Q_k at the reassign scan; between scans, read only the k tracked wells; the readout is the tracked minimum. There is no guard.
- **GUARD_k_global / GUARD_k_basin / GUARD_k_local:** as AUTO_k, but at the readout step the cumulative η must satisfy tracked_min ≤ b − Σ η. If it does, the local readout is used (it must be exact). If not, emit REFRESH_REQUIRED and do one L1 basin rescan.

## Reassign steps
Labels change for many wells, the neighbour-label candidate sets change, and there are births and merges. This is treated as a global rebuild that piggybacks on the engine's own reassign scan.

Event-aware cost is reported as the "dirty" fraction: wells whose own track changed or whose neighbours changed track. The maximum number of entrants per basin, k_event, is reported too.

## Verdict rules
- **FULLY_AUTONOMOUS_FINITE_STATE_CANDIDATE:** some fixed k gives zero mismatches with zero refreshes, including across reassign steps.
- **EXACT_HYBRID_TRACKER_WITH_CERTIFIED_REFRESH:** guard-pass readouts show zero mismatches, and every unsafe case is caught as REFRESH_REQUIRED, on all prospective runs.
- **LOW_DIMENSIONAL_TRACKER_INSUFFICIENT:** anything else.
