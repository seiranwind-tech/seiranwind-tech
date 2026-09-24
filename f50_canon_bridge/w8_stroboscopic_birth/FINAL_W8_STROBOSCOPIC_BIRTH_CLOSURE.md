# W8 — Stroboscopic birth-channel closure: final report

**Line:** Claude Cloud exploratory analysis. This is not a Research OS or Science Supervisor ruling, and it is not a formal F50 closure.

## Inputs
- **Engine:** sha 4a71aeb4…, unmodified.
- **Vectors:** GloVe compact20k, sha a0997bb3….
- **Instruction file:** sha 20dbf290….
- **Protocol and harness:** frozen before any W8 data. PROTOCOL_W8.md 61303239…, w8_harness.py af3bf867… (W8_FREEZE_LOCK.txt, commit 451e7aa). The lock was re-verified as OK after the runs.
- **Prospective runs:** 2 new word partitions (4242, 9191) × 2 new engine seeds (5, 7), 6000 steps each. This gave 364 reassign gates, 43,000 basin readouts, and 301 basins with births.
- **Discovery run:** permutation 12345, block 0, seed 1. Descriptive only.

## Verdicts

| claim | result |
|---|---|
| A. EXACT_STROBOSCOPIC_BIRTH_SIGN_HANDOFF | **PASS**: 0 sign mismatches in 43,000 readouts, and no additional high-dimensional pass |
| B. EXACT_STROBOSCOPIC_BIRTH_SET_HANDOFF | **PASS**: 0 count mismatches against the engine's own counter, 0 mismatches against lineage births, 0 identity violations, 0 budget-selection mismatches. The 256 budget was never reached (max 10 orphans per gate), so the budget ordering was never exercised. |
| C. ZERO_EXTRA_REASSIGN_SCAN | **PASS**: every handoff comes from (own, best_score, labels), which `_reassign_sparse` already computes. The extra work is O(N): a max, a segmented min, a count and a where. |
| D. FULLY_AUTONOMOUS_PER_STEP_L1.5 | **NOT CLAIMED / FAIL by design**: the design uses the periodic L1 gate. |

Engine-array identity: the harness's orphan count equals the engine's `_birth_orphan_candidates`, and its near-miss count equals `_birth_near_misses`, at every gate (0 mismatches). This is strong evidence that q is built from the engine's own arrays.

## Answers
1. **Is one q_min enough for the exact birth sign?** Yes. Γ_i = 0.9 − q_min,i, with 0 mismatches.
2. **Is q_min enough for exact birth dynamics?** No. **96 of 301 birth basins (32%) had several simultaneous births**, up to N_B = 7 in one basin and up to 10 orphans in a gate. Multiplicity histogram: 1:205, 2:57, 3:21, 4:8, 5:5, 6:3, 7:2. Each orphan becomes its own new basin, so the post-birth state depends on who the orphans are.
3. **Minimum exact payload for multiplicity plus identity:** H4 = q_min per basin plus (well id, own) per orphan. That is about **483 bytes per gate** (H1 alone is 473 B). own is only needed to order the 256 budget, which was never reached, so id alone would do empirically. H3 (bottom-k) never overflowed even at k = 8, but costs 5.2–15.6 KB per gate, which is 11–32× H4.
4. **Does it need a new high-dimensional pass?** No. NO_ADDITIONAL_HIGH_DIMENSIONAL_SCAN = true.
5. **Can W7's 45% refresh be removed from the exact gate path?** **Yes, entirely.** The engine only decides births at the 50-step gate, and there q_min and O_i come from the base scan. W7's bottom-16 tracker stays only as an EARLY_WARNING_PREDICTOR. Re-seeding it after a reassign from (stale pre-reassign bottom-16 ∪ that reassign's entrants) contains the next argmin 99.983% of the time. That is not exact, so this reset is heuristic.
6. **Birth channel isolated. What is still missing for full L1.5?**
   - **Transport** (reassign moves with best ≥ 0.9 > own) comes from the same base scan but was not audited here.
   - **Merge** (centre–centre cosine ≥ 0.98 on the basin graph) needs only L1.5 centre state; not audited here.
   - **The continuous dynamics between gates** (member and centre motion, which W5–W6's smooth law only approximates) are not closed.

## Result equations (birth channel, stroboscopic, exact on this data)
q_w(t_r) = max(s_own,w, s_best,w)   (from the base scan at t_r = 50, 100, …)

q_i^(1)(t_r) = min over w in i of q_w(t_r)

Γ_B,i(t_r) = 0.9 − q_i^(1)(t_r)

B_i(t_r) = 1[Γ_B,i > 0]

O_i(t_r) = {w : q_w(t_r) < 0.9},   N_B,i = |O_i|

Q_L1.5(t_r^+) = RESET(H4(t_r))

(If |O| > 256: keep the 256 with the lowest own_w.)

**Claim ceiling:** exact on 4 new GloVe runs. The birth channel only; not full L1.5 closure.
