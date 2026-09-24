# W8 protocol — Stroboscopic birth-channel closure
This file was frozen before any W8 data was generated.

- **Instruction:** CLAUDE_W8_STROBOSCOPIC_BIRTH_CLOSURE.md, sha 20dbf290….
- **Engine:** sha 4a71aeb4…, unmodified.
- **Vectors:** GloVe compact20k, sha a0997bb3….
- **Run length:** T = 6000; statistics start at step 1500.

## Data
- **Discovery (descriptive only):** permutation 12345 block 0, engine seed 1. This is W6/W7 material.
- **Prospective, generated only after W8_FREEZE_LOCK.txt exists.** Two new word partitions (permutation seeds 4242 and 9191) crossed with two new engine seeds (5 and 7):
  - (4242, b0, s5)
  - (4242, b1, s7)
  - (9191, b0, s5)
  - (9191, b1, s7)

## Base scan (from `_reassign_sparse`, read-only)
- own_w = zin_w · C[label_w]
- best_w = max over k of zin_w · C[label[nbr_wk]]
- orphan_w = (best_w < thr) ∧ (own_w < thr), since birth_requires_own_centre_miss = True
- If there are more than 256 orphans, the engine keeps the 256 with the lowest own.

The harness recomputes own and best with the engine's own formulas, for audit only. It checks them against the engine's own counters `_birth_orphan_candidates` and `_birth_near_misses`, and against lineage births.

## Handoff, computed from (own, best, labels) only
q_w = max(own_w, best_w). Per basin: q_min, N_B = |{q_w < thr}|, and the orphan set O_i.

Handoff definitions:
- **H1:** q_min. One float32 per basin.
- **H2:** (q_min, N_B). float32 + uint16 per basin.
- **H3_k:** bottom-k of (q, well id, own) plus q_(k+1), for k ∈ {8, 16, 32, 64}. Overflow means N_B > k.
- **H4:** q_min per basin, plus (well id, own) for each orphan.

Payload sizes: float32 = 4 bytes, id = 4 bytes, count = 2 bytes.

## Checks at every prospective reassign
1. qmin sign vs engine births per basin
2. orphan count vs the engine's `_birth_orphan_candidates`
3. orphan-ID set vs wells that received new track ids (merge absorption is allowed; predicted ⊇ observed, and the sizes must match lineage births)
4. budget selection
5. H3 overflow
6. payload bytes
7. extra high-dimensional operations
8. post-reassign reset: can the stale pre-reassign bottom-k plus that reassign's entrants seed the next interval's argmin?

## Verdicts
- **A. EXACT_STROBOSCOPIC_BIRTH_SIGN_HANDOFF:** 0 sign mismatches and no additional high-dimensional pass.
- **B. EXACT_STROBOSCOPIC_BIRTH_SET_HANDOFF:** count, identity and budget all have 0 mismatches.
- **C. ZERO_EXTRA_REASSIGN_SCAN:** every handoff is derivable from own, best and labels.
- **D. FULLY_AUTONOMOUS_PER_STEP_L1.5:** FAILS by construction here, because this design uses a periodic L1 gate. It is not claimed.
