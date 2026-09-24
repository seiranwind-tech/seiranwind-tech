# W1 (canonical line) — CANON-BRIDGE-R1 observable constitution

Engine: F40 V3.9R3 `f40_v39r3_engine.py`, sha256 4a71aeb4a66a8a07…. This is the engine SHA cited in the Research Admin preflight. It was supplied by the user in F50_G3_H2_V11_ENGINEERING_HARDENING_HOTFIX2.zip and is not modified.
Config: frozen defaults; profile F40_V3_9_NATIVE_L1_NONCANONICAL, N=2000, d=300, EXACT_GLOBAL interaction, assignment_threshold=0.9, basin_reassign_every=50, basin_update_every=5, neighbor_k=20.
**Deviation:** the vectors are spaCy en_core_web_md-3.8.0 (300d), not GloVe, because the GloVe download was blocked in this container. Each run uses a disjoint block of 2,000 rows (permutation seed 12345). DEV = blocks 0–3; FRESH = blocks 4–7, generated only after both freezes.

Hidden gate, replicated read-only from `_reassign_sparse`:
- s_own = zin·C[label]
- s_best = max over the 20 reachable neighbour wells' basin centres (this can include the well's own basin)
- Γ_birth = 0.9 − max(s_own, s_best)

Fidelity check: recorded births equal the engine's lineage birth count (137/137 in the smoke test; 1,502/1,502 on DEV).

Row = (block as `seed`, step, basin track id) for n ≥ 2, from step 800 on. Window convention: redraw_index = (step−1)//50 and phase = (step−1)%50, so the reassignment step is phase 49.

Column classes are the same as the PU line:
- LEGAL: n, age, R, skew, yo, yb, cob, dR, dskew, dmu. C_rival is the best centre in the engine's own basin-neighbour graph.
- PROPOSED_AUGMENTATION: so, sb, rho.
- HIDDEN: G_birth (the target), G_shape (a diagnostic).
- The events file holds births (hidden) and moves (realised centre movement W per track, per reassignment).
