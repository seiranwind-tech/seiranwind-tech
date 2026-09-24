# W6 fixed protocol (written before any GloVe DEV result was read)

- **Engine:** f40_v39r3_engine.py, sha 4a71aeb4…, frozen defaults. N=2000, d=300, threshold 0.9, reassign every 50 steps.
- **Vectors:** glove.6B.300d.compact20k.w2vbin, sha a0997bb3…, read by the engine's own WORD2VEC_BINARY loader.
- **Run length:** T = 6000 steps, recording from step 1500. This is fixed now and will not be changed after FRESH is seen.
- **Blocks:** permutation seed 12345, blocks of 2,000 words. DEV = 0–3, FRESH = 4–7, FRESH2 = 8–9.
- **Held-out isolation:** FRESH and FRESH2 stay sealed until CANDIDATE_FREEZE.json and TIMING_FREEZE.json are hashed into FREEZE_LOCK.txt.
- **What DEV decides:** γ, β, the derivative window, and whether a single rescue scalar is admitted. Rescue is chosen by DEV-internal validation: fit on blocks 0–2, validate on block 3. The candidates are rival_gap, cob, sb_tail and rho.
- **spaCy use:** spaCy data was used only for a code dry-run. No GloVe coefficient comes from spaCy. B0 is the spaCy W5 F3 formula, transferred unchanged.
- **Pre-declared insufficient-sample rule:** if DEV has fewer than 200 birth events at redraw steps, run replicate DEV trajectories on blocks 0–3 with engine seed = 2, under the same protocol, and pool them into DEV. Held-out data is never touched to decide this.
- **Closure levels:**
  - L1 requires held-out redraw precision ≥ 0.95 and recall ≥ 0.95.
  - L2 requires zero held-out mismatches and zero mixed-sign fiber cells.
  - Anything else is reported as NOT DETERMINISTICALLY CLOSED.
