# W10 protocol — member-motion / H_S closure audit
Frozen before any W10 prospective data. Only a 120-step smoke test was run, on the discovery partition.

- **Instruction:** CLAUDE_W10_MEMBER_MOTION_CLOSURE.md, sha 10bdfbcc….
- **Engine:** sha 4a71aeb4…, unmodified.
- **Vectors:** GloVe compact20k, sha a0997bb3….
- **W9 shadow:** sha 07ef512c…, imported unchanged.
- **Discovery** (permutation 12345, b0, seed 1): w10_discovery_audit.py builds the dependency graph and force decomposition. Samples fall at cs mod 25 = 12, from step 1500 to 6000. It also runs the counterexamples P1 and P2 and a compression audit. Diagnostics only.
- **Prospective (new):** (perm 2468, b0, seed 13) and (perm 8642, b1, seed 17), T = 6000, run with w10_cadence.py.

## Frozen candidates
- **Cadences:** H_S delivered every {5, 10, 15, 20, 25, 50} steps, and always at gates (the base scan runs there anyway).
- **Predictors** between deliveries:
  - HOLD: the last delivered observed centres.
  - EXTRAP: linear extrapolation of the last two deliveries.
  - Cadence 5 is the exact W9 control.
- **Moment states S0–S3:** tested by construction. They have no fitted coefficients.
  - **P1:** rotate a basin's members inside a plane orthogonal to m. This keeps n and m exactly and changes M₂.
  - **P2:** swap the zin of two members of the same basin. This keeps n, m, M₂ and every higher zin moment of the basin exactly; only the pairing of zin with each well's static teeth and hidden zout changes.
  - If Δm differs under P2, no function of the basin's zin moments of any order determines Δm.
- **Low-rank basis regression (W10-E):** not run, because P2 decides the question. Any finite basis built from L1.5 state is a function of moments. A basis fit would be an approximation, not exact closure.

## Verdict rules
- **EXACT_AUTONOMOUS_HS_CLOSURE / FINITE_MOMENT_HS_CLOSURE:** FAIL if the P2 or P1 counterexamples show Δm differences well above float noise.
- **MINIMUM_PERIODIC_HS_HANDOFF:** the smallest cadence that stays bitwise exact. Also reported: the smallest cadence with 0 gate-birth mismatches, and the one with track agreement ≥ 0.99.
- **PAYLOAD_COMPRESSION_ONLY:** lossless zlib / XOR-delta ratios.
