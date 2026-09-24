# W10 — Member-motion / H_S closure audit: final report

**Line:** Claude Cloud exploratory analysis. This is not a Research OS or Science Supervisor ruling, and it is not a formal F50 closure.

## Inputs
- **Engine:** sha 4a71aeb4…, unmodified.
- **Vectors:** GloVe compact20k, sha a0997bb3….
- **W9 shadow:** sha 07ef512c….
- **Protocol and code:** PROTOCOL_W10 b0447a43…, frozen in commit b729831 before any prospective data. The lock was re-verified as OK afterwards.
- **Discovery:** permutation 12345, b0, seed 1.
- **Prospective (new):** (2468, b0, s13) and (8642, b1, s17), each 6000 steps.

## Exact member-motion equation (verified: reconstruction max error 3e-8)
z_w⁺ = normalize(z_w + dt·d_w), where d_w = salt_w + wave_w + 0.3·A_w:

- salt_w = ½·[f_salt(z_w; teeth_w, mask_w, skew_w) − z_w(z_w·f_salt)]
- wave_w = 0.6·(zout_w⁺ − zop_w)/wall, where zout_w⁺ = clip_wall(zout_w + dt·(0.3·blade_w(zout_w) + f_salt(zout_w)))
- A_w = normalize(Σ_v softmax_v(z_w·z_v / h)·z_v − z_w), summed over **all N wells**

Δm_i = Σ over w in i of [normalize(z_w + dt·d_w) − z_w].

## Dependency graph (DEPENDENCY_GRAPH_W10.json)
- **Member-level L1:** z_w.
- **Member-level hidden field:** zout_w and zop_w, with their own blade, salt and wall dynamics.
- **Static but per-well:** teeth_w, mask_w and skew_w (N × 4 × 300).
- **Global N-body kernel:** A_w.
- **Per-well normalization.**
- **Not exactly aggregatable:** none of the terms in Δm_i reduces to basin aggregates.

## Force decomposition (FORCE_DECOMPOSITION_AUDIT.json, 180 samples)
Median basin norms, n ≥ 5:
- |Δm| = 0.019
- |Σ dt·salt| = 0.044
- |Σ dt·0.3·A| = 0.065
- |Σ dt·wave| = 0.0007
- **|R_norm| = 0.038, which is 3.1 × |Δm|** (median ratio)

Net member motion is a small remainder left after large memberwise cancellation: the attraction and salt forces are mostly removed by the per-well normalization. Wave is small in magnitude, but it carries the hidden zout field.

## Sufficiency counterexamples (MOMENT_SUFFICIENCY_AUDIT.json, 12 constructions)
- **P2, swapping the zin of two members of the same basin.** m_i and M₂,i are unchanged within float noise (≤ 4e-6 and ≤ 1e-6), and so is every higher zin moment. **Yet Δm_i changes by 4.0–11.5% of |Δm_i|**, 1,000–3,000 times the noise.
  → Δm_i is not a function of the basin's zin moments of any order. The pairing of each zin with that well's static teeth and hidden zout matters.
- **P1, rotating members in a plane orthogonal to m.** n and m are unchanged; M₂ changes by about 7e-4. Δm changes by 0.4–2.5%.
  → The first moment alone is insufficient.

## Cadence ablation (PROSPECTIVE_W10_RESULTS.json, 2 prospective runs + discovery)
| H_S cadence | bitwise exact | gate birth-count mismatches (of 90) | mean track agreement |
|---|---|---|---|
| **5 (every centre update)** | **yes, 3 of 3 runs** | **0** | **1.000** |
| 10 | no, diverges at the first missing update | 5–12 | 0.85–0.99 |
| 15 | no | 10–23 | 0.85–0.97 |
| 20 | no | 14–34 | 0.88–0.92 |
| 25 | no | 24–50 | 0.77–0.90 |
| 50 | no, collapses: 3,500–4,500 spurious births | 89–100 | about 0.10 |

Neither predictor (HOLD or linear EXTRAP) restores exactness or zero event mismatch at any cadence above 5.

## Payload (PAYLOAD_COMPRESSION_AUDIT.json)
- **Per H_S update:** about 126 KB raw at K ≈ 105 (1,200 B per basin).
- **Amortized over engine steps:** about 25 KB per step.
- **Best lossless compression:** XOR against the previous delivery plus zlib gives **1.29×**. The payload is essentially incompressible. This is PAYLOAD_COMPRESSION_ONLY.
- **Correction to W9's figure:** W9's "≈80 KB/step" averaged over all steps, including the early phase with K ≈ 2000.

## Verdicts
| claim | result |
|---|---|
| EXACT_AUTONOMOUS_HS_CLOSURE | **FAIL** (P2: Δm depends on member-level pairing with zout and teeth) |
| FINITE_MOMENT_HS_CLOSURE | **FAIL** for any finite, or even complete, set of basin zin moments (P2). A state that could close it would have to include per-well zout and teeth, i.e. member level. |
| APPROXIMATE_HS_CLOSURE_CANDIDATE | Not established. The frozen predictors give bounded but event-changing errors from cadence 10 on. |
| **MINIMUM_PERIODIC_HS_HANDOFF** | **Every 5 steps**, the engine's own centre-update cadence. It cannot be thinned without event errors. |
| PAYLOAD_COMPRESSION_ONLY | ≤ 1.29× lossless |

## Answers
1. **Can m_i be pushed forward exactly by a finite L1.5 state?** No.
2. **Is the first moment enough?** No (P1).
3. **Does the second moment solve it?** No. P2 keeps M₂ and all higher moments and still changes Δm.
4. **What causes the non-closure?** **Memberwise structure:**
   - per-well normalization (R_norm ≈ 3 × Δm);
   - the per-well hidden zout feeding the wave term;
   - per-well static teeth in salt;
   - the global kernel in attraction.
   No single force channel is the culprit; the obstruction is that the update acts per well before summing.
5. **Minimum H_S handoff:** every 5 steps.
6. **Lossless compression limit:** about 1.29×.
7. **What is left for full autonomy:** H_S is **proven not closable** from basin-level state; it is an irreducible L1 channel under this engine. **And even if H_S were closed, the H_T / transport gate handoff** (per-well best labels at each reassign) **would still not be eliminated.** H_S is not the only remaining input.

## Mathematical status of L1.5 after W8–W10
L1.5 is an **exact hybrid system**:

X_L1.5⁺ = F(X_L1.5, H_S every 5 steps, H_T every 50 steps)

F is fully explicit and bitwise exact (W9). H_S cannot be derived from X_L1.5 or any finite moment extension (W10). So L1.5 cannot be fully separated from L1 for this engine; the minimum coupling is (H_S every 5 steps, H_T every 50 steps).
