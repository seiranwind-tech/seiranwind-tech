# W12 — Parallel agent round on the three W11 gaps (tape replay, frozen-then-prospective)

**Line:** Claude Cloud exploratory analysis. This is **not** a Research OS or Science Supervisor ruling, and **not** an F50 closure claim.

**Setup**
- Engine: f40_v39r3_engine.py, sha 4a71aeb4…, frozen and used only as a static function library.
- Each worker fitted on 5 DISCOVERY tapes (the W11 tapes) and froze a protocol (PROTOCOL + FREEZE_LOCK sha256, committed) **before** opening 2 new PROSPECTIVE tapes.
- Prospective tapes (block, perm, seed): (2, 8080, 32) and (3, 4242, 31). Both are new partitions and new seeds. Tape 31 crashed at save and was re-recorded with identical seeds; the engine is deterministic, and nobody had read the partial file.

| stage | gap | frozen law | freeze → result commits | prospective verdict |
|---|---|---|---|---|
| W12A | core internal pull A_a | AR1_ENV: A⁺ = 0.99906·A + 1.579e-5·Env_a(M), with Env_a = point-mass pull toward the other cores | fccd20c → 4d367e4 | **FAIL**. 250-step skill 0.331 / 0.356 vs AR3 0.398 / 0.396 |
| W12B | boundary-transfer flux (residual H_T) | Poisson rate λ_ab = exp(−6.667 + 3.115·sim + (−0.193)·log n_a + 0.343·log n_b + (−1.960)·rank + 0.278·spread_a); founding rate μ_a = exp(−4.812 + 0.221·log n_a + 0.398·spread_a) | b9cf562 → b98cad9 | **PASS** on both tapes. Pooled log-lik −230 vs −513 (constant) / −335 (nearest-neighbour); AUC 0.998 vs 0.968 |
| W12C | halo ring ("outer circulation") | each halo well follows its own single-well dynamics: salt_w + 0.3·nr(attraction to point-mass cores + other halo), integrated from core-level state only | 2d84765 → 40146e7 | **PASS** on both tapes. Predicted halo gives core-force cos 0.99985 / 0.99994 (n ≥ 30, 50-step horizon); n ≥ 100 gives 0.9991 / 0.9996 |

## What this closes
- **Halo (W12C).** The halo co-moves with its core. Ring-angle std is 1.7° against a radius of about 31°, and the transported offset has cos 0.9994. Evolving the halo from core-level state reproduces the core forces, so **halo positions need no handoff**.
- **Residual H_T (W12B).** Boundary transfers become a **macro flux law** over (sim(M_a, M_b), n_a, n_b, rank, spread_a). It is well discriminating (AUC 0.998) and destinations are nearest-neighbour. Caveats:
  - Total event counts are over-predicted prospectively: 1.55 predicted vs 0.95 actual events per gate.
  - The founding law is only modestly better than a constant rate.
  - Event counts are small, 25–59 per tape.

## What stays open
- **Core internal pull A_a (W12A).** No tested law beat the universal AR3 on centre increments (AR3 250-step skill about 0.40 prospectively).
  - The internal pull remains the single unclosed state variable in the core-centre equation.
  - Explosive fits (ρ > 1) and tiny environment coefficients suggest that A_a is driven by core-internal shape, not by the environment. This agrees with W11: the shape is not slaved to M and needs about 32 principal components.

## Candidate L1.5 macro system after W12 (exploratory)
```
M_a⁺ = nr(M_a + 5dt·[S_MF,a(M_a) + A_a])                 # salt mean-field closed (W11)
A_a  : latent internal pull — OPEN (best available: universal AR3 on ΔM)
halo : h_w⁺ = own dynamics(h_w; {M_b, n_b}, halo)        # closed prospectively (W12C)
flux : transfers a→b ~ Poisson(λ_ab(sim, n_a, n_b, rank, spread_a))   # closed prospectively (W12B)
```
The system is not autonomous: A_a still has to be carried empirically, or handed off.
