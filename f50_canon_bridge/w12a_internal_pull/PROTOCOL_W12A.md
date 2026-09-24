# PROTOCOL W12A — frozen law for the core's latent internal pull A_a

**Line:** Claude Cloud exploratory analysis (F50 L1.5 closure, W12A gap). Not a Research OS or
Science Supervisor ruling, not a formal F50 closure claim.

## 1. The law (exact, frozen)

State per river core a (n_a ≥ 5 members, fixed partition per 10-gate block, h=12): centre M_a,
latent vector A_a. Environment: the OTHER cores in the same block, as point masses (M_b, n_b);
halo wells are **dropped** from the environment term in this frozen version (see §5 caveat).

```
M_a⁺ = normalize( M_a + 5*dt * S_MF,a(M_a) + A_a )     # salt mean field, closed (R^2 0.99, w11_meanfield.py)

Env_a(M) = tangent_{M_a}[ attract_gain * normalize( softmax_b(M_a·M_b / bw + log n_b) @ M  -  M_a ) ]
           softmax excludes self (b=a); bw = eng.bandwidth (engine's own kernel/gain, unmodified)

A_a⁺ = rho * A_a + beta * Env_a(M)      (gamma = 0: the balance/relaxation term BAL was fit and
                                          rejected -- see §3)
```

Frozen coefficients (fit by pooled isotropic least squares over A_a(t+1) ~ rho*A_a(t) + beta*Env_a(t)
+ gamma*S_MF,a(t), on 12,560,700 pooled (core x dim) samples from the 3 training discovery tapes,
late regime only: gates ≥ 60, cores n ≥ 5):

```
rho   = 0.9990596941276166
beta  = 1.579285859519589e-05
gamma = 0.0            (BAL term dropped)
```

This is the **AR1_ENV** law: it is the fitted 2-parameter model on {AR(1) persistence, environment
point-mass drive}, chosen over 6 other candidates (see §3) because it was the *only* candidate that
was not dominated by plain AR1 on both held-out discovery tapes, while the full 3-parameter fit was
explosive (rho > 1) and its clipped/balance variants were all worse. It is still, honestly, **not
expected to beat the AR3 baseline** (see §4) -- it is frozen as the best of a negative result, per
instruction.

Off-core (small, n<5) river groups are advected by S_MF alone (A=0 there), matching w11_rollout.py's
convention, since eng._fsalt needs the true per-well teeth for ALL wells (via the fixed partition's
inv/K), not just the cores being scored.

## 2. Candidates tried

1. **AR(1) relaxation to zero** (existing w11 L3 baseline; frozen A decaying, rho≈0.99): included
   as subset "AR1".
2. **Relaxation to balance**, A⁺ = A + (−gamma·S_MF(M) − A)/tau, folded into the pooled regression
   as the "BAL" regressor gamma·S_MF,a(M_a(t)): subset "BAL", and combined "AR1+BAL".
3. **Environment-driven**, A driven by the tangent pull toward point-mass OTHER cores, weighted by
   the engine's own softmax kernel (bandwidth eng.bandwidth, weight log n_b): subset "ENV", and
   combined "AR1+ENV".
4. **Full combination** rho·A + beta·Env + gamma·S_MF: subset "AR1+ENV+BAL" ("FULL").
5. Halo wells were **not** added to Env_a individually (candidate left for future work, see §5).

All 7 subsets (single, pairs, full) were obtained from one pooled 3x3 normal-equation fit (cheap,
no extra data passes) using `w12a_fit.py`.

## 3. Fit / validate results (discovery tapes only)

Fit (train) tapes: tape_0_1357_11, tape_1_1357_12, tape_0_9753_21 (12,560,700 pooled samples).
Validate (held-out) tapes: tape_2_1357_13_zo, tape_1_9753_22_zo.

Raw fitted coefficients (rho, beta, gamma):
- AR1: (0.99932, 0, 0)
- ENV: (0, 0.02551, 0)
- BAL: (0, 0, -0.11837)
- AR1+ENV: (0.99906, 1.579e-05, 0)
- AR1+BAL: (1.00573, 0, 0.001031)
- ENV+BAL: (0, 0.01035, -0.09920)
- AR1+ENV+BAL ("FULL"): (1.00563, 5.537e-06, 0.001030) — **rho > 1, explosive**

Rollout skill (1 − err/disp, angle-weighted) at 250 steps, on the 2 held-out discovery tapes
(FULL_clip / AR1_BAL_clip clip rho to ≤ 1):

| law | tape_2_1357_13_zo | tape_1_9753_22_zo |
|---|---|---|
| **L0 AR3 baseline** | **0.4238** | **0.4842** |
| FULL (unclipped) | 0.3327 | 0.3752 |
| FULL_clip (rho=1) | 0.2425 | 0.2920 |
| AR1_BAL_clip (rho=1) | 0.2393 | 0.2888 |
| AR1_ENV (chosen) | 0.3902 | 0.4235 |
| AR1 (plain) | 0.3853 | 0.4187 |

At 5 and 50 steps all candidate laws are close to but slightly below AR3 as well (e.g. AR1_ENV:
0.939/0.807 and 0.945/0.825 vs AR3's 0.957/0.881 and 0.958/0.894). **No candidate law beats the
AR3 baseline at any of the reported horizons on either held-out discovery tape.** The clipped and
balance-driven variants are markedly worse (they still drift, i.e. clipping rho to 1 does not by
itself produce stability against the fixed-partition population change). AR1_ENV is chosen as best
because it is the least-bad law and the only one where the environment term measurably helps over
plain AR1 (both held-out tapes), consistent with A_a being "slow" (residual AR1 coefficient ≈0.998,
close to our fitted 0.99906) with a small, second-order environment correction.

## 4. Pass criterion

**Pass**: AR1_ENV's rollout skill at 250 steps (angle-weighted, 1 − err/disp) must exceed the AR3
baseline's skill at 250 steps, on **both** prospective tapes (tape_3_4242_31_zo, tape_2_8080_32_zo),
evaluated once, with no retuning after this freeze.
**Fail**: otherwise (which, per §3, is the expected outcome — this closes W12A as a documented
negative result, not a discovered law).

## 5. Caveats (honest, not resolved here)

- **Halo dropped.** Env_a uses only the other CORES as point masses; the halo ring (~1-1.3% of
  wells, kept individually in w11_decouple.py's exact force reconstruction) is not included, for
  compute-cost reasons. Per the decouple table this mainly matters for n≥100 cores; it may explain
  part of why AR1_ENV still underperforms AR3.
- **Isotropic, scalar coefficients.** rho/beta/gamma are shared across all 300 embedding dims and
  all core sizes; no per-size or per-dimension structure was fit.
- **Other cores' future positions are co-evolved** with the same salt-MF+A law during rollout
  (not frozen at t0), but the halo and any macro-event (boundary transfer, wanderer founding) are
  not modeled at all during rollout.
- Frames/cores were not further subsampled in the reported runs (all late-regime blocks, all
  cores n≥5, t0 stride 10 gates); this fit comfortably within the ~8 minute budget per tape.
