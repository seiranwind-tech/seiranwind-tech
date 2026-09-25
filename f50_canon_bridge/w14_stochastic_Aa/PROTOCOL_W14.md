# W14 — Statistical closure of the core-centre residual force A_a

**Line:** exploratory Claude line. Not a Research OS / Science Supervisor ruling, not an F50 closure claim.

**Idea (W13's option 1):** treat the unclosed internal pull A_a as a coloured stochastic
force on top of the universal AR3 drift, and test whether an effective SDE reproduces the
*distribution* of core-centre displacements, instead of the trajectory itself.

## Model

- Drift: universal AR3 on centre increments d_t (frames = every 5 steps), frozen coefficients
  from W11: `coef = [1.1639, 0.5491, -0.7167]`.
- Residual: `r_t = d_{t+1} - AR3(d_t, d_{t-1}, d_{t-2})`, fit on the FIT tapes
  (river partition h=12, g0=60, minsize=5, per W11's `river_series`).
- SDE step: `M' = nr(M + AR3(d) + xi)`, with `xi` isotropic Gaussian tangent noise,
  AR(1)-coloured: `xi_t = rho*xi_{t-1} + sqrt(1-rho^2)*sigma(n)*eps_t`, `eps_t ~ N(0, I_D)`, `D=300`.
- Amplitude law: `sigma(n)^2 = sigma0^2 * n^alpha` (pooled per-dimension residual variance,
  averaged within each unique core size n across the FIT tapes, log-log fit).

## Frozen parameters (fit on 5 FIT tapes, `$S/tapes/*.npz`)

```
sigma0  = 2.9494719317454124e-05   (per-dim residual std at n=1)
alpha   = -0.8269400567340163       (sigma^2 ~ n^alpha: bigger cores => quieter residual)
rho     = -0.21832198320831697      (lag-1 autocorrelation of the AR3 residual, pooled)
D       = 300
n_samples = 72345, n_series = 761
ang_per_step_deg (at n = n_ref = 47, median core size) = 0.00596 deg
```

Fit script: `w14_fit.py` (residual defined exactly as `r_t` above, one-step-ahead,
using the TRUE history at every t — not a closed-loop rollout).

## Evaluation protocol (frozen before opening TEST tapes)

For each TEST tape, for each core-centre series `M` (n >= 5, gates >= 60):
- at each start t0, roll out **50 stochastic draws** of the SDE for 10 and 50 frames
  (= 50 and 250 steps), and also the pure-deterministic AR3-only rollout `Md`.
- (i) mean and p10/p50/p90 of angular displacement `ang(M(t0+k), M(t0))`, simulated vs true.
- (ii) coverage: fraction of true one-instance angular error `ang(M_true(t0+k), Md(t0+k))`
  that falls at or below the **simulated ensemble's p90** of `ang(M_sim(t0+k), Md(t0+k))`.
  Target ~ 0.90.
- (iii) Kolmogorov-Smirnov statistic between the pooled simulated displacement distribution
  (50 draws x all starts) and the pooled true displacement distribution, at each horizon.

## Pass criterion (frozen)

On **each** of the 4 TEST tapes (`$S/ptapes/*.npz`, `$S/ptapes2/*.npz`), at the 250-step horizon:

```
coverage in [0.85, 0.95]  AND  KS statistic < 0.1
```

A tape failing either condition is a FAIL for that tape. This is an exploratory statistical
test, not a formal closure; a FAIL here would extend W13's finding that A_a's O(n) content
resists a low-dimensional closure, this time in distribution rather than trajectory.

## Self-check on a held-in FIT tape (not a substitute for the TEST verdict)

`tape_0_1357_11.npz`: 50 steps — sim/true disp mean 1.73/1.76 deg, KS 0.054, coverage 0.0.
250 steps — sim/true disp mean 5.80/7.77 deg, KS 0.315, coverage 0.0.
The displacement *mean* is tracked reasonably; the coverage criterion is not met even in-sample,
because the true rollout diverges from the deterministic AR3 path faster (closed-loop, chaotic
compounding of the unmodeled force) than the fitted **open-loop, one-step** residual variance
predicts when simply integrated. This is disclosed here before touching the TEST tapes.
