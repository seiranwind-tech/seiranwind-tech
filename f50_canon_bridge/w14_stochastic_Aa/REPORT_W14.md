# W14 — Statistical closure of A_a: prospective result

**Line:** exploratory Claude line. Not a Research OS / Science Supervisor ruling, not an F50 closure claim.

**Protocol:** frozen in `PROTOCOL_W14.md` (commit `ce4dc2d`) before opening any TEST tape.
Evaluated once on the 4 TEST tapes: `$S/ptapes/*.npz`, `$S/ptapes2/*.npz`.

## Frozen law

`sigma0 = 2.9495e-05`, `alpha = -0.8269`, colour `rho = -0.2183` (AR(1) lag-1, pooled residual),
`D = 300`.

## Result (250-step horizon, pass = coverage in [0.85, 0.95] AND KS < 0.1)

| tape | coverage | KS | verdict |
|---|---|---|---|
| tape_2_8080_32_zo | 0.00 | 0.336 | FAIL |
| tape_3_4242_31_zo | 0.00 | 0.404 | FAIL |
| tape_3_6262_42_zo | 0.00 | 0.309 | FAIL |
| tape_4_5151_41_zo | 0.00 | 0.313 | FAIL |

**Overall: FAIL** on all 4 TEST tapes (matches the pre-registered FIT-tape self-check exactly,
so this is not a surprise from freezing too early).

At 50 steps the picture is better: KS is 0.043-0.065 (passes the <0.1 bar) and the mean/p10/p50/p90
of the *displacement* distribution `ang(M(t0+k), M(t0))` are tracked closely (e.g. true p50 1.14-1.26 deg
vs sim p50 1.09-1.22 deg). So the SDE reproduces short-horizon displacement statistics reasonably well.

## Why it fails at 250 steps

The coverage test compares against the deterministic AR3-only path `Md`, i.e. it asks whether
the fitted noise amplitude explains how far the **true** trajectory strays from the pure AR3
prediction. The true rollout diverges from `Md` far faster than the SDE's noise does:
at 250 steps, true error from `Md` averages ~4-9 deg across tapes, while the SDE's simulated
spread around `Md` is only ~0.04-0.07 deg -- roughly two orders of magnitude too small.

The residual `r_t` was fit **open-loop** (one-step-ahead, using the true history at every t),
giving a tiny per-step scale (~0.006-0.01 deg). But the true error against `Md` accumulates
**closed-loop**: each step's unmodeled force feeds into the next step's own drift, and this
compounding is much stronger than an i.i.d. random walk with the open-loop residual variance
would produce, even with AR(1) colouring at the fitted (weakly anti-persistent) `rho`. A single
Gaussian-per-step, short-memory noise term is the wrong colour/amplitude model for A_a's
long-horizon effect, even though it happens to reproduce the *marginal* displacement distribution
at 50 steps by coincidence of matching the dominant AR3 drift.

## Conclusion

A_a's contribution **cannot be statistically closed** by this simple coloured-noise SDE at the
250-step horizon that matters for the L1.5 macro equation, even though the drift + a naive
noise term visually track the raw displacement distribution at short (50-step) horizons. This
extends W13's conclusion (A_a is not closable as a low-dimensional deterministic force) to the
distributional/statistical route as well, at least for a memoryless-coloured SDE ansatz.
A longer-memory or heavier-tailed noise model, or a closed-loop-fit variance (i.e. fitting
sigma directly to match the accumulated 250-step spread rather than the one-step residual),
is the natural next attempt but is out of scope for this pass.

## Artifacts

- `PROTOCOL_W14.md`, `FREEZE_LOCK.txt` -- frozen before TEST tapes were opened.
- `w14_fit.py`, `w14_sim.py`, `w14_eval.py`, `w11_lib.py` -- fit/sim/eval code.
- `params.json` -- frozen fit parameters.
- `results_ptapes.json`, `results_ptapes2.json`, `RESULTS_W14.json` -- full per-tape results and verdicts.
