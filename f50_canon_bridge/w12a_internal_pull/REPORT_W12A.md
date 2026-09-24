# REPORT W12A — prospective test of the frozen A_a law (negative result)

**Line:** Claude Cloud exploratory analysis. Not a Research OS / Science Supervisor ruling, not a
formal F50 closure claim.

## Frozen law

`PROTOCOL_W12A.md` (sha256 `8c070a8d212fc504dbe21dce7092e0eb0d204eac1ca3202a22a2b9b158269aa6`,
committed at `fccd20c`) freezes **AR1_ENV**:

```
M_a+ = normalize(M_a + 5*dt*S_MF,a(M_a) + A_a)
A_a+ = rho*A_a + beta*Env_a(M)          rho = 0.9990596941276166, beta = 1.579285859519589e-05
```

`Env_a` is the tangent point-mass attraction pull toward the OTHER cores (engine softmax, bandwidth
`eng.bandwidth`, weight `log n_b`), halo dropped. Pass criterion: beat the universal AR3 baseline's
250-step rollout skill on **both** prospective tapes, evaluated once with no retuning.

## Prospective result

| tape | horizon | AR3 skill | AR1_ENV skill |
|---|---|---|---|
| tape_2_8080_32_zo | 5 steps | 0.963 | 0.943 |
| | 50 steps | 0.879 | 0.786 |
| | **250 steps** | **0.398** | **0.331** |
| tape_3_4242_31_zo | 5 steps | 0.964 | 0.942 |
| | 50 steps | 0.880 | 0.785 |
| | **250 steps** | **0.396** | **0.356** |

**FAIL.** AR1_ENV is below AR3 at every reported horizon on both prospective tapes, by the same
margin seen on the held-out discovery tapes during validation (see `PROTOCOL_W12A.md` section 3, and
`RESULTS_W12A.json` for the discovery-validation numbers alongside these). The result is consistent
across all 5 tapes tested (2 held-out discovery + 2 prospective): the environment coupling term
never overtakes plain AR1, and the balance/relaxation term (gamma) and full 3-parameter combination
are worse still (the unconstrained fit was even explosive, rho = 1.0056 > 1).

## Interpretation

This closes W12A as a **documented negative result**, not a discovered law:

- The universal AR3 baseline, which uses no environment information at all, remains the best
  closed-form description of the core's latent internal pull found so far.
- The environment mechanism this protocol tested (other cores as point masses, softmax-weighted,
  halo dropped) is measurably in the right direction -- AR1_ENV beats plain AR1 on 3 of 4 held-out
  tapes -- but the effect is far too small (beta ~1.6e-5) to close the gap to AR3, let alone exceed
  it.
- Two likely reasons this law under-performs AR3, both left open: (1) halo wells were dropped from
  `Env_a` for compute cost, and the halo ring is known (w11_decouple.py) to matter increasingly for
  larger cores; (2) the fit is isotropic and scalar (one `rho`, `beta` for every dimension and core
  size), whereas AR3's 3-lag history implicitly captures more of A_a's own autocorrelation structure
  than a 1-lag + small-environment-forcing model can.
- A genuinely closed law for A_a is still open. A natural next step (not attempted here, to respect
  the freeze and the compute budget) is an AR(3)-on-A model plus the environment term, and/or adding
  the halo ring back into `Env_a`.

## Files

- `PROTOCOL_W12A.md`, `FREEZE_LOCK.txt` -- the frozen law (commit `fccd20c`).
- `fit_train.json` -- pooled fit coefficients for all 7 candidate subsets.
- `val_13.json`, `val_22.json` -- discovery-tape validation runs (5 candidate laws each).
- `prospective_32.json`, `prospective_31.json` -- this report's raw prospective-eval outputs.
- `RESULTS_W12A.json` -- machine-readable summary of discovery validation + prospective result.
