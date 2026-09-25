# PROTOCOL W13B — autonomous sub-bunch ("次生葡萄串") law for the core's own internal dynamics

**Line:** Claude Cloud exploratory analysis. This is **not** a Research OS or Science Supervisor
ruling, and **not** an F50 closure claim.

## Goal

Close the core-centre rollout gap left open by W11/W12A (the core's latent internal pull `A_a`) with
a low-dimensional, **autonomous** model: represent each core by `k` sub-bunches (centroid + count)
and evolve them by their own macro law, with no further ground truth after the block start. Pass
criterion: **beat the universal AR3 baseline's 250-step core-centre rollout skill on both prospective
tapes**, frozen before those tapes are opened.

## Frozen law

At the start of each 10-gate block, for every core `a` with `n_a >= 10` members (late regime, gates
>= 60), split its members by spherical k-means (**k = 8**) on their true positions at the block's
first evaluated frame, giving sub-bunch centroids `m_j` and counts `n_j` (`j = 1..8`).

Each sub-bunch then evolves **autonomously**, integrated per ENGINE micro-step (`dt`, not a coarse
5-step jump — this is the W12C halo template):

```
m_j <- normalize( m_j + dt * [ salt_MF,j(m_j) + g * attract_gain * nr( pop_attract(m_j) - m_j ) ] )
```

- `salt_MF,j(m_j)`: mean-field salt — every REAL member of sub-bunch `j` is evaluated with ITS OWN
  teeth/mask/skew bundle at the shared position `m_j`, then averaged (same convention as
  `w11_rollout.py` / `w12a_eval.py`, applied at sub-bunch instead of core granularity).
- `pop_attract(m_j)`: softmax (engine bandwidth `eng.bandwidth`) weighted mean of the point-mass
  population = {the core's OTHER 7 sub-bunches} u {all OTHER river groups' point masses `(M_b, n_b)`,
  core or not}, self excluded, weights `log(count)`. Halo is not resolved individually (dropped for
  compute cost, as in W12A).
- `g = 0.5` (fitted scalar multiplying the engine's own `attract_gain = 0.3`; see
  `discovery_runs/gain_k_sweep_tape11.md` for the sweep over `k in {2,4,8,16}` and
  `g in {0.05 .. 1.3}` that selected it — the minimum is shallow and no grid point tested closes the
  gap to AR3).
- Non-tested groups (everything else) are held FIXED at their block-start point-mass position for the
  whole rollout (a compute simplification; only the tested cores' own sub-bunch dynamics is scored).

**Core-centre prediction** = `normalize( sum_j n_j * m_j )`, the count-weighted mean of the 8
sub-bunch centroids.

Implementation: `w13b_subbunch.py`. Baseline: universal AR3, coefficients `[1.1639, 0.5491,
-0.7167]` on 5-step centre increments (`w11_rollout.py`), identical harness (river-home groups,
gates >= 60, `n_a >= 10` cores, horizons 5/50/250 steps, `skill = 1 - err/disp`).

## Discovery result (fit + validation, 3 tapes)

| tape | horizon | AR3 skill | SB8_G05 skill |
|---|---|---|---|
| tape_0_1357_11 | 5 | 0.930 | -2.34 |
| | 50 | 0.863 | -1.74 |
| | 250 | 0.409 | -1.25 |
| tape_0_9753_21 | 5 | 0.977 | -0.20 |
| | 50 | 0.921 | 0.126 |
| | 250 | 0.517 | 0.185 |
| tape_1_1357_12 | 5 | 0.947 | -2.93 |
| | 50 | 0.899 | -1.80 |
| | 250 | 0.550 | -1.39 |

`SB8_G05` never beats AR3 on any tape at any horizon. It is occasionally weakly positive (250 steps,
tape_0_9753_21) but usually strongly negative. The gap is not close to a fair fight: it is roughly
an order of magnitude in error at short horizons.

## Interpretation (why this fails, going in with eyes open)

- W11 (`w11_meanfield.py`) already established that **attraction is not point-mass mean-field
  closable at the core level**: collapsing a core's own members to its centroid gives R² -3.3 to
  -4.3 on the instantaneous force. Splitting into a handful of sub-bunches (`k` up to 16 tested) does
  not fix this: the sweep shows no benefit from `k=2` to `k=16` at any gain.
- Salt alone (mean-field, `g=0`) is *not* a reasonable law either: salt and attraction nearly cancel
  (`|S|~0.039, |A|~0.05, |D|~0.02` per W11), so a law missing the attraction term altogether is
  already strongly negative-skill, consistent with what is measured here at `g=0`.
- The unit-normalized, fixed-magnitude attraction (`attract_gain * nr(target - m_j)`, the same
  functional form that worked for individual halo wells in W12C) is well suited to a *single real
  point* (an actual member with an actual local neighbourhood), but is the wrong functional form for
  a *coarse aggregate* standing in for dozens of real members: the true per-member attraction largely
  self-cancels within a group (that is exactly why `S_MF` alone under-predicts and `A_a`'s empirical
  magnitude is small), and a rigid unit-length pull applied to the collapsed centroid does not
  reproduce that cancellation at any of the gains or `k` tried.

## Freeze / pass criterion

Frozen law: `SB8_G05` above (`k=8`, `g=0.5`), exactly as implemented in `w13b_subbunch.py`, no
further retuning. Evaluated ONCE, no refit, on the two prospective tapes:
`ptapes2/tape_4_5151_41_zo.npz` and `ptapes2/tape_3_6262_42_zo.npz`.

**Pass criterion:** the frozen law's 250-step core-centre rollout skill beats the universal AR3
baseline's 250-step skill on **both** prospective tapes.

Given the discovery result above, the expected prospective outcome is **FAIL** — this freeze is
made anyway, per protocol, so the negative result is documented honestly and not retroactively
tuned to the prospective tapes.
