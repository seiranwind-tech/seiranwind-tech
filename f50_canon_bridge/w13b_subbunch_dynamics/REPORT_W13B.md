# REPORT W13B — autonomous sub-bunch dynamics for the core's own internal pull (negative result)

**Line:** Claude Cloud exploratory analysis. Not a Research OS / Science Supervisor ruling, not a
formal F50 closure claim.

## Goal

Close the core-centre rollout gap (the latent internal pull `A_a`, still open after W11/W12A) with a
low-dimensional **autonomous** model: represent each core by `k` sub-bunches (centroid + count) that
evolve by their own macro law, with no further handoff after the block start, and beat the universal
AR3 baseline at 250-step core-centre rollout.

## Frozen law

`PROTOCOL_W13B.md` (sha256 `18e0a1c768cb81d6eb95548ba7f83e8e6f763ca3a8c2cf79a3504a8f5623c585`,
committed at `b9a12e3`) freezes **SB8_G05**:

```
m_j <- normalize( m_j + dt * [ salt_MF,j(m_j) + 0.5 * attract_gain * nr(pop_attract(m_j) - m_j) ] )
```

`k = 8` spherical-k-means sub-bunches per core (n>=10, gates>=60), split from true member positions
at each 10-gate block's start; `pop_attract` = softmax mean-field target over {sibling sub-bunches of
the same core} u {all other river groups as point masses}; integrated per engine `dt` micro-step
(the W12C halo template), not a coarse 5-step jump. Core prediction = count-weighted mean of the 8
sub-bunch centroids. Pass criterion: beat AR3's 250-step rollout skill on **both** prospective tapes,
evaluated once with no retuning.

## Prospective result

| tape | horizon | AR3 skill | SB8_G05 skill |
|---|---|---|---|
| tape_4_5151_41_zo | 5 steps | 0.959 | -1.419 |
| | 50 steps | 0.898 | -0.833 |
| | **250 steps** | **0.458** | **-0.514** |
| tape_3_6262_42_zo | 5 steps | 0.958 | -1.799 |
| | 50 steps | 0.902 | -1.082 |
| | **250 steps** | **0.424** | **-0.733** |

**FAIL.** SB8_G05 is far below AR3 at every horizon on both prospective tapes -- not a close miss but
roughly an order of magnitude worse in raw angular error, consistent with the discovery-tape
validation (see `RESULTS_W13B.json`): 2 of 3 discovery tapes were also strongly negative, and the one
tape with weakly positive skill (0.13-0.19) was still far under AR3 (0.52-0.92) on that same tape.

## What was tried before freezing

- `k in {2, 4, 8, 16}` sub-bunches, `g in {0.05, ..., 1.30}` (fraction of the engine's own
  `attract_gain=0.3`), both a coarse 5dt-per-frame jump (matching w11/w12a's core-level convention)
  and the finer per-engine-dt micro-step integration used by the successful W12C halo law. The
  per-dt-step form (matching the validated halo template) was adopted as the frozen family since it
  performed markedly better than the coarse 5dt jump, but "markedly better than terrible" was still
  never close to AR3.
- An AR1-memory variant on top of the sub-bunch attraction term (carrying a decaying accumulator
  instead of adding the physics term directly each step) was drafted but dropped before the freeze:
  the pure-physics sweep already showed no k or gain came within striking distance of AR3, so the
  extra free parameter was not worth the compute budget (see `discovery_runs/gain_k_sweep_tape11.md`).
- Increasing `k` from 2 to 16 gave **no measurable improvement** at any gain tested -- the sub-bunch
  resolution is not the bottleneck.

## Interpretation

- W11 (`w11_meanfield.py`) already found that collapsing a core's own members to a *single* point
  mass destroys the attraction term (R^2 -3.3 to -4.3 on the instantaneous force), while salt is
  mean-field-closed (R^2 0.99). Splitting into a handful of sub-bunches does not repair this: our
  sweep over k=2..16 shows the sub-bunch count is not the limiting factor.
- Salt and attraction nearly cancel in this system (`|S|~0.039, |A|~0.05, |D|~0.02`, W11). A law
  missing or mis-scaling the attraction term is not merely "somewhat worse" than the truth, it
  actively predicts in roughly the wrong direction/magnitude, which is exactly the large, systematic
  (not noisy) error observed here at every gain and k.
- The functional form that worked well for the halo (`w12c_law_test.py`: a fixed-magnitude,
  unit-normalized attraction pull applied to an *actual individual real member*) does not transfer to
  a *coarse aggregate* standing in for dozens of real members: real per-member attraction mostly
  self-cancels inside a group (which is why `A_a`'s empirical magnitude is small relative to
  `attract_gain`), and a rigid unit-length pull on the collapsed centroid cannot reproduce that
  cancellation regardless of the scalar gain chosen.
- **Net conclusion:** representing a core's own internal shape by a handful of autonomously-evolving
  point masses is not sufficient to close `A_a`. This is consistent with W11's independent finding
  (`w11_shape_pc.py`) that the internal shape needs k ~= 32 principal components to reconstruct the
  instantaneous force accurately -- a bar that 2-16 point masses, with mean-field attraction between
  them, does not come close to clearing dynamically.

## What stays open

- `A_a` (the core's latent internal pull) remains unclosed by any low-dimensional autonomous model
  tried across W12A (AR1 + environment) and W13B (autonomous sub-bunches). The universal empirical
  AR3 law on 5-step centre increments remains the best available closed-form description (250-step
  skill 0.42-0.55 across all tapes tested in this project).
- A natural next step, not attempted here to respect the freeze and compute budget: combine the
  universal AR3 memory (which implicitly captures much of A_a's own autocorrelation) with a small
  sub-bunch-derived correction term, rather than replacing AR3 outright with a from-scratch physical
  law.

## Files

- `PROTOCOL_W13B.md`, `FREEZE_LOCK.txt` -- the frozen law (commit `b9a12e3`).
- `w13b_subbunch.py` -- implementation (harness matches `w11_rollout.py` / `w12a_eval.py`).
- `discovery_runs/gain_k_sweep_tape11.md`, `discovery_runs/discovery_val.json` -- pre-freeze fitting
  and 3-tape discovery validation.
- `prospective_w13b.json`, `prospective_w13b.log` -- this report's raw prospective-eval output.
- `RESULTS_W13B.json` -- machine-readable summary of discovery validation + prospective result.
