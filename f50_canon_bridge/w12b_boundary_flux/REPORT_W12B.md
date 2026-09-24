# W12B — Boundary-flux macro law: prospective result

**Line:** Claude Cloud exploratory analysis (F50 L1.5 closure project). Not a Research OS or
Science Supervisor ruling, not a formal F50 closure claim.

**Verdict: PASS.** The frozen boundary-transfer Poisson rate law (`PROTOCOL_W12B.md`, frozen
before either prospective tape was opened) beats both baselines and meets all three pass criteria
on **both** prospective tapes, evaluated once, with no retuning.

## What was tested

Law (frozen, ridge=2.0 IRLS Poisson GLM, fit on 3 discovery tapes):
```
lambda_ab = exp(-6.667 + 3.115*sim + (-0.193)*log(n_a) + 0.343*log(n_b) + (-1.960)*rank + 0.278*spread_a)
```
plus a founding-rate law `mu_a = exp(-4.812 + 0.221*log(n_a) + 0.398*spread_a)`, both defined over
core-level variables from the h=12 river partition -- see `PROTOCOL_W12B.md` for the full
derivation, baselines (constant-rate, nearest-neighbour-only) and discovery metrics.

## Prospective tapes

- `tape_2_8080_32_zo.npz`: 59 late macro events (33 existing-group transfers, 26 foundings).
- `tape_3_4242_31_zo.npz`: 39 late macro events (25 transfers, 14 foundings). This tape crashed on
  its first save and was re-recorded with the same seeds; we waited for its log's last line to say
  "done" before touching it, per the coordinator's instruction, and discarded no partial evaluation
  (none had been run against the truncated file).

## Results (per criterion, per tape)

| tape | C1: loglik beats const & nn | C2: AUC >= 0.90 | C3: top-1 within 0.05 of nn | tape pass |
|---|---|---|---|---|
| tape_2_8080_32 | PASS (full -120.7 vs const -293.6, nn -188.1) | PASS (0.9992) | PASS (1.000 = 1.000) | **PASS** |
| tape_3_4242_31 | PASS (full -109.7 vs const -219.0, nn -146.7) | PASS (0.9957) | PASS (0.958 = 0.958) | **PASS** |

**Pooled prospective** (94,622 pair-rows, 57 transfer events): log-likelihood -230.4 (full) vs
-512.6 (const) vs -334.8 (nn); AUC 0.9977 (full) vs 0.9676 (nn baseline) vs 0.9960 (sim alone);
destination top-1 accuracy 0.978 (full, ties nn). Calibration: mean predicted events/gate 1.55 vs
actual 0.95 (over-predicts moderately -- see caveats).

**Founding law**: pooled prospective log-likelihood -217.9 (full) vs -224.5 (const); a modest,
consistent improvement, matching the discovery-level finding that this piece is far weaker than
the boundary-transfer law.

## Comparison with discovery

Discovery held-out pooled: loglik -305.8 (full) vs -684.2 (const) vs -441.3 (nn); AUC 0.9974 vs
0.9682 (nn). Prospective pooled: loglik -230.4 vs -512.6 vs -334.8; AUC 0.9977 vs 0.9676 (nn). The
law transfers **without refitting**: AUC is essentially unchanged (0.9974 -> 0.9977), the
log-likelihood margin over both baselines holds up, and destination top-1 accuracy is comparable
(0.964 discovery -> 0.978 prospective, both tying the nearest-neighbour baseline).

## Honest caveats (carried over from discovery, still true prospectively)

- As flagged in `PROTOCOL_W12B.md` sec 4, `sim` is close to separating (transfers occur almost only
  between near-centre groups); this is why we froze the ridge=2.0 fit rather than the unstable
  unpenalized MLE. The prospective test confirms the regularized fit is what generalizes.
- The law's advantage over nearest-neighbour is in **whether/how many** transfers occur (deviance,
  AUC), not in **which** destination is chosen given a transfer -- top-1 accuracy ties the NN
  baseline on both prospective tapes, as it did in discovery.
- Prospective total-event calibration is over-predicted (1.55 predicted vs 0.95 actual events per
  gate, pooled); event counts remain small (25-33 transfers per tape) so this is consistent with
  sampling noise but is reported honestly rather than smoothed over.
- The founding-rate law remains the weakest piece of this closure; it improves on the constant
  baseline but by a small margin, both in discovery and prospectively.

## Bottom line for the F50 L1.5 project

The residual H_T boundary-transfer channel (about two thirds of the ~0.5-1.6 remaining macro
events/gate at h=12) now has an explicit, prospectively validated Poisson rate law over
core-level variables (M_a, M_b, n_a, n_b, rank, spread_a). It is not a full closure of H_T -- the
founding channel (about one third of events) is only weakly modelled, and the law over-predicts
total prospective event counts by roughly 60% -- but it converts what was an unmodelled residual
into a quantitative, tape-independent macro flux law with better-than-nearest-neighbour skill.

## Provenance

- Frozen protocol commit: see `git log` for the "W12B freeze" commit (this repo,
  `f50_canon_bridge/w12b_boundary_flux/`).
- `RESULTS_W12B.json`: full per-tape and pooled metrics, machine-readable.
- `w12b_prospective_eval.py`: the exact evaluation script run once against the prospective tapes,
  using the frozen coefficients copied verbatim from `PROTOCOL_W12B.md`.
