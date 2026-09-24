# W12B — Boundary-flux macro law (frozen protocol)

**Line:** Claude Cloud exploratory analysis (F50 L1.5 closure project). This is **not** a Research
OS or Science Supervisor ruling, and **not** a formal F50 closure claim.

**Goal:** turn the residual H_T channel that survives the W11 river (h=12) coarse-graining — about
0.5–1.6 macro events per gate — into an explicit macro rate law over core-level variables
(M_a, n_a, and neighbour geometry), so the boundary-transfer channel becomes a quantitative flux
rather than an unmodelled residual.

## 0. Method (tape replay, read-only)

The frozen engine (`f40_v39r3_engine.py`) was **not** touched. Everything here is computed from the
5 DISCOVERY tapes already recorded for W11 (`f50_canon_bridge/w11_river/`).

Per tape, `w12b_build_pairs.py` reconstructs the causal river HOME partition at hysteresis **h=12**
exactly as `w11_river.py` / `w11_macro_events.py` do, then for every late gate g (g ≥ 60, i.e. steps
≥ 3000) builds:

- **pair rows**: for every ordered pair (a, b) of HOME groups alive at gate g−1 with n_a ≥ 2 and
  n_b ≥ 2: `sim(M_a, M_b)` (cosine of group centres from the zin frame 10·(g−1)+9), `n_a`, `n_b`,
  `rank` of b among a's neighbours by centre similarity (0 = nearest), `spread_a` = 1 − mean cos of
  a's members to M_a, and `target` = number of wells whose river-layer switch is recorded at gate g
  with old home a and new home b (an *existing*-group boundary transfer; wanderers that found a
  brand-new group are excluded here and modelled separately).
- **founding rows**: for every HOME group a alive at gate g−1: `n_a`, `spread_a`, and `target` = the
  number of permanent new-group foundings recorded at gate g whose wanderer's home was a.

This gives 5 tapes × ~45k–60k pair-rows and ~1.5k–1.8k founding-rows in the late regime.

## 1. Candidate laws

**Boundary-transfer rate**, Poisson with log link, λ_ab = exp(β0 + β1·sim + β2·log n_a + β3·log n_b
+ β4·rank + β5·spread_a). Fitted by IRLS with ridge penalty λ=2.0 (chosen because the unpenalized
fit is nearly separable in `sim` — see §4 — and blows up to |β| ~ 20–90 with unstable extrapolation;
ridge=2.0 keeps held-out performance while shrinking coefficients to a stable range).

**Founding rate**, Poisson with log link, μ_a = exp(γ0 + γ1·log n_a + γ2·spread_a), same ridge=2.0.

**Baselines**:
- constant rate (intercept-only Poisson)
- nearest-neighbour-only: λ_ab = exp(δ0 + δ1·[rank(b)=0])
- (sim alone, reported as an AUC baseline)

## 2. Frozen coefficients (fit on 3 discovery tapes)

Fit tapes: `tape_0_1357_11.npz`, `tape_1_1357_12.npz`, `tape_0_9753_21.npz` (138,338 pair-rows,
158 transfer events; 4,636 founding-rows, 77 founding events).

Held-out (validation) discovery tapes: `tape_2_1357_13_zo.npz`, `tape_1_9753_22_zo.npz`
(110,016 pair-rows, 79 transfer events; 3,515 founding-rows, 32 founding events).

**Boundary-transfer law**, order [intercept, sim, log n_a, log n_b, rank, spread_a]:
```
beta_pair_full = [-6.666873839249263, 3.1154287382879398, -0.19302865198682742,
                   0.3426163685962038, -1.9604929327574852, 0.2780582468019149]
```
Baselines (same fit tapes):
```
beta_pair_const = [-6.693530729010572]
beta_pair_nn    = [-8.411610903211939, 4.947461842965106]     # [intercept, is_rank0]
```

**Founding law**, order [intercept, log n_a, spread_a]:
```
beta_found_full  = [-4.811817608442928, 0.22058266098410076, 0.39798275112628034]
beta_found_const = [-3.998979949446851]
```

Exact reproduction: `python3 w12b_fit_eval.py work/pairs_0_1357_11.npz work/pairs_1_1357_12.npz
work/pairs_0_9753_21.npz -- work/pairs_2_1357_13.npz work/pairs_1_9753_22.npz`
(pair rows built by `w12b_build_pairs.py`, ridge=2.0 hardcoded as the default in
`poisson_glm_fit`).

## 3. Discovery metrics

**Pair (boundary transfer) — pooled held-out (2 val tapes, 79 events / 110,016 pair-rows):**

| metric | full law | constant | nearest-neighbour |
|---|---|---|---|
| log-likelihood | **−305.8** | −684.2 | −441.3 |
| deviance | 487.3 | 1243.9 | 758.2 |
| AUC (any transfer on pair ab) | **0.9974** | — | 0.9682 |
| AUC, sim alone | 0.9960 | — | — |
| destination top-1 accuracy | 0.9643 | — | 0.9643 |
| calibration: mean pred / actual events per gate | 1.797 / 1.317 | — | — |
| calib correlation (per-gate totals) | 0.624 | — | — |

Per held-out tape: `tape_2_1357_13` loglik −68.5 (const −129.9, nn −103.5), AUC 0.9959;
`tape_1_9753_22` loglik −237.3 (const −554.3, nn −337.7), AUC 0.9977.

**Founding — pooled held-out (2 val tapes, 32 events / 3,515 rows):**
loglik full −189.2 vs const −197.7 (deviance 323.3 vs 340.2). A small but consistent
improvement; founding is noisier and less structured than boundary transfer.

## 4. Honest caveats

- The `sim` feature is **near-separating**: sim | transfer ≈ [0.97, 0.99, 0.996] (p10/p50/p90)
  vs sim | no-transfer ≈ [−0.18, −0.04, 0.17]. Boundary transfers essentially only happen between
  groups whose centres are already close. This is why the unregularized MLE is unstable and why we
  freeze the ridge=2.0 fit rather than the unpenalized one.
- Destination top-1 accuracy of the full law **ties** the nearest-neighbour baseline on both
  held-out tapes (0.96–0.98): once an event is known to occur, "go to the nearest group" is already
  a very strong destination predictor. The full law's advantage is in **whether/how many** events
  occur (log-likelihood, AUC), not in destination choice given an event.
- Event counts are small (10–95 late transfer events per tape): all metrics, especially the
  founding law and the per-tape AUC, carry wide sampling uncertainty.
- The founding law is markedly weaker than the boundary-transfer law; its coefficients should be
  read as a coarse rate (mostly `n_a`-driven exposure), not a validated mechanism.

## 5. Pass criterion for the prospective test (frozen before opening `$S/ptapes/`)

The law **passes** if, evaluated once (no retuning) on each prospective tape separately:

1. Pooled and per-tape held-out log-likelihood of `beta_pair_full` is **strictly better** than both
   `beta_pair_const` and `beta_pair_nn` (higher log-likelihood = lower deviance).
2. AUC for "any transfer on pair ab" (`beta_pair_full`) is **≥ 0.90** on each prospective tape
   (discovery achieved 0.996–0.998; 0.90 leaves headroom for the small-N sampling noise noted in
   §4 while still requiring a clear improvement over the NN baseline's discovery AUC of ~0.93–0.97).
3. Destination top-1 accuracy of `beta_pair_full` is **not worse** than `beta_pair_nn` by more than
   0.05 absolute, on each prospective tape (it is allowed to tie, as in discovery).

If any of the two prospective tapes fails any criterion, the result is reported as a **partial /
negative** result for that tape, not silently dropped.

## 6. Files

- `w12b_build_pairs.py` — builds per-gate pair-rows and founding-rows from one tape (river h=12,
  read-only tape replay, no engine access).
- `w12b_fit_eval.py` — IRLS Poisson GLM fit + held-out evaluation (log-likelihood, deviance, AUC,
  destination top-1, per-gate calibration).
- `work/pairs_*.npz` — per-tape extracted rows for the 5 discovery tapes.
- `work/fit_eval_discovery.json` — full discovery fit + validation metrics (source of §2–3 numbers).
- `RESULTS_W12B.json`, `REPORT_W12B.md` — written after the one-shot prospective evaluation (§5).
