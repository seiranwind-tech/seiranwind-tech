# W3 — ChatGPT recurrent HOLD/SWITCH controller (parallel-universe analog)

Bridge line: PU-BRIDGE-R1. Trained/frozen against `w1/pu_engine.py`
(sha256 `4ace90d0…`) and `w1/DEV_table.npz` / `w1/DEV_events.json`
("moves" list only — realised per-basin centroid movement W at each
redraw). No column outside the LEGAL set (`n, age, R, skew, yo, yb, cob,
dR, dskew, dmu`) and no births/`G_birth`/`G_shape`/`so`/`sb`/`rho` were
ever read; `w2/`, `w4/`, `sealed/` were never opened.

## 1. What "next-interval W" means here

Redraws happen every 50 steps (`REDRAW=50`); `redraw_index = step // 50`.
`w1/DEV_events.json["moves"]` gives, for each `(seed, basin)` that
survives a redraw, the realised centroid movement `W` at that redraw
step. A table row `(seed, step, basin)` with `redraw_index = r` belongs
to the interval whose outcome is `W_r` for that `(seed, basin)`. The
controller predicts `W_r` **during** interval `r`, using only rows of
interval `r` seen so far and `W` values of **already-closed** intervals
of the same basin lineage — never `W_r` itself.

## 2. Candidate predictors (all causal, per `(seed,basin)` lineage)

- **A — EMA of past W.** At each interval boundary crossed, once the
  realised `W` of the interval that just closed is known:
  `ema_W ← η_A·ema_W + (1−η_A)·W_prev` (η_A = 0.6). `A_pred = ema_W`
  (constant through an interval; falls back to `W_prior` = train-mean
  realised W, 0.1182, before any interval has closed).
- **D — rotation/direction based.** `cum_dskew` resets to 0 at the start
  of each interval and accumulates `|dskew|` of every row seen so far in
  the current interval. `D_pred = k_D · cum_dskew`, with `k_D` fit by
  closed-form least squares (`k_D = Σ D_raw·W / Σ D_raw²`) on DEV seeds
  1–4 labeled rows. Fitted `k_D = 4.749`.
- **M — magnitude/momentum based.** Continuous slow EMA of the per-step
  aggregate drift: `ema_M ← η_M·ema_M + (1−η_M)·(dmu·50)` (η_M = 0.9),
  never reset at interval boundaries. `M_pred = ema_M`.

## 3. Layer 1 — recurrent softmax ranker

Features `X1 = [1, n, age, R, skew, yo, yb, cob, dR, dskew, dmu, τ]`
(τ = clip(age/500, 0, 1)), standardized with train-set mean/std.
`p1 = softmax(Θ1·X1)` (3-way, one logit per candidate). Recurrent memory:
`h1 ← p1` at lineage start, else `h1 ← η_h1·h1 + (1−η_h1)·p1` (η_h1=0.7).
`champion` is initialized to `argmax(h1)` at lineage start and persists
until a switch; `challenger = argmax(h1)` every row.
`Θ1` (3×12) trained by full-batch multinomial-logistic gradient descent
(400 iters, lr 0.3, L2 1e-3) on DEV seeds 1–4, label
`label1 = argmin_i |candidate_i − W_r|` (only rows with a resolved `W_r`).
Train top-1 accuracy vs. this argmin label: **0.663** (chance ≈ 0.33 for
3 balanced classes; classes are not balanced here since A dominates).

## 4. Layer 2 — recurrent HOLD/SWITCH learner

`X2 = [1, p1_A, p1_D, p1_M, p_top−p_champ, norm_pred_gap, τ, dτ,
log1p(champion_age), h2]`, `norm_pred_gap = |pred_chall − pred_champ| /
(|pred_champ|+ε)`, recurrent memory `h2 ← η_h2·h2 + (1−η_h2)·(p_top−p_champ)`
(η_h2 = 0.7, reset to the current gap at lineage start).
`[P_HOLD, P_SWITCH] = softmax(Θ2·X2)`; rule:
`do_switch = (challenger ≠ champion) AND P_SWITCH > P_HOLD`.

**Label construction (train only).** Because the champion trajectory that
X2 depends on is itself a function of past switch decisions, Θ2 is fit in
two passes, both strictly causal and using only seeds 1–4:
1. *Pass A (provisional)* runs the sequential layer-1/2 recurrence on
   seeds 1–4 with the trivial reference rule
   `do_switch = challenger_valid AND (p_top > p_champ)` (equivalent to
   "always follow the current top-ranked candidate") to produce a
   champion/challenger trace and the `X2` feature stream, with **no**
   dependence on Θ2.
2. `label2 = 1` iff `challenger_valid` and the challenger's error beats
   the champion's error by `margin=0` at the row's realised `W_r`
   (`|pred_chall−W_r| < |pred_champ−W_r|`); only rows with
   `challenger_valid=True` and a resolved `W_r` are used
   (**1,684 rows** out of 1,113,700 train rows — challenger ≠ champion
   is rare under the reference rule since a stable smoothed ranking
   rarely flips). Positive-label fraction 0.562.
3. `Θ2` (2×10) fit the same way as `Θ1` on those 1,684 rows. Train
   accuracy **0.638** (weak edge over the 56.2% base rate — see
   Limitations).
4. The **frozen, deployed controller** (`mode="deploy"` in
   `controller_run.run_pass`) always uses the *learned* `Θ2` rule above,
   for both the DEV trace and any future replay — Pass A only exists to
   manufacture labels for training and is not part of the frozen
   artifact.

`Γ_G = P_SWITCH − 0.5` is reported as a convenience "switch margin"
signal.

## 5. Freeze artifacts

- `w3/CONTROLLER_FREEZE.json` — `Θ1, Θ2`, standardization stats,
  `η_A, η_M, η_h1, η_h2`, `k_D`, `W_prior`, `margin`, feature name lists,
  train/eval seed split, and sha256 of the engine + DEV table/events used
  to fit it.
- `w3/controller_run.py` — `trace(table_npz_path, events_json_path,
  freeze_path=None) -> dict[str, np.ndarray]`, self-contained (numpy
  only), replays the frozen controller over any table with the same
  schema. It reads **only** `LEGAL_COLS` + `age`/`redraw_index`/`step`/
  `phase`/`seed`/`basin` from the table and **only** the `"moves"` key of
  the events file (the `"events"`/births list is never indexed). Runtime
  causality: candidate `A_pred` only updates from a past interval's `W`
  once that interval has closed; nothing reads the current interval's
  realised `W`.
- `w3/DEV_controller_trace.npz` — frozen-controller replay over the full
  DEV table (seeds 1–6), row-aligned to `w1/DEV_table.npz`.
- `w3/train_controller.py` — the fitting driver (kept for
  reproducibility; not required at replay time).

**SHA256:**
```
CONTROLLER_FREEZE.json      ce3612f7d8f17dc2047e83442d6c91047c03acdbffb86760e4c410edabf6a94f
controller_run.py           46e577b5f03b6d5520a9d55d2218d90152cd05cd8f6f5725c04f3d392c690684
DEV_controller_trace.npz    46355d3089b18cb2454481c46ddf88c0ec620f7c01f11624bb50a9090d2e3a45
(inputs) w1/pu_engine.py    4ace90d04b44a9efa873242a4e3751d73dff02a09edb3a501de1b11603f6fc10
(inputs) w1/DEV_table.npz   082669e153917becad7d507910c79c7fd0a9c6da3d6ddcfb356c05e21f9ae414
(inputs) w1/DEV_events.json f6a62d15492f5497e21cf1f53a3efc0b2df1ffa2c7a31190dcf223ec4e709eaf
```

## 6. DEV-validation metrics (eval seeds 5–6, 548,900 labeled rows)

RMSE against realised `W`:

| predictor | RMSE |
|---|---|
| **A** — EMA of past W | **0.0456** |
| D — rotation/direction | 0.0640 |
| M — momentum | 0.6309 |
| **Controller (chosen_pred)** | 0.0513 |

Champion occupancy (eval): A 74.4%, D 25.6%, M 0.0% (layer 1 essentially
never ranks M first — its RMSE is an order of magnitude worse, since
`dmu` is a per-step aggregate delta and its 50×-scaled EMA is a poor
proxy for a redraw-interval centroid jump).

Switch statistics (eval): 671 champion switches over 549,000 rows
(≈0.12% of rows trigger a switch); `challenger_valid` fraction 0.237
(train: 0.216). `P_SWITCH` distribution (eval): mean 0.371, std 0.194,
range [0.0006, 1.000).

Train-set (seeds 1–4) figures for comparison: 1,294 switches over
1,114,150 rows, `challenger_valid` fraction 0.216, champion occupancy A
76.6% / D 23.4% / M 0.0%.

## 7. Honest limitations

- **The controller does not beat the best single candidate on this DEV
  split.** RMSE(chosen) = 0.0513 > RMSE(A alone) = 0.0456. Layer 1's
  46-point (0.663 vs. ~0.5 majority-class) edge over guessing is real but
  modest, and layer 2 is fit on only 1,684 "challenger valid" rows — an
  order of magnitude less data than layer 1 — so its switch decisions
  add nearly as much noise as signal on this split. A more faithful
  bridge would likely need either a better challenger candidate (D's own
  RMSE, 0.064, is worse than A's) or a stricter switch margin/threshold
  tuned for precision rather than the unregularized 0-margin label used
  here.
- **Layer-2 label construction is a bootstrapped two-pass scheme**, not
  a closed-form target: it depends on a provisional "always follow
  layer-1's top pick" reference policy to generate a champion trace for
  labeling, since the true labels are only well-defined once a champion
  trajectory exists. This is a legitimate way to break the circularity
  (Θ2 needs champion history; champion history needs a switch rule) but
  it means Θ2 is trained on the *reference* policy's champion path, not
  its own — a mild train/deploy mismatch, mitigated by the small
  variance in when `challenger_valid` actually holds.
- **Candidate D's coefficient `k_D` is fit at the row level, pooling all
  phases (0–49) of the interval**, so its accuracy is necessarily worse
  early in an interval (little `dskew` has accumulated yet) than late.
  The trace does not separately calibrate by phase.
- **`τ` (slow time) is a fixed function of `age` alone** (`clip(age/500,
  0,1)`), not tied to any engine-internal clock beyond what LEGAL columns
  expose; it is a design choice standing in for the "slow time τ"
  described in the real system, not a literal transplant of it.
- **Recurrent state (`h1, h2`, `champion`, `ema_W`, `ema_M`, `cum_dskew`)
  is scoped per `(seed, basin)` lineage** and resets at basin birth. This
  matches "one controller instance per tracked basin" but means a basin
  that is reborn under a new id starts the controller from scratch —
  faithful to what LEGAL data alone can support (no cross-basin identity
  is observable without birth information, which is off-limits).
- No hyperparameter search was run over `η_A, η_M, η_h1, η_h2` (fixed by
  design judgment, not tuned) — per the "keep compute modest" directive.
  A grid search over these might materially change the RMSE gap noted
  above.
