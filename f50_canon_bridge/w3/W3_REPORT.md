# W3 — ChatGPT recurrent HOLD/SWITCH controller (canonical-line retrain, CANON-BRIDGE-R1)

Same architecture, feature lists and code as the parallel-universe bridge
(`f50_pu_bridge/w3/`); **only weights and standardization statistics were
re-estimated** on the canonical DEV data. Trained against
`w1/canon_recorder.py` (read-only tape recorder around the frozen F40
V3.9R3 engine; the engine file itself, `f40_v39r3_engine.py`, lives
outside this bridge directory at `legacy/V39R3_FROZEN/`, cited sha256
`4a71aeb4a66a8a071a6b6f9a9b065687d237909aea5e89a537ee6d80d2aa3b1d` per
`w1/OBSERVABLE_CONSTITUTION.md` and the `ENG_SHA` assertion inside
`canon_recorder.py`) and `w1/DEV_table.npz` / `w1/DEV_events.json`
("moves" list only). No column outside the LEGAL set
(`n, age, R, skew, yo, yb, cob, dR, dskew, dmu`) and no
`G_birth`/`G_shape`/`so`/`sb`/`rho`/births/`lineage`/`threshold` were
read; `w2/`, `w4/`, `sealed/` (in this bridge or the old
`f50_pu_bridge/`) were never opened.

## 1. Canonical-line conventions (differences from the PU bridge)

- Table rows start at step 800 (`Row = (block as seed, step, basin track
  id) for n≥2, from step 800 on`).
- `redraw_index = (step−1)//50`, `phase = (step−1)%50` — the
  reassignment/redraw step is `phase==49` (vs. the PU line's
  `redraw_index = step//50`, redraw at `step≡49 (mod 50)` under its own
  0-indexed step).
- `moves[i]["step"]` is the *reassignment-completed* step (a multiple of
  50), which is exactly the table's `step` value at that basin's
  `phase==49` row for that interval. Mapping realised `W` to
  `(seed,basin,redraw_index)` therefore uses
  `redraw_index = (moves_step − 1) // 50` (offset 1), not the PU line's
  offset-0 formula. This was verified empirically: a random sample of
  2000 table rows resolved to a recorded move in 1997/2000 (99.85%)
  cases using this formula, and 1997/2000 of the `phase==49` rows
  specifically. `controller_run.build_moves_map(moves, step_offset)` now
  takes this as an explicit parameter (`step_offset=0` for the PU
  convention, `step_offset=1` for canonical), stored in
  `CONTROLLER_FREEZE.json["step_offset"]=1` so `trace()` uses the right
  convention automatically.
- `REDRAW=50` (`basin_reassign_every=50`) is unchanged.
- Train/eval split: **train seeds (blocks) 0,1,2 — evaluate on seed
  (block) 3** (vs. the PU line's 1-4 / 5-6).

Everything else — the 3 candidate predictors (A: EMA of past realised W
across closed intervals; D: `k_D`·running `|dskew|` sum since interval
start; M: slow EMA of `dmu*50`), Layer-1 softmax ranker with recurrent
`h1` over `X1=[1,n,age,R,skew,yo,yb,cob,dR,dskew,dmu,τ]`, Layer-2
HOLD/SWITCH softmax with recurrent `h2` over
`X2=[1,p1(3),p_top−p_champ,norm_pred_gap,τ,dτ,log1p(champion_age),h2]`,
`do_switch = challenger_valid AND P_SWITCH>P_HOLD`, and the two-pass
bootstrap label construction for Θ2 — is architecturally identical to
`f50_pu_bridge/w3/W3_REPORT.md` §2–4; see that report for the equations.
Only the fitted numbers below differ.

## 2. Freeze artifacts

- `w3/CONTROLLER_FREEZE.json` — Θ1 (3×12), Θ2 (2×10), standardization
  stats, `η_A=0.6, η_M=0.9, η_h1=η_h2=0.7`, `k_D=6.0007`,
  `W_prior=0.07262` (train-mean realised W), `margin=0`,
  `step_offset=1`, feature names, train/eval seed split, sha256 of
  `canon_recorder.py` + DEV table/events, and the cited (not locally
  hashed — the engine file is outside this bridge dir) F40 V3.9R3 engine
  sha256.
- `w3/controller_run.py` — same `trace(table_npz_path, events_json_path,
  freeze_path=None) -> dict[str,np.ndarray]` API, row-aligned, strictly
  causal (candidate `A_pred` only updates from a closed past interval's
  `W`; the current interval's realised `W` is never read). Verified to
  replay standalone from just the freeze file (18.7s over the full
  677,340-row DEV table).
- `w3/DEV_controller_trace.npz` — frozen-controller replay over the full
  canonical DEV table (seeds 0–3).
- `w3/train_controller.py` — fitting driver (kept for reproducibility).

**SHA256:**
```
CONTROLLER_FREEZE.json      4c1038ec507a1fa51e68de1fc9d211839cd0508ff1f29e00b48bf7074f88ae3d
controller_run.py           a19cccdfe2a3ab1a46cc96d8c247a0a63fcd512cf980e3775dfc14b9c578d23e
DEV_controller_trace.npz    68db84ab1f708860de99ce4c3ddce8e4508afdcf73a4ca61f278ac6dc2612a4d
(input) w1/canon_recorder.py  facc817b531f8ddb575d1fbe51744f39cd082016c8d2b705db74bfc06adfa8bb
(input) w1/DEV_table.npz      2f7cd7067573423ef0838ca160ec544f40b28adf8ef3ce45338f3f6e809b0e4c
(input) w1/DEV_events.json    1d90d443a5716864d18c6aef35f5d5a97c6f0302a0732d9111e31ce84d6e4479
(cited, not locally hashed) f40_v39r3_engine.py  4a71aeb4a66a8a071a6b6f9a9b065687d237909aea5e89a537ee6d80d2aa3b1d
```

## 3. DEV-validation metrics (eval seed 3, 166,974 labeled rows)

RMSE against realised `W`:

| predictor | RMSE |
|---|---|
| **A** — EMA of past W | **0.04586** |
| D — rotation/direction | 0.06381 |
| M — momentum | 0.08387 |
| Controller (chosen_pred) | 0.05772 |

Champion occupancy (eval seed 3): A 20.6%, D 11.2%, **M 68.2%** — a
striking reversal from the PU line, where Layer-1 almost never picked M
(0.0% occupancy there). On the canonical engine, `dmu` (the per-step mean
drift) is evidently a much more informative signal relative to `A`/`D`
than in the PU simulator, so Layer 1 ranks M highly despite M having the
*worst* standalone RMSE (0.084, nearly double A's). This is the main
red flag in this retrain (see Limitations).

Switch statistics (eval seed 3): 79 champion switches over 167,274 rows
(0.047%); `challenger_valid` fraction 0.318 (train: 0.287). `P_SWITCH`
distribution (eval): mean 0.073, std 0.080, range
[9.9e-119, 0.99999998) — much more concentrated near 0 (HOLD-biased)
than the PU line's controller (mean 0.371 there).

Train-set (seeds 0–2) figures: 203 switches over 510,066 rows,
`challenger_valid` fraction 0.287, champion occupancy A 18.0% / D 10.8%
/ M 71.3%.

Layer-1 train top-1 accuracy (vs. argmin-error label): 0.663 (same as
the PU line, coincidentally). Layer-2 train accuracy: 0.755 on 1,214
challenger-valid labeled rows (positive rate 0.462) — a real improvement
over the PU line's 0.638/1,684-row fit, though still on a small sample.

## 4. Honest limitations

- **The controller does not beat candidate A alone on this DEV split**
  (RMSE 0.0577 vs. 0.0459), same qualitative finding as the PU-line
  report. Here the gap is driven by Layer 1 over-favoring M (worst
  single-predictor RMSE) in 68–71% of rows, which the RMSE-weighted
  argmin-error training label does not obviously predict — Layer 1's
  training accuracy (0.663) measures whether it recovers the *locally
  best* candidate per row, not whether the induced champion policy
  minimizes RMSE end-to-end; those are not the same objective, and this
  retrain surfaces that gap more starkly than the PU line did (there,
  Layer 1 happened to favor A, the best global predictor, most of the
  time; here it favors M, the worst).
- **Candidate M's coefficient is unfitted** (it is a fixed-η EMA, not
  least-squares calibrated like `k_D` for D) — its raw scale (`dmu*50`)
  may simply not be well matched to canonical-engine `W`'s scale/regime,
  which is consistent with M being both frequently chosen (by Layer 1's
  aggregate-feature-based ranking) and RMSE-worst (in an absolute-error
  sense). A calibrated `k_M` (parallel to `k_D`) was **not** added here
  to keep this a like-for-like architecture retrain, but is a natural
  next step flagged for the coordinator.
- **Layer-2 label construction remains a two-pass bootstrap** (provisional
  "always follow Layer 1's top pick" reference policy → champion trace →
  labels → fit Θ2), inheriting the same train/deploy mismatch caveat
  documented in the PU-line report.
- **`k_D` (6.001) and `W_prior` (0.0726) are refit from scratch** on
  canonical seeds 0–2, not reused from the PU line's values (4.749,
  0.1182) — the two engines produce realised `W` on different absolute
  scales (canonical mean W is roughly 40% smaller), so no cross-line
  weight transplant was attempted.
- **`τ` (slow time) is still `clip(age/500,0,1)`**, an unchanged design
  choice, not tied to any canonical-engine-internal clock.
- **No hyperparameter search** over `η_A, η_M, η_h1, η_h2` was run for
  this retrain either (same fixed values as the PU line, 0.6/0.9/0.7/0.7)
  — per "architecture and feature lists unchanged, only weights ...
  re-estimated." Given how much the champion-occupancy balance shifted
  between lines, these etas are prime candidates for retuning if this
  bridge is taken further.
- **Coverage of the moves→row match (99.85%) is not 100%.** The
  remaining ~0.15% are almost certainly tracks whose interval closes
  after that track's last recorded row (track dies/merges before its
  reassignment), consistent with the PU line's own near-100%-but-not-100%
  coverage; those rows simply have no `label_W` and are excluded from
  training/eval, never imputed.
