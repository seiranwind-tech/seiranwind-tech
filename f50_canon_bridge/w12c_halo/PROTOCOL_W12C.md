# W12C — Halo co-motion and a frozen halo law

**Line:** Claude Cloud exploratory analysis (F50 L1.5 closure project). **Not** a Research OS or
Science Supervisor ruling, and **not** an F50 closure claim.

**Question:** does the halo (singleton / tiny-group wells, ~1–2% of wells, river home-partition
h=12, n<5) co-move with its nearest core, well enough that a low-dimensional law can predict its
position from core-level state alone, so individual halo positions need not be handed off?

## 0. Data

- DISCOVERY tapes (fit): `tape_0_1357_11.npz`, `tape_1_1357_12.npz`, `tape_0_9753_21.npz`.
- DISCOVERY tapes (validate): `tape_2_1357_13_zo.npz`, `tape_1_9753_22_zo.npz`.
- PROSPECTIVE tapes (held out, evaluated once after this freeze, no retuning):
  `tape_3_4242_31_zo.npz`, `tape_2_8080_32_zo.npz`.
- Analysis window: gates ≥ 60 (late regime). Core = river home-group (h=12) with n ≥ 5. Halo =
  every other well. "Nearest core" of a halo well = argmax cosine to core centre M_c.

## 1. Co-motion measurement (`w12c_comotion.py`)

Measured on `tape_0_1357_11.npz`, 1-gate (50-step) spacing:

| quantity | median | p10 | p90 |
|---|---|---|---|
| angle(halo well, nearest core M_c), deg | 30.7 | 3.1 | 44.0 |
| per-well std of that angle over time, deg | 1.7 | 0.5 | 3.9 |
| offset-direction cos after parallel transport of M_c's own motion (1 gate) | 0.9994 | 0.9865 | 0.99993 |
| velocity cos(halo, nearest core) (1 gate) | 0.955 | 0.808 | 0.992 |
| dwell time on same nearest core, gates | 3 | 1 | 10 |

**Verdict:** the halo co-moves with its nearest core. The ring radius (angle to M_c) is
well-defined and stable per well (std ~1–4°, much smaller than the ring radius itself). The
offset direction is close to rigidly parallel-transported over a 50-step step (median cos
0.9994). Velocity is correlated but not identical (median cos 0.955, magnitude ratio ~0.79),
consistent with the halo well having its own, faster internal dynamics superposed on rigid
co-motion. Dwell time on the same nearest core is short relative to a tape (median 3 gates =
150 steps), i.e. the "which core" assignment itself is not static over long horizons — this is
why the horizon-250-step test matters.

## 2. Candidate laws tested (`w12c_law_test.py`)

Predict halo well position at t+Δ (Δ = 5, 50, 250 steps) from core-level state only (core
centres M_a(t) and sizes n_a(t), known continuously — this is the state that is *not* being
proposed for elimination):

- **baseline (frozen)**: halo well does not move (sanity floor).
- **(i) rigid co-motion**: offset to source-time nearest core, parallel-transported to the
  core's position at t+Δ, angle to core held fixed.
- **(iii) own single-well dynamics**: the halo well's own force law,
  `salt_w + 0.3·nr(attraction to point-mass cores + other halo)`, jointly integrated forward Δ
  steps for the whole halo subset (~20–60 wells per tape), using the *true* core trajectory
  (point masses `(M_b, n_b)`) read at each 5-step frame as the (assumed-known) core-level state.
  This is a self-contained, low-dimensional simulation — no individual non-halo well data is
  used.
- Law (ii) (offset relaxing to a stationary ring radius) was not separately fit: co-motion
  measurement (§1) shows the ring angle is already close to constant over the tested horizons,
  so it collapses to law (i) and was not pursued further; this is a candidate for future work at
  longer horizons.

**Decisive test:** plug the law's predicted halo positions into an m=1-style core force
(own core members exact, other cores as point masses `(M_b,n_b)`, halo individual — predicted
for the wells the law covers, true otherwise) and compare, by cosine and relative error, to the
FULL exact force (every other well individually), by core size (n10–29, n30–99, n100+, and
pooled n30+).

## 3. Result (median cos, 5 discovery tapes, `n30+` pooled)

| tape | h=5 | h=50 | h=250 |
|---|---|---|---|
| 11 (fit) | 0.99988 | 0.99982 | 0.99977 |
| 12 (fit) | 0.99997 | 0.99996 | 0.99998 |
| 21 (fit) | 0.99993 | 0.99989 | 0.99988 |
| 13 (validate) | 0.99998 | 0.99997 | 0.99996 |
| 22 (validate) | 0.99999 | 0.99999 | 0.99999 |

Law (iii) meets **median cos ≥ 0.999 for n30+ on all 5 discovery tapes at all three horizons**,
and is never worse than the rigid or frozen baselines; it is the clear winner at n≥100 and long
horizon specifically (e.g. tape 21, n100+, h250: baseline 0.828, rigid 0.949, **law iii 0.998**;
tape 11, n100+, h250: baseline 0.901, rigid 0.952, **law iii 0.999**). The n100+ bin alone is
noisier (per-tape median as low as 0.964 at h50 on fit tape 21) — consistent with the known
sensitivity of that bin (FINAL_W11_RIVER_CORE_HALO.md §2: this is the bin where dropping the
halo entirely, or folding it into core centres, fails outright at cos 0.73–0.94). Law (iii)
never fails that badly; it stays ≥0.95 at n100+ alone in every discovery-tape/horizon cell
measured.

## 4. Frozen law

**Law (iii): own single-well dynamics**, integrated forward from the last known halo state
using `salt_w + 0.3·nr(attraction to point-mass cores + other halo)`, core point masses taken
from the (assumed continuously available) core-level trajectory. Implementation: `law_owndyn()`
in `w12c_law_test.py`.

## 5. Pass criterion (frozen before opening prospective tapes)

**Primary:** median core-force cosine ≥ 0.999, pooled over cores with n ≥ 30, using
law-(iii)-predicted halo positions, at a 50-step horizon, on **both** prospective tapes
(`tape_3_4242_31_zo.npz`, `tape_2_8080_32_zo.npz`), computed by `w12c_law_test.py` with no
retuning of the law or its parameters.

**Secondary (reported, not binding):** median cos ≥ 0.95 for the n100+ bin alone, at h=50, on
both prospective tapes; and the same primary check repeated at h=5 and h=250, to see whether the
law degrades over the horizon.

A tape that does not reach the primary threshold is reported as a FAIL for that tape; the
overall verdict is a PASS only if both prospective tapes pass.
