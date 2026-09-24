# W12C — Halo co-motion: prospective evaluation

**Line:** Claude Cloud exploratory analysis (F50 L1.5 closure project). **Not** a Research OS or
Science Supervisor ruling, and **not** an F50 closure claim.

Protocol and law frozen before opening prospective data: `PROTOCOL_W12C.md`,
`FREEZE_LOCK.txt`, commit `2d84765`. No retuning was done for this evaluation.

## Verdict: **PASS**

Both prospective tapes clear the primary pass criterion (median core-force cosine >= 0.999,
pooled over cores with n >= 30, using law-(iii)-predicted halo positions, at a 50-step horizon):

| tape | n30+ h=50 median cos | primary | n100+ h=50 median cos | secondary (>=0.95) |
|---|---|---|---|---|
| tape_2_8080_32_zo.npz | 0.99985 | PASS | 0.99911 | PASS |
| tape_3_4242_31_zo.npz | 0.99994 | PASS | 0.99957 | PASS |

`tape_3_4242_31_zo.npz` was re-recorded from scratch after its first save crashed (OOM); the
engine is deterministic and the (block, perm, seed) triple was unchanged, so this is the same
trajectory as originally scheduled.

Both tapes also pass the primary n30+ check at the other two tested horizons (5 and 250 steps,
all medians in 0.9993-0.9999), i.e. the law does not visibly degrade over the tested horizon
range. Neither prospective tape reproduces the one weak discovery-tape cell (fit tape 21, n100+,
h=50, median 0.964); both stay >=0.99 there. That earlier dip looks like tape-specific noise in a
small-count bin, not a systematic weakness of the law.

## Summary of the whole W12C line

1. **Co-motion is real**: the halo rigidly co-moves with its nearest core (per-well ring-angle
   std ~1-4 deg, parallel-transported offset cos 0.999 median over 50 steps), with its own faster
   jitter superposed (velocity cos 0.955 median) and finite dwell (median 3 gates on the same
   nearest core).
2. **A low-dimensional law closes it**: each halo well's own force law, integrated from
   core-level state alone (no individual non-halo data), predicts halo positions well enough
   that plugging them into the core force in place of the true halo positions barely moves the
   cosine to the full exact force -- the halo need not be handed off.
3. **Prospectively confirmed**, unchanged, on two held-out tapes.

Files: `w12c_comotion.py`, `w12c_law_test.py`, `PROTOCOL_W12C.md`, `FREEZE_LOCK.txt`,
`RESULTS_W12C.json`, `discovery_runs/`, `prospective_31.json`, `prospective_32.json`.
