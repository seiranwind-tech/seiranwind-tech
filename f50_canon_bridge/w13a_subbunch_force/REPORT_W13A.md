# REPORT W13A — sub-bunch representation of a core's own members: FAIL

**Line:** Claude Cloud exploratory analysis. This is **not** a Research OS or Science Supervisor
ruling, and **not** an F50 closure claim.

## Result

**FAIL.** The frozen candidate (spherical k-means sub-bunch, k=32 point masses, dimension 9632
numbers per core, independent of n) does not meet the pre-registered pass criterion on either
prospective tape.

| | pass bar | tape_4_5151_41_zo | tape_3_6262_42_zo |
|---|---|---|---|
| median cos, n >= 30 (pooled n30-99 + n100+) | >= 0.999 | **0.808** | **0.819** |
| median cos, n >= 100 | >= 0.99 | **0.738** | **0.808** |
| median cos, n 10-29 (not gated; trivial - k=32 >= typical n here, no real compression) | -- | 0.999 | 0.999 |
| median relative error, n30-99 | -- | 1.24 | 1.31 |
| median relative error, n100+ | -- | 3.11 | 2.29 |

This matches the discovery-tape trend exactly (n30-99: 0.81-0.91 median cos across 5 tapes; n100+:
0.73-0.83): the two prospective tapes reproduce the same failure mode with no surprises and no
degradation beyond what discovery already showed.

## What was tried

- **A - spherical k-means sub-bunch (frozen candidate).** k in {8,16,32,64,128} point masses,
  weight = member count, evaluated with the self-cluster weight corrected to exclude the querying
  member. Best of the three candidates, but even at k=128 (tape 11 discovery run) median cos only
  reaches 0.999/0.909 for n30-99/n100+ -- i.e. even when k approaches or exceeds n for many n30-99
  cores (near-trivial, no real compression) the n>=100 criterion (0.99) is still not met.
- **B - the engine's own raw instantaneous track/group id.** Rejected before freeze: inspection
  shows a river-home core's raw instantaneous group is almost always a single id (checked
  explicitly on 8 sampled cores of tape 11: 7/8 had exactly 1 raw group, the 8th had 5, one of
  which held 378/382 members and four were 1-well wanderers). The river/home layer (h=12
  hysteresis) exists precisely to *merge* raw excursions back into one group, so its raw
  instantaneous grouping essentially never subdivides a core into useful sub-bunches. Sample
  counts in the harness confirm this (1-4 cores per bucket per gate) -- not a usable candidate.
- **C - centroid + per-sub-bunch spread scalar, Monte-Carlo expanded.** Rejected before freeze:
  worse than A at every k tested on discovery tape 11 (k=32: cos_median 0.591 vs 0.772 for n100+,
  0.870 vs 0.907 for n30-99; k=16: 0.320 vs 0.658, 0.600 vs 0.718). Sampling synthetic members from
  an isotropic tangent perturbation matched only in RMS magnitude injects noise that is
  uncorrelated with each member's actual fine offset direction -- it does not correct the bias
  from collapsing a cluster to a single point, and the attraction's per-member unit normalisation
  (nr()) amplifies that mismatch, same mechanism W11 already found for PC reconstruction.

## Why this is consistent with W11/W12, not a contradiction

W11's `w11_shape_pc.py` found that top-k **principal-component reconstruction** (which keeps a
per-member k-dimensional coefficient -- an O(n*k) representation, not O(k)) needs k~=32 to reach
cos 0.9998/0.965-0.987/0.91-0.92 for n10-29/n30-99/n100+ respectively, and even that fails the
stricter 0.999/0.99 bar used here for n>=30/n>=100. A genuinely n-independent point-mass sub-bunch
(candidate A here) carries strictly less information than PC reconstruction at the same k (no
per-member coefficient, only a shared centroid per cluster), and scores worse at matched k (e.g.
k=32, n100+: ~0.77-0.83 here vs ~0.91-0.92 for PC). This is the expected ordering, not a new
failure mode: **the core's own internal shape is not compressible below O(n) information at
attraction accuracy**, confirming and sharpening W11 section 3's conclusion ("internal shape is
not compressible to a few moments") for the specific case of fixed-size sub-bunch point masses.

## What stays open

The core's own internal pull A_a (attraction term) still needs either (a) the exact member
positions, (b) an O(n*k) per-member reduced representation (W11 PC, itself insufficient at the
0.999/0.99 bar), or (c) a different low-dimensional description not yet tried (e.g. one that
targets the specific fine-offset structure the attraction's per-member normalisation is sensitive
to, rather than a generic point-mass or isotropic-spread cloud). W13A rules out plain k-means
point-mass sub-bunching, raw engine track structure, and isotropic Monte-Carlo spread correction
as candidates for that description.

## Commits
- Freeze (protocol + FREEZE_LOCK + harness, before opening prospective tapes): see
  `f50_canon_bridge/w13a_subbunch_force/FREEZE_LOCK.txt` for the sha256 of the frozen files.
- A small, non-substantive script change was made **after** the freeze commit and **before**
  running the prospective evaluation: an additional pooled `n30+` output bucket was added to
  `w13a_test.py` purely for reporting (to compute the pooled n>=30 median the pass criterion asks
  for); no change to the frozen candidate's representation, dimension, salt/attraction formulas,
  or per-bucket (n10-29 / n30-99 / n100+) numbers, which are identical before and after that edit.
  This is disclosed here for transparency.
- Results commit: this report, `RESULTS_W13A.json`, and the updated `w13a_test.py` /
  `FREEZE_LOCK.txt`.
