# PROTOCOL W13A — sub-bunch representation of a core's own members (frozen before prospective)

**Line:** Claude Cloud exploratory analysis. This is **not** a Research OS or Science Supervisor
ruling, and **not** an F50 closure claim.

**Question.** W11 (`w11_shape_pc.py`) showed that top-k principal-component reconstruction of a
core's own members needs k ≈ 32 (and still needs O(n·k) numbers, since every member keeps its own
k coefficients). This protocol asks whether a genuinely **low-dimensional, n-independent**
sub-bunch (次生葡萄串) description of a core's OWN members — a handful of point masses, dimension
fixed regardless of core size — can reproduce the core's group-mean attraction/force as well as
the exact members do, using the harness and closed environment from W11/W12 (other cores as
point masses (M_b, n_b), halo wells individually).

## Candidates tried (discovery, tapes tape_0_1357_11, tape_1_1357_12, tape_0_9753_21; validated
on tape_2_1357_13_zo, tape_1_9753_22_zo)

All candidates keep the environment fixed (other cores as point masses, halo individual) and the
already-closed S_MF mean-field salt (every member placed at the group centre M_a, own teeth kept)
— this isolates the cost of reducing the OWN attraction population/query to a sub-bunch.

- **A — spherical k-means sub-bunches**, k ∈ {8,16,32,64,128}, each a point mass at centroid C_j
  with weight w_j = member count. The core force is the count-weighted average of the attraction
  evaluated **at each centroid**, with the OWN population being the k centroids themselves and the
  self-cluster weight corrected to (w_j−1) (excluding the querying member). Dimension: k·(d+1)
  numbers per core, independent of n.
- **B — the engine's own raw instantaneous track/group id** (`trk` at the sampled gate) as the
  sub-bunch label, instead of k-means. **Rejected outright**: inspection (and the harness) shows
  a river-home core's raw instantaneous group is almost always a single raw id (1 raw group for
  the large majority of home-labelled cores across 8 sampled cores on tape 11; a few wanderers
  split off 1-well raw groups). It essentially never subdivides a core into useful sub-bunches, so
  it cannot serve as a sub-bunch scheme (sample counts collapse: e.g. only 1-4 cores per bucket per
  gate, cos as low as 0.14–0.93, no consistent signal). This is a **negative finding**, kept in
  `w13a_test.py` (candidate `B`) for the record, not carried forward.
- **C — A + a per-sub-bunch spread scalar** (RMS 1−cos to centroid), expanded at evaluation time
  into `S_synth` synthetic members via an isotropic tangent perturbation of that magnitude,
  renormalized, attraction evaluated per synthetic member and averaged (Monte-Carlo estimate of
  E[nr(attraction)] over the cluster instead of nr(E[attraction]) at the centroid). **Worse than A
  at every k tested** (e.g. k=32, tape 11: cos_median 0.591 vs 0.772 for n100+, 0.870 vs 0.907 for
  n30-99). The isotropic assumption injects noise uncorrelated with each member's true fine offset
  (the same per-member normalisation sensitivity that made PC need k≈32); it does not fix the bias
  from collapsing to a point mass. Rejected.

## Frozen representation: **A — spherical k-means sub-bunch, k = 32**

Per-core state: 32 centroids (32×300 floats) + 32 counts = 9632 numbers, independent of n (far
below the 32×300×n needed to store n exact members, and no per-member coefficients as PC needs).

Attraction estimate:
```
A_est(a) = (1/n) * sum_j w_j * attract_gain * nr( sum_pop softmax(sim/bw + logw)·pop  -  C_j )
```
where `pop` = {other (k-1) centroids at full log w_l; own centroid at log(w_j-1)} ∪ {other cores'
centroids at log n_b} ∪ {halo wells individually, log-weight 0}.
Total core force estimate = tangent( S_MF(M_a) + A_est(a) , M_a ), compared by cos / relative
error to the FULL exact force (exact per-member salt + exact attraction over the true full
population), by core size (n 10–29, 30–99, ≥100).

## Discovery results (candidate A, k=32; median cos, 5-tape summary)

| tape | n10-29 (trivial: k≥n, no reduction) | n30-99 | n100+ |
|---|---|---|---|
| tape_0_1357_11 | 0.9995 | 0.907 | 0.772 |
| tape_1_1357_12 | 0.9991 | 0.890 | 0.733 |
| tape_0_9753_21 | 0.9993 | 0.908 | 0.796 |
| tape_2_1357_13_zo | 0.9983 | 0.888 | 0.802 |
| tape_1_9753_22_zo | 0.9995 | 0.805 | 0.826 |

(n10-29 core sizes are ≤29 < k=32, so k-means there degenerates to the exact members — that
bucket is not informative about compression and is not part of the pass criterion below.)

Consistent, non-trivial finding across all 5 discovery tapes: candidate A's median cos is
**0.80–0.91 for n30-99** and **0.73–0.83 for n≥100** — far below the pass bar, and this holds even
though k=32 is already near the "far below 32×300 numbers per core" ceiling suggested for this
task. Pushing k up to 128 (tape 11) only reaches 0.999/0.909 for n30-99/n100+ respectively — i.e.
even at k≈n for many n30-99 cores (near-trivial) and k still < n for most n100+ cores, the
criterion is still not met for n≥100. Increasing k further would abandon the "low-dimensional,
n-independent" premise of this test.

**Conclusion at freeze time: candidate A is expected to FAIL the pass criterion below on n≥30 and
n≥100, prospectively.** It is frozen anyway, per protocol, and evaluated once on the two held-out
prospective tapes for an honest, pre-registered prospective number.

## Pass criterion (frozen before opening the prospective tapes)

On **both** prospective tapes `tape_4_5151_41_zo.npz` and `tape_3_6262_42_zo.npz`:
- median cos ≥ 0.999 for cores with n ≥ 30 (buckets n30-99 and n100+ pooled), and
- median cos ≥ 0.99 for cores with n ≥ 100,
against the FULL exact force, using candidate A (spherical k-means sub-bunch, k=32) exactly as
implemented in `w13a_test.py`.

**Dimension used:** 32 point masses × (300-d centroid + 1 count) = 9632 numbers per core,
independent of core size n — well below the 32×300×n needed for n exact members, and below the
O(n·32) needed by the W11 top-k PC reconstruction.

## Files
- `w13a_test.py` — engine-as-library harness (candidates A, B, C), unmodified engine, static
  functions only (`from w10_common import make_engine, E`).
- `FREEZE_LOCK.txt` — sha256 of this protocol and of `w13a_test.py` at freeze time.
- `RESULTS_W13A.json`, `REPORT_W13A.md` — written after the single prospective evaluation.
