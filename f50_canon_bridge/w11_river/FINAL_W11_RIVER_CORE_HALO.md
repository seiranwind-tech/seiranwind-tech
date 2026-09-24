# W11 — River / typhoon coarse-graining of L1.5: tape-replay discovery report

**Line:** Claude Cloud exploratory analysis. This is **not** a Research OS or Science Supervisor ruling, and **not** an F50 closure claim.

**Mode:** DISCOVERY by tape replay ("錄影帶回放"), as requested by the user.
- The engine was run unmodified, and every analysis was done afterwards on recorded tapes.
- No prospective or held-out gate was frozen for W11. Any law stated below is a *candidate* and needs a later frozen prospective test.
- Cross-tape transfer is reported where it was measured: the AR coefficients were fitted on one tape and applied unchanged to four others.

## 0. Tapes
- **Engine:** f40_v39r3_engine.py, sha 4a71aeb4…, frozen. Profile F40_V3_9_NATIVE_L1_NONCANONICAL, N=2000, d=300, T=6000.
- **Vectors:** GloVe compact20k (the W6 cache).
- **Five tapes:** (block, perm, seed) = (0,1357,11), (1,1357,12), (0,9753,21), (2,1357,13)+zout, (1,9753,22)+zout.
- **Recorded per tape:**
  - the track id of every well at every step
  - zin (float16) every 5 steps
  - engine centres every 5 steps
  - zout every 5 steps, on the two "+zout" tapes
- **Analysis window:** late regime, gates ≥ 60 (steps 3000–6000), where K ≈ 25–60.
- **Recorder:** `w11_tape.py` / `w11_tape_zo.py`.

## 1. The user's picture, tested on tape
**Picture:** a wandering singleton is a member of its group on a short excursion, so count it inside the group. The back-and-forth oscillation is an *internal state* of the group, not an event.

**Births (`w11_replay.py`)**
- Every birth in every tape is a **singleton**: size p50 = p90 = max = 1.
- **Fate of born tracks:** 59–83% return to their parent group, 19–31% go elsewhere, 7–11% are still alive at the end of the tape.
- **Lifetime:** median about 10 gates.

**Excursions:** of the wells that separated from their companions, **92–94% rejoined them within 20 gates**. The median lag is 2–3 gates and p90 is 11–12 gates.

**River layer (`w11_river.py`)**
- Rule: a well keeps its **home** group while it wanders. Home switches only after the well has been away for h consecutive gates. Merged tracks are carried to their successor.
- **L1.5 label events per gate:**

| tape | raw | h=4 | h=12 | h=20 |
|---|---|---|---|---|
| 0_1357_11 | 26.1 | 1.70 | 1.27 | 0.75 |
| 1_1357_12 | 31.5 | 1.03 | 1.07 | 0.88 |
| 0_9753_21 | 37.2 | 2.23 | 1.58 | 1.02 |
| 2_1357_13 | 21.2 | 0.92 | 0.47 | 0.38 |
| 1_9753_22 | 22.8 | 3.08 | 1.38 | 1.05 |

**→ 94–98% of the H_T gate events are internal oscillation**, and they disappear in the river layer. The fraction of wells wandering at any moment is 0.4–1.4%.

**Remaining macro events at h=12 (`w11_macro_events.py`)**
- About 0.5–1.6 events per gate.
- Roughly one third are wanderers that permanently found a new group.
- Roughly two thirds are **boundary transfers to an adjacent group**. The destination is the most-similar group in 44–92% of cases and among the 3 most similar in 90–100%, with centre similarity about 0.84–0.94 against a typical value of about 0.

## 2. Core-centre motion: the forces, taken apart
Engine fact: `basin_feedback = 0` is enforced, so the basin layer never feeds back into zin. The whole L1.5 layer is therefore a readout of well motion, and the question is which *low-dimensional* description of the cores (葡萄串 / 多人盆) closes.

**Force accounting (`w11_meanfield.py`)**
- The exact group-mean force reproduces the actual 5-step group displacement with **R² 0.998**, on both zout tapes.

**The wave / zout clock is not the driver (`w11_wave_drive.py`)**
- zout is autonomous: its update never reads zin. It is therefore an exogenous clock.
- Its group mean, however, cancels: R² 0.0005–0.0009 and cos about 0.05. **This hypothesis is falsified.**

**Salt is mean-field closed**
- Salt was evaluated with every member placed at its group centre (S_MF, using only M_a and the group's static teeth bundle).
- It reproduces the true group salt: **R² 0.987–0.993, cos 0.99985–0.99988**.

**Attraction is not point-mass closed**
- Attraction with members collapsed to the centre gives R² −3.3 to −4.3.
- Salt and attraction nearly balance: cos(S, −A) = 0.93, |S| ≈ 0.039, |A| ≈ 0.050, |D| ≈ 0.020.
- The net motion is therefore a small difference of two large forces. A simple mobility law D = c·S_MF fails (R² 0.08).

**Core / halo decoupling — the typhoon picture (`w11_decouple.py`, `w11_subbunch.py`)**
- **Core** = river group with n ≥ 5. **Halo** = everything else: singleton basins and tiny groups, about 1–1.3% of wells.
- The core force was recomputed with the attraction population replaced by a reduced description:

| environment of core a | n 10–29 | n 30–99 | n ≥ 100 |
|---|---|---|---|
| other cores as **point masses (M_b, n_b)** + **halo wells individually** | cos 0.9998–0.99994, err 1.7–2.4% | cos 0.99999, err 0.25–0.5% | **cos 0.9999994, err 0.1%** |
| other cores as point masses, halo dropped | 0.997–0.999 | 0.9998 | 0.73–0.94 (fails) |
| other cores as sub-bunches (次生葡萄串, m=20), halo dropped | 0.999 | 0.99996 | 0.73–0.94 (fails) |
| halo folded into the nearest core centre | 0.9993–0.9998 | 0.99994–0.99997 | 0.73–0.93 (fails) |
| own members only (no other cores, no halo) | median over all sizes 0.30–0.32 (fails) | | |

**→ The coupling of a core to its environment is low-dimensional:**
- The other cores enter only as point masses (M_b, n_b). Sub-bunch structure of the *other* cores is irrelevant.
- The halo ring (about 20 wells) must be kept as individual positions. For large cores it cannot be folded into a centre: this is the "outer circulation" of the typhoon.

## 3. The one remaining unclosed part: the core's own internal shape
All of the environment results above keep the core's **own** members at their exact positions.

**Is the internal shape slaved to M, i.e. can it be eliminated adiabatically? (`w11_slaved.py`)**
- Members were relaxed from a collapsed start with the centre pinned. This does **not** recover the tape configuration:
  - cos(D_collapsed, D_tape) has a median of 0.77 on tape 13 and 0.95 on tape 22.
  - The relaxed collapsed cloud stays about 9–14× tighter than on tape after 400 steps (median spread).
- **The internal shape is a slow, memory-carrying variable, not a fast slaved one.**

**Dimension of the internal shape (`w11_shape_pc.py`)**
- Own members were replaced by their top-k principal-axis reconstruction, with the environment described as above.
- The force needs k ≈ 32: cos 0.9998 at n 10–29, 0.965–0.987 at n 30–99, 0.91–0.92 at n ≥ 100.
- With k ≤ 8 the error is 1.7–9×. Per-member unit normalisation of the attraction amplifies sub-degree offsets.
- **The internal shape is not compressible to a few moments.** This is consistent with W10's P2 counterexample.

## 4. Candidate macro law (discovery level)
State per core a: centre M_a, size n_a, and a latent internal-pull vector A_a. Environment: {(M_b, n_b)} ∪ halo positions.

```
M_a⁺ = normalize( M_a + 5dt·[ S_MF,a(M_a) + A_a ] )     # S_MF closed: R² 0.99
A_a  : latent core-internal pull (not closed; slow, memory ≈ hundreds of steps)
coupling of A_a to the environment: other cores as point masses + halo ring (cos 0.9999+)
events: macro events ≈ 0.5–1.6 per gate (raw 21–37); boundary transfers go to the nearest neighbours
```

**Empirical closure of the latent part by a universal AR law (`w11_ar.py`)**
- Law: d_{t+1} = Σ a_i d_{t−i} on 5-step centre increments.
- The coefficients were fitted on tape 11 and applied unchanged to the four other tapes.

| law | 5 steps | 50 steps | 250 steps (held-out tapes) |
|---|---|---|---|
| AR1 (0.997) | 0.95–0.96 | 0.80–0.83 | 0.33–0.43 |
| AR3 (1.164, 0.549, −0.717) | 0.97–0.98 | 0.88–0.90 | 0.42–0.48 |
| AR8 | 0.97–0.98 | 0.90–0.91 | 0.45–0.52 |
| S_MF + frozen A (`w11_rollout.py`) | 0.94–0.95 | 0.81–0.82 | 0.38–0.41 |

The coefficients are identical across tapes: no refit, and no transfer loss.

## 5. Verdict (exploratory; no formal closure claim)
- **H_T (transport-gate events):** the river layer removes **94–98%** of them. The rest are nearest-neighbour boundary transfers plus rare permanent singleton foundings. H_T is **reduced, not eliminated**.
- **Core–environment coupling:** **closed at low dimension.** It needs point-mass cores plus the halo ring, with cos ≥ 0.9999 for n ≥ 30.
- **Salt:** **mean-field closed** (R² 0.99).
- **Wave / zout clock:** irrelevant at the core level.
- **Open gap:** each core's own internal pull A_a. It is a slow latent variable that is not reducible to a few shape moments. Today it is carried only empirically, by a universal AR law: about 0.97 skill at 5 steps, 0.89 at 50, 0.45 at 250.
- **Even if A_a were closed**, the residual H_T boundary-transfer channel (about 1 event per gate) would still need its own law.

## 6. Next gap, concrete
1. **An evolution law for A_a.** Candidate: dA_a/dt = −(A_a + P_a)/τ + coupling to the environment and the halo, with P_a the balance pull. Fit it on tapes, then freeze it and test it prospectively.
2. **A rate law for boundary transfers**, as a function of (M_a, M_b, n_a, n_b), so that H_T becomes a macro flux.
3. **A halo-ring law** (about 20 wells), to test the "outer circulation co-moves with the core" picture directly.
