# W9 — Full L1.5-state closure (transport, birth, merge and centre channels): final report

**Line:** Claude Cloud exploratory analysis. This is not a Research OS or Science Supervisor ruling, and it is not a formal F50 closure.

## Inputs
- **Engine:** sha 4a71aeb4…, unmodified.
- **Vectors:** GloVe compact20k, sha a0997bb3….
- **Protocol and code:** PROTOCOL_W9.md 79eeed8c…, w9_shadow.py 07ef512c…. Frozen (commit 4699363) before any formal run; the lock was re-verified as OK afterwards.
- **Runs:** 3 × 6000 steps. Discovery (12345, b0, s1) and two new prospective runs, (3131, b0, s9) and (5757, b1, s11).

## Verdicts

| claim | result |
|---|---|
| **EXACT_L15_STATE_CLOSURE_WITH_AGGREGATE_HANDOFF** | **PASS**: 18,000 of 18,000 steps with 0 mismatches in K, labels, track ids, ages and the basin neighbour graph, and **bitwise-identical centres** (max abs error 0.0) at every step. This includes 334 births, 2,180 merges and 4,012 deaths. |
| HS_THINNING_ADMISSIBLE | **FAIL → HS_EVERY_UPDATE_REQUIRED**. With H_S only at reassign gates, track disagreement is 91–92% and there are about 5,500–5,800 spurious births per run. |
| Autonomy of H_S (member-motion channel) | Not tested, not claimed |

## The closed L1.5 map
L1.5 state: X = (labels ∈ ℕ^N, C ∈ S^{d−1 × K}, tracks, ages, basin graph G_B).

Static structure: the well kNN graph 𝒩 (fixed at init) and the constants (dt, ρ_B, κ_B, g = 0.15, bandwidth h, θ = 0.9, θ_merge = 0.98).

**Between gates** (cs mod 5 = 0):

 C⁺ = normalize((1−g)·P_G(C) + g·Ō)

 P_G(C)_i = normalize(C_i + dt·(ρ_B·blade_i + κ_B·(Σ_j softmax_j(C_i·C_j/h)·C_j − C_i))), with j ∈ G_B(i)

 Ō_i = normalize(Σ_{w∈i} zin_w / n_i)   ← H_S, the only L1 input

 G_B ← rank(labels, 𝒩, C)

**At a gate** (t_r = 50·m):
1. **Base scan** (the engine's own reassign pass): best_w, own_w, best_label_w.
2. **Transport:** label_w ← best_label_w if best_w ≥ θ.
3. **Birth:** a new basin with centre normalize(zin_w) for each w in O = {max(own, best) < θ} (the 256 lowest own if |O| > 256).
4. **Compact**, then take H_S on the new labels. If K changed, C ← Ō and ages ← 1; otherwise apply the mixing update.
5. **Merge:** union-find on G_B edges with C_i·C_j ≥ θ_merge, keep the minimum track id, centre = normalize(mean).

The only L1 → L1.5 inputs are H_S (a K × d first-moment vector every 5 steps and at gates) and H_T (per-well best label plus orphan zin at gates). **Everything else is L1.5-native and exact.**

## Payload
- **H_S:** about 79–84 KB per step on average (K × 300 float32 every 5 steps; K falls from about 1,900 to 100).
- **H_T:** about 2.9–3.2 KB per gate.
- H_S dominates the cost.

## What is still open
1. **The member-motion channel.** H_S = Ō(t) is the basin's first moment of zin, which moves under L1 dynamics: global softmax attraction, the zout wave, and the salt terms. Closing it would need an aggregate law for Σ_{w∈i} Δzin_w, which involves inter-well kernel sums. It is the only remaining non-L1.5 input.
2. H_T at gates also needs per-well information from the base scan. W8 showed birth needs only about 483 B per gate. Transport needs per-well best labels.
3. This is exact reproduction of L1.5 given the handoff, not autonomous L1.5 dynamics.

**Claim ceiling:** exact on 3 GloVe runs. It establishes that H_S and H_T are a sufficient interface. It is not a full autonomous closure.
