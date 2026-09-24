"""F50 parallel-universe (PU) exact engine + L1.5 observable table builder (W1).

PARALLEL-UNIVERSE SIMULATOR. Built from the documented rule text only; it is
NOT the canonical F50 engine and uses no F50 tape.

Hidden layer: unit vectors z_i on S^{d-1}; each belongs to one basin.
Basin centroid C_k is (re)assigned only at redraw (every 50 steps) = normalize(mean members).
Birth gate at redraw (per member j):
    s_own = z_j.C_own,  s_best = max_{k != own} z_j.C_k
    Gamma_birth(j) = 0.9 - max(s_own, s_best)
    Gamma_birth > 0  -> birth (new singleton basin at z_j)
    else if s_own < 0.9 <= s_best -> reassign to best
L1.5 observables are basin aggregates only (C, mu=M/n, n, R, skew, rival centroid, deltas, age)
plus an explicitly flagged PROPOSED augmentation (projected second moments).
"""
import numpy as np, json, sys, hashlib

TH, REDRAW, D = 0.9, 50, 6

def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def tangent(z, v):
    return v - (v * z).sum(-1, keepdims=True) * z

def run(seed, T=3000, N=500, K0=10):
    rng = np.random.default_rng(seed)
    C = unit(rng.normal(size=(K0, D)))
    lab = rng.integers(0, K0, N)
    z = unit(C[lab] + 0.12 * rng.normal(size=(N, D)))
    cent = {k: C[k] for k in range(K0)}
    born = {k: 0 for k in range(K0)}
    nxt = K0
    leak_v = np.zeros((N, D)); leak_t = np.zeros(N, int)
    basin_drift = {k: 0.0015 * unit(rng.normal(size=D)) for k in range(K0)}
    rows, events, moves = [], [], []
    prev = {}
    for t in range(T):
        ids = np.array(sorted(cent))
        Cm = np.stack([cent[k] for k in ids])
        idx = {k: i for i, k in enumerate(ids)}
        own = np.array([idx[l] for l in lab])
        Cown = Cm[own]
        # --- dynamics ---
        newleak = (leak_t == 0) & (rng.random(N) < 0.0025)
        leak_v[newleak] = 0.009 * unit(rng.normal(size=(newleak.sum(), D)))
        leak_t[newleak] = rng.integers(40, 160, newleak.sum())
        drift = np.stack([basin_drift[l] for l in lab])
        step = 0.012 * rng.normal(size=(N, D)) + 0.012 * (Cown - z) + drift + leak_v * (leak_t > 0)[:, None]
        z = unit(z + tangent(z, step)); leak_t = np.maximum(leak_t - 1, 0)
        # --- per-member hidden projections ---
        S = z @ Cm.T                                  # N x K
        s_own = S[np.arange(N), own]
        Sm = S.copy(); Sm[np.arange(N), own] = -2
        s_best = Sm.max(1)
        g_mem = TH - np.maximum(s_own, s_best)
        # --- basin aggregate table ---
        for k in ids:
            m = lab == k; n = int(m.sum())
            if n < 2:
                continue
            Z = z[m]; c = cent[k]; mu = Z.mean(0); R = float(np.linalg.norm(mu))
            skew = 1 - float(mu @ c) / R
            others = [j for j in ids if j != k]
            cr = cent[others[int(np.argmax([cent[j] @ c for j in others]))]]
            Y = np.stack([Z @ c, Z @ cr], 1); V = np.cov(Y.T, bias=True)
            p = prev.get(k)
            row = dict(seed=seed, step=t, basin=int(k), redraw_index=t // REDRAW, phase=t % REDRAW,
                       n=n, age=t - born[k], R=R, skew=skew, yo=float(mu @ c), yb=float(mu @ cr),
                       cob=float(c @ cr),
                       dR=R - p[0] if p else 0.0, dskew=skew - p[1] if p else 0.0,
                       dmu=float(np.linalg.norm(mu - p[2])) if p else 0.0,
                       so=float(np.sqrt(V[0, 0])), sb=float(np.sqrt(V[1, 1])),
                       rho=float(V[0, 1] / (np.sqrt(V[0, 0] * V[1, 1]) + 1e-12)),
                       G_birth=float(g_mem[m].max()), G_shape=float(TH - s_own[m].min()))
            prev[k] = (R, skew, mu)
            rows.append(row)
        # --- redraw ---
        if (t + 1) % REDRAW == 0:
            old = {k: cent[k].copy() for k in cent}
            birth_mask = g_mem > 0
            reas = (~birth_mask) & (s_own < TH) & (s_best >= TH)
            for i in np.where(reas)[0]:
                lab[i] = ids[int(np.argmax(Sm[i]))]
            for i in np.where(birth_mask)[0]:
                events.append(dict(seed=seed, step=t, parent=int(lab[i]), new=nxt))
                lab[i] = nxt; cent[nxt] = z[i].copy(); born[nxt] = t
                basin_drift[nxt] = 0.0015 * unit(rng.normal(size=D)); nxt += 1
            for k in list(cent):
                m = lab == k
                if not m.any():
                    del cent[k]; prev.pop(k, None); continue
                cent[k] = unit(z[m].mean(0))
                if k in old:
                    moves.append(dict(seed=seed, step=t, basin=int(k), W=float(np.linalg.norm(cent[k] - old[k]))))
            # singletons that drift back merge into nearest basin (keeps K bounded)
            for k in list(cent):
                m = np.where(lab == k)[0]
                if len(m) == 1 and t - born[k] >= 2 * REDRAW:
                    others = [j for j in cent if j != k]
                    j = others[int(np.argmax([z[m[0]] @ cent[j] for j in others]))]
                    lab[m[0]] = j; del cent[k]; prev.pop(k, None)
    return rows, events, moves

def save(prefix, seeds):
    allr, alle, allm = [], [], []
    for s in seeds:
        r, e, m = run(s); allr += r; alle += e; allm += m
        print(f"seed {s}: rows={len(r)} births={len(e)}", flush=True)
    cols = list(allr[0])
    arr = {c: np.array([r[c] for r in allr]) for c in cols}
    np.savez_compressed(prefix + "_table.npz", **arr)
    json.dump(dict(events=alle, moves=allm), open(prefix + "_events.json", "w"))

if __name__ == "__main__":
    split = sys.argv[1]; out = sys.argv[2]
    seeds = {"dev": [1, 2, 3, 4, 5, 6], "fresh": [101, 102, 103, 104, 105, 106]}[split]
    save(out, seeds)
