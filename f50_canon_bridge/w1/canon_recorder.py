"""W1 (canonical line) — read-only tape recorder around the frozen F40 V3.9R3 engine.

Engine: legacy/V39R3_FROZEN/f40_v39r3_engine.py sha256 4a71aeb4a66a8a07... (verified at import).
Vectors: spaCy en_core_web_md-3.8.0 300d table (GloVe unavailable in this container).
Hidden gate, replicated from _reassign_sparse (does not modify the engine):
    s_own_j  = zin_j . C[label_j]
    s_best_j = max_k zin_j . C[label[nbr_jk]]    (k = neighbor_k reachable wells; may include own)
    Gamma_birth_j = thr - max(s_own_j, s_best_j);   birth iff Gamma_birth_j > 0 at a reassign step.
Hook runs before engine.step(); the reassign inside that step sees the same zin and centres.
"""
import sys, os, json, hashlib, numpy as np
ENG_DIR = os.environ["F50_ENGINE_DIR"]
ENG_SHA = "4a71aeb4a66a8a071a6b6f9a9b065687d237909aea5e89a537ee6d80d2aa3b1d"
assert hashlib.sha256(open(os.path.join(ENG_DIR, "f40_v39r3_engine.py"), "rb").read()).hexdigest() == ENG_SHA
sys.path.insert(0, ENG_DIR)
import f40_v39r3_engine as E

def make_engine(block, n=2000, seed=1):
    V = np.load(os.environ["F50_VEC"])
    perm = np.random.default_rng(12345).permutation(len(V))
    idx = np.sort(perm[block * n:(block + 1) * n])
    cfg = E.ScalableRunConfigV32(profile="F40_V3_9_NATIVE_L1_NONCANONICAL", glove_path="none",
                                 output_dir="/tmp/o", n_wells=n, dimension=V.shape[1], total_steps=2600, seed=seed)
    return E.ScalableF40EngineV32(cfg, [f"w{i}" for i in idx], V[idx])

def record(block, T=2600, t0=800):
    eng = make_engine(block); thr = float(eng.assignment_threshold); R_EVERY = eng.config.basin_reassign_every
    first_seen, prev, rows, moves, births = {}, {}, [], [], []
    last_c = {}
    for _ in range(T):
        cs = eng.step_number + 2                      # completed_step of the upcoming step
        zin = eng.zin; C = eng.centres; lab = eng.labels; tr = eng.centre_track_ids
        K = len(C)
        s_own = np.einsum("nd,nd->n", zin, C[lab])
        s_best = np.einsum("nd,nkd->nk", zin, C[lab[eng.neighbors]]).max(1)
        g = thr - np.maximum(s_own, s_best)
        for k in range(K):
            first_seen.setdefault(int(tr[k]), cs)
        if cs % R_EVERY == 0:                         # realised centre movement W per track since last reassign
            for k in range(K):
                t = int(tr[k])
                if t in last_c:
                    moves.append(dict(seed=block, step=cs, basin=t, W=float(np.linalg.norm(C[k] - last_c[t]))))
                last_c[t] = C[k].copy()
            born_mask = g > 0
            for j in np.where(born_mask)[0]:
                births.append(dict(seed=block, step=cs, parent=int(tr[lab[j]])))
        if cs >= t0:
            n_k = np.bincount(lab, minlength=K)
            M = np.zeros_like(C); np.add.at(M, lab, zin)
            mu = M / np.maximum(n_k, 1)[:, None]
            R = np.linalg.norm(mu, axis=1)
            # legal rival: nearest other centre in the engine's basin-neighbour graph
            bn = eng.basin_neighbors if eng.basin_neighbors is not None else None
            Ssim = C @ C.T; np.fill_diagonal(Ssim, -2)
            if bn is not None and bn.shape[0] == K:
                cand = np.where(bn == np.arange(K)[:, None], -1, bn)
                sc = np.where(cand >= 0, Ssim[np.arange(K)[:, None], np.maximum(cand, 0)], -2)
                riv = cand[np.arange(K), sc.argmax(1)]; riv = np.where(riv >= 0, riv, Ssim.argmax(1))
            else:
                riv = Ssim.argmax(1)
            Cr = C[riv]
            po = np.einsum("nd,nd->n", zin, C[lab]); pb = np.einsum("nd,nd->n", zin, Cr[lab])
            mo = np.bincount(lab, po, K) / np.maximum(n_k, 1); mb = np.bincount(lab, pb, K) / np.maximum(n_k, 1)
            vo = np.bincount(lab, po * po, K) / np.maximum(n_k, 1) - mo ** 2
            vb = np.bincount(lab, pb * pb, K) / np.maximum(n_k, 1) - mb ** 2
            cob_ = np.bincount(lab, po * pb, K) / np.maximum(n_k, 1) - mo * mb
            gmax = np.full(K, -9.0); np.maximum.at(gmax, lab, g)
            smin = np.full(K, 9.0); np.minimum.at(smin, lab, s_own)
            for k in np.where(n_k >= 2)[0]:
                t = int(tr[k]); Rk = float(R[k]); skew = 1 - float(mu[k] @ C[k]) / max(Rk, 1e-12)
                p = prev.get(t)
                so, sb = float(np.sqrt(max(vo[k], 0))), float(np.sqrt(max(vb[k], 0)))
                rows.append((block, cs, t, cs // R_EVERY, cs % R_EVERY, int(n_k[k]), cs - first_seen[t], Rk, skew,
                             float(mu[k] @ C[k]), float(mu[k] @ Cr[k]), float(C[k] @ Cr[k]),
                             Rk - p[0] if p else 0.0, skew - p[1] if p else 0.0,
                             float(np.linalg.norm(mu[k] - p[2])) if p else 0.0,
                             so, sb, float(cob_[k] / (so * sb + 1e-12)), float(gmax[k]), float(thr - smin[k])))
                prev[t] = (Rk, skew, mu[k].copy())
        eng.step()
        if len(eng.centre_track_ids) < 4:
            break
    # window convention: redraw step has phase 49 (same as PU line)
    cols = "seed step basin redraw_index phase n age R skew yo yb cob dR dskew dmu so sb rho G_birth G_shape".split()
    A = np.array(rows, dtype=np.float64)
    tab = {c: A[:, i] for i, c in enumerate(cols)}
    for c in ("seed", "step", "basin", "redraw_index", "phase", "n", "age"):
        tab[c] = tab[c].astype(np.int64)
    tab["redraw_index"] = (tab["step"] - 1) // R_EVERY; tab["phase"] = (tab["step"] - 1) % R_EVERY
    return tab, dict(events=births, moves=moves, lineage=eng.lineage_counts, threshold=thr)

def build(blocks, out):
    from multiprocessing import Pool
    with Pool(4) as p: res = p.map(record, blocks)
    tab = {c: np.concatenate([r[0][c] for r in res]) for c in res[0][0]}
    np.savez_compressed(out + "_table.npz", **tab)
    json.dump(dict(events=[e for r in res for e in r[1]["events"]], moves=[m for r in res for m in r[1]["moves"]],
                   lineage=[r[1]["lineage"] for r in res], threshold=res[0][1]["threshold"]), open(out + "_events.json", "w"))
    print(out, len(tab["step"]), "births", sum(len(r[1]["events"]) for r in res), [r[1]["lineage"] for r in res])

if __name__ == "__main__":
    split, out = sys.argv[1], sys.argv[2]
    build({"dev": [0, 1, 2, 3], "fresh": [4, 5, 6, 7], "fresh2": [8, 9]}[split], out)
