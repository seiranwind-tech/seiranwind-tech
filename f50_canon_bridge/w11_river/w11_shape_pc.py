"""W11 internal-shape dimension test. Environment = other cores as point masses + halo wells individually (the closed coupling).
Own members replaced by M + projection of offsets onto top-k principal axes of the core cloud (k = 0,1,2,4,8,16,32).
Cos / rel error of the core force vs FULL exact force, by core size."""
import sys, os, json, collections, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E
nr = E.normalize_rows
f, block, perm, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); ks = [0, 1, 2, 4, 8, 16, 32]
eng = make_engine(block, perm, seed); c = eng.config; bw = eng.bandwidth
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]; T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]; h = 12; g0 = 60
home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
for g in range(1, len(G)):
    prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
    s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
    if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
    back = cur == home; away[back] = 0; away[~back] += 1; sw = away >= h; home[sw] = cur[sw]; away[sw] = 0; HOME.append(home.copy())
teeth, mask, skew = eng.teeth, eng.mask, eng.skew
def salt_sub(z, idx):
    zn = nr(z); aff = np.einsum("ntd,nd->nt", teeth[idx], zn); aff = np.where(mask[idx] > 0, aff, -1e9)
    p = np.exp(3.0 * aff); p /= p.sum(1, keepdims=True) + 1e-9; up = np.triu(skew[idx], 1); sym = up + up.transpose(0, 2, 1)
    inner = np.einsum("nt,ntd->nd", p * (1.0 + c.salt * np.einsum("nab,nb->na", sym, p)), teeth[idx])
    return 0.5 * (inner - z * np.sum(z * inner, 1, keepdims=True))
def attr(zg, pop, logw):
    sim = zg @ pop.T; sim[np.arange(len(zg)), np.arange(len(zg))] = -1e30
    L = sim / bw + logw[None]; L -= L.max(1, keepdims=True); w = np.exp(L); w /= w.sum(1, keepdims=True) + 1e-9
    return c.attract_gain * nr(w @ pop - zg)
def tan(v, M): return v - np.dot(v, M) * M
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
res = collections.defaultdict(list); evr = collections.defaultdict(list)
for g in range(g0, len(G) - 1, 4):
    fr = g * 10 + 12; z = nr(Z[fr].astype(np.float32)); lab = HOME[g]
    u, cnt = np.unique(lab, return_counts=True); cores = u[cnt >= 10]; allc = u[cnt >= 5]; is_core = np.isin(lab, allc); halo = z[~is_core]
    Mc = {b: nr(z[lab == b].mean(0, keepdims=True))[0] for b in allc}; nc = {b: int((lab == b).sum()) for b in allc}
    for a in cores:
        idx = np.where(lab == a)[0]; n = len(idx); M = Mc[a]; zg = z[idx]
        order = np.concatenate([idx, np.setdiff1d(np.arange(N), idx)]); full = tan((salt_sub(zg, idx) + attr(zg, z[order], np.zeros(N))).mean(0), M)
        P = np.stack([Mc[b] for b in allc if b != a]); Wl = np.log([nc[b] for b in allc if b != a])
        X = zg - zg.mean(0); U, Sv, Vt = np.linalg.svd(X, full_matrices=False); key = "n10-29" if n < 30 else "n30-99" if n < 100 else "n100+"
        evr[key].append(np.cumsum(Sv ** 2)[[min(k, len(Sv)) - 1 for k in [1, 2, 4, 8, 16]]] / np.sum(Sv ** 2))
        for k in ks:
            zk = zg.mean(0) + (X @ Vt[:k].T @ Vt[:k] if k > 0 else 0 * X); zk = nr(zk.astype(np.float32))
            pop = np.concatenate([zk, P, halo]); lw = np.concatenate([np.zeros(n), Wl, np.zeros(len(halo))])
            est = tan((salt_sub(zk, idx) + attr(zk, pop, lw)).mean(0), M)
            res[(k, key)].append((cos(est, full), float(np.linalg.norm(est - full) / np.linalg.norm(full))))
out = {"tape": f.split("/")[-1], "explained_var_top_1_2_4_8_16": {k: np.median(np.array(v), 0).round(3).tolist() for k, v in evr.items()}}
for (k, key), v in sorted(res.items()):
    a = np.array(v); out[f"k{k}_{key}"] = dict(count=len(v), cos_median=round(float(np.median(a[:, 0])), 5), cos_p10=round(float(np.quantile(a[:, 0], 0.1)), 4), relerr_median=round(float(np.median(a[:, 1])), 4))
print(json.dumps(out, indent=1))
