"""W11 two-level grape test (葡萄串 / 次生葡萄串): environment of core a = other cores represented by
sub-bunch point masses (spherical k-means, k = ceil(n_b/m)), weights = sub-bunch counts; halo ignored.
Own members exact. Compare core force with the FULL exact force (cos, rel err), by core size."""
import sys, os, json, collections, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E
nr = E.normalize_rows
f, block, perm, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); ms = [int(x) for x in sys.argv[5].split(",")]
eng = make_engine(block, perm, seed); c = eng.config; dt = c.dt; bw = eng.bandwidth
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
def kmeans(x, k, it=15):
    rng = np.random.default_rng(0); C = x[rng.choice(len(x), k, replace=False)]
    for _ in range(it):
        a = np.argmax(x @ C.T, 1); C = np.stack([nr(x[a == j].mean(0, keepdims=True))[0] if (a == j).any() else C[j] for j in range(k)])
    a = np.argmax(x @ C.T, 1); return C, np.bincount(a, minlength=k)
def tan(v, M): return v - np.dot(v, M) * M
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
res = collections.defaultdict(list)
for g in range(g0, len(G) - 1, 4):
    fr = g * 10 + 12; z = nr(Z[fr].astype(np.float32)); lab = HOME[g]
    u, cnt = np.unique(lab, return_counts=True); cores = u[cnt >= 5]; is_core = np.isin(lab, cores)
    subs = {m: {b: kmeans(z[lab == b], 1 if m in (1, 2, 3) else int(np.ceil((lab == b).sum() / m))) for b in cores} for m in ms}
    for a in cores:
        idx = np.where(lab == a)[0]; M = nr(z[idx].mean(0, keepdims=True))[0]; zg = z[idx]; sa = salt_sub(zg, idx)
        order = np.concatenate([idx, np.setdiff1d(np.arange(N), idx)]); full = tan((sa + attr(zg, z[order], np.zeros(N))).mean(0), M)
        n = len(idx); key = "n10-29" if n < 30 else "n30-99" if n < 100 else "n100+" if n >= 100 else "n5-9"
        if n < 10: key = "n5-9"
        for m in ms:
            P = [subs[m][b][0] for b in cores if b != a]; W = [subs[m][b][1] for b in cores if b != a]
            if not P: continue
            P = np.concatenate(P); W = np.concatenate(W).astype(float); keep = W > 0
            halo = z[~is_core]
            if m == 1: pop = np.concatenate([zg, P[keep], halo]); lw = np.concatenate([np.zeros(n), np.log(W[keep]), np.zeros(len(halo))])
            elif m == 2:   # halo folded into nearest OTHER core as extra mass (user's 'count the wanderer in the group')
                Pc = P[keep]; Wc = W[keep].copy(); near = np.argmax(halo @ Pc.T, 1); sims_own = halo @ M; sims_oth = np.max(halo @ Pc.T, 1)
                own_side = sims_own > sims_oth
                np.add.at(Wc, near[~own_side], 1.0)
                pop = np.concatenate([zg, Pc]); lw = np.concatenate([np.zeros(n), np.log(Wc)])
                # halo nearest to OWN core: fold as extra members at own centre M
                if own_side.any(): pop = np.concatenate([pop, M[None]]); lw = np.concatenate([lw, [np.log(own_side.sum())]])
            elif m == 3:   # halo kept individually but other cores dropped
                pop = np.concatenate([zg, halo]); lw = np.zeros(len(pop))
            else: pop = np.concatenate([zg, P[keep]]); lw = np.concatenate([np.zeros(n), np.log(W[keep])])
            est = tan((sa + attr(zg, pop, lw)).mean(0), M)
            res[(m, key)].append((cos(est, full), float(np.linalg.norm(est - full) / np.linalg.norm(full))))
out = {"tape": f.split("/")[-1]}
for (m, key), v in sorted(res.items()):
    a = np.array(v); out[f"m{m}_{key}"] = dict(count=len(v), cos_median=float(np.median(a[:, 0])), cos_p10=float(np.quantile(a[:, 0], 0.1)), relerr_median=float(np.median(a[:, 1])))
print(json.dumps(out, indent=1))
