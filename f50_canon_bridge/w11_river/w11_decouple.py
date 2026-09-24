"""W11 core/halo decoupling (typhoon picture).
core  = river group (h=12 hysteresis, wanderers counted as members) with n>=5
halo  = all wells in singleton tracks that are not home-members of a core (permanent singletons) + small groups (n<5)
For each core a at sampled frames: exact group-mean force F_a (salt + 0.3*nr(attraction)), tangent to M_a, with the
attraction population = (FULL) all wells | (NO_HALO) cores only | (SELF) own members only | (SELF+CORES_PM) own members + other cores as point masses(weighted n_b)
Compare with FULL (which matches the actual motion, cos 0.999)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E
nr = E.normalize_rows
f, block, perm, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
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
def attr(zg, pop, logw=None, self_in_pop=True):
    sim = zg @ pop.T
    if self_in_pop: sim[np.arange(len(zg)), np.arange(len(zg))] = -1e30
    L = sim / bw + (0 if logw is None else logw[None]); L -= L.max(1, keepdims=True); w = np.exp(L); w /= w.sum(1, keepdims=True) + 1e-9
    return c.attract_gain * nr(w @ pop - zg)
def tan(v, M): return v - np.dot(v, M) * M
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
rows = []; halo_frac = []
for g in range(g0, len(G) - 1, 4):
    fr = g * 10 + 12; z = nr(Z[fr].astype(np.float32)); z1 = nr(Z[fr + 1].astype(np.float32)); lab = HOME[g]
    u, cnt = np.unique(lab, return_counts=True); cores = u[cnt >= 5]; is_core = np.isin(lab, cores); halo = np.where(~is_core)[0]
    halo_frac.append(float(np.mean(~is_core)))
    Mc = {a: nr(z[lab == a].mean(0, keepdims=True))[0] for a in cores}; nc = {a: int((lab == a).sum()) for a in cores}
    for a in cores:
        idx = np.where(lab == a)[0]; M = Mc[a]; zg = z[idx]; sa = salt_sub(zg, idx)
        Dact = tan((z1[idx] - zg).mean(0) / (5 * dt), M)
        order = np.concatenate([idx, np.setdiff1d(np.arange(N), idx)])
        full = tan((sa + attr(zg, z[order])).mean(0), M)
        coreidx = np.concatenate([idx, np.setdiff1d(np.where(is_core)[0], idx)])
        nohalo = tan((sa + attr(zg, z[coreidx])).mean(0), M)
        selfonly = tan((sa + attr(zg, zg)).mean(0), M)
        oth = [b for b in cores if b != a]
        if oth:
            P = np.stack([Mc[b] for b in oth]); pop = np.concatenate([zg, P]); lw = np.concatenate([np.zeros(len(idx)), np.log([nc[b] for b in oth])])
            selfpm = tan((sa + attr(zg, pop, lw)).mean(0), M)
        else: selfpm = selfonly
        rows.append(dict(n=len(idx), full_vs_actual=cos(full, Dact), nohalo_vs_full=cos(nohalo, full), self_vs_full=cos(selfonly, full), selfPM_vs_full=cos(selfpm, full),
                         r_nohalo=float(np.linalg.norm(nohalo - full) / np.linalg.norm(full)), r_self=float(np.linalg.norm(selfonly - full) / np.linalg.norm(full)), r_selfPM=float(np.linalg.norm(selfpm - full) / np.linalg.norm(full))))
import collections
bins=collections.defaultdict(list)
for r in rows: bins[("n5-9" if r["n"]<10 else "n10-29" if r["n"]<30 else "n30-99" if r["n"]<100 else "n100+")].append(r)
BY={b:dict(count=len(v),selfPM_cos_median=float(np.median([r["selfPM_vs_full"] for r in v])),selfPM_cos_p10=float(np.quantile([r["selfPM_vs_full"] for r in v],0.1)),nohalo_cos_p10=float(np.quantile([r["nohalo_vs_full"] for r in v],0.1)),r_selfPM_median=float(np.median([r["r_selfPM"] for r in v]))) for b,v in bins.items()}
R = {"by_size":BY, "tape": f.split("/")[-1], "cores_sampled": len(rows), "halo_well_fraction_median": float(np.median(halo_frac))}
for k in rows[0]:
    if k != "n": R[k] = dict(median=float(np.median([r[k] for r in rows])), p10=float(np.quantile([r[k] for r in rows], 0.1)), p90=float(np.quantile([r[k] for r in rows], 0.9)))
print(json.dumps(R, indent=1))
