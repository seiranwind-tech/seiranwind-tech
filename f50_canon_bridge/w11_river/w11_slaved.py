"""W11 adiabatic elimination test: is a group's internal shape slaved to its centre M?
For a sampled group a at frame f (members fixed, rest of population frozen at tape positions):
  run the member dynamics (salt_w + 0.3*nr(attraction over population)) with the group mean PINNED to M_a
  (common-mode removed each step), starting from (i) all members collapsed at M_a + tiny noise, (ii) the true tape config.
  After S internal steps read the common-mode velocity D_slaved = mean_w F_w  (tangent to M).
Compare with actual D_a (tape) and between (i) and (ii) -> if (i)~(ii)~actual, internal state is a function of M."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E
nr = E.normalize_rows
f, block, perm, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); S_list = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [25, 100, 400]
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
def fsalt_sub(z, idx):
    zn = nr(z); aff = np.einsum("ntd,nd->nt", teeth[idx], zn); aff = np.where(mask[idx] > 0, aff, -1e9)
    p = np.exp(3.0 * aff); p /= p.sum(1, keepdims=True) + 1e-9; up = np.triu(skew[idx], 1); sym = up + up.transpose(0, 2, 1)
    inter = np.einsum("nab,nb->na", sym, p); return np.einsum("nt,ntd->nd", p * (1.0 + c.salt * inter), teeth[idx])
def force(zg, idx, others):
    inner = fsalt_sub(zg, idx); salt = 0.5 * (inner - zg * np.sum(zg * inner, 1, keepdims=True))
    pop = np.concatenate([zg, others]); sim = zg @ pop.T; sim[np.arange(len(zg)), np.arange(len(zg))] = -1e30
    w = np.exp((sim - sim.max(1, keepdims=True)) / bw); w /= w.sum(1, keepdims=True) + 1e-9
    return salt + c.attract_gain * nr(w @ pop - zg)
def tan(v, M): return v - np.dot(v, M) * M
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
rng = np.random.default_rng(0); res = []
for g in range(g0, len(G) - 1, 12):
    fr = g * 10 + 12; lab = HOME[g]; z = nr(Z[fr].astype(np.float32)); z1 = nr(Z[fr + 1].astype(np.float32))
    u, cnt = np.unique(lab, return_counts=True)
    for a in u[cnt >= 8][:4]:
        idx = np.where(lab == a)[0]; oth = np.setdiff1d(np.arange(N), idx); M = nr(z[idx].mean(0, keepdims=True))[0]
        Dact = tan((z1[idx] - z[idx]).mean(0) / (5 * dt), M); Dexact = tan(force(z[idx], idx, z[oth]).mean(0), M)
        row = dict(g=int(g), n=int(len(idx)), cos_exact_actual=cos(Dexact, Dact))
        for init in ["collapsed", "tape"]:
            zg = (M[None] + 1e-3 * rng.standard_normal((len(idx), len(M)))).astype(np.float32) if init == "collapsed" else z[idx].copy()
            zg = nr(zg); done = 0
            for S in S_list:
                for _ in range(S - done):
                    zg = nr(zg + dt * force(zg, idx, z[oth])); m = nr(zg.mean(0, keepdims=True))[0]
                    # pin: rotate the cloud back so its mean is M (small-angle: subtract common displacement)
                    zg = nr(zg - (m - M)[None])
                done = S; Ds = tan(force(zg, idx, z[oth]).mean(0), M)
                row[f"{init}_S{S}_cos_actual"] = cos(Ds, Dact); row[f"{init}_S{S}_ratio"] = float(np.linalg.norm(Ds) / np.linalg.norm(Dact))
                row[f"{init}_S{S}_spread"] = float(1 - np.mean(zg @ M))
            row[f"{init}_final"] = zg
        row["cos_collapsed_vs_tape_final_D"] = cos(tan(force(row.pop("collapsed_final"), idx, z[oth]).mean(0), M), tan(force(row.pop("tape_final"), idx, z[oth]).mean(0), M))
        row["tape_spread"] = float(1 - np.mean(z[idx] @ M)); res.append(row); print(json.dumps(row), flush=True)
keys = [k for k in res[0] if k not in ("g", "n")]
print(json.dumps({"median": {k: float(np.median([r[k] for r in res])) for k in keys}, "groups": len(res)}, indent=1))
