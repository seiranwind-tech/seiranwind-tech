"""W12C step 2+3: halo laws + decisive test.
Laws (predict halo well position at t+delta from core-level state only):
  (i)   rigid co-motion: offset to nearest core parallel-transported, angle held fixed
  (iii) own single-well dynamics: salt_w + 0.3*nr(attraction to point-mass cores + other halo),
        jointly integrated for the halo subset (low-dim, ~20-60 wells); core point masses are
        taken from the true tape trajectory at each 5-step frame (core-level state assumed
        already available/closed -- that is the whole point of not handing off the halo).
Decisive test: plug PREDICTED halo positions into the m=1-style core-force harness (own core
members exact, other cores as point masses (M_b,n_b), halo individual) and compare to the FULL
exact force (every other well individually) at eval time, by core size bin. Horizons: 5,50,250 steps."""
import sys, os, json, collections, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E
nr = E.normalize_rows

f, block, perm, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
gstep = int(sys.argv[5]) if len(sys.argv) > 5 else 4
laws_arg = sys.argv[6] if len(sys.argv) > 6 else "i,iii"
horizons = [int(x) for x in sys.argv[7].split(",")] if len(sys.argv) > 7 else [5, 50, 250]
laws_wanted = laws_arg.split(",")

eng = make_engine(block, perm, seed); c = eng.config; dt = c.dt; bw = eng.bandwidth
teeth, mask, skew = eng.teeth, eng.mask, eng.skew
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]; T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]; h = 12; g0 = 60

home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
for g in range(1, len(G)):
    prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
    s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
    if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
    back = cur == home; away[back] = 0; away[~back] += 1; sw = away >= h; home[sw] = cur[sw]; away[sw] = 0; HOME.append(home.copy())

def tan_rows(V, M): return V - (np.sum(V * M, 1, keepdims=True)) * M
def cos1(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def pt_transport(v, Mfrom, Mto):
    dot = np.clip(np.sum(Mfrom * Mto, 1), -1, 1); theta = np.arccos(dot)
    sinT = np.sin(theta); small = sinT < 1e-9; sinT_safe = np.where(small, 1.0, sinT)
    e = (Mto - Mfrom * dot[:, None]) / sinT_safe[:, None]
    a = np.sum(v * e, 1); v_perp = v - a[:, None] * e
    transported = a[:, None] * (-Mfrom * sinT[:, None] + e * np.cos(theta)[:, None]) + v_perp
    return np.where(small[:, None], v, transported)

def salt_sub(z, idx):
    zn = nr(z); aff = np.einsum("ntd,nd->nt", teeth[idx], zn); aff = np.where(mask[idx] > 0, aff, -1e9)
    p = np.exp(3.0 * aff); p /= p.sum(1, keepdims=True) + 1e-9; up = np.triu(skew[idx], 1); sym = up + up.transpose(0, 2, 1)
    inner = np.einsum("nt,ntd->nd", p * (1.0 + c.salt * np.einsum("nab,nb->na", sym, p)), teeth[idx])
    return 0.5 * (inner - z * np.sum(z * inner, 1, keepdims=True))

def attr(zg, pop, logw, self_start=None):
    sim = zg @ pop.T
    if self_start is not None:
        ar = np.arange(len(zg)); sim[ar, self_start + ar] = -1e30
    L = sim / bw + logw[None]; L -= L.max(1, keepdims=True); w = np.exp(L); w /= w.sum(1, keepdims=True) + 1e-9
    return c.attract_gain * nr(w @ pop - zg)

def core_state(lab, z):
    u, cnt = np.unique(lab, return_counts=True); cores = u[cnt >= 5]
    Mc = {int(a): nr(z[lab == a].mean(0, keepdims=True))[0] for a in cores}
    nc = {int(a): int((lab == a).sum()) for a in cores}
    return cores, Mc, nc

def law_rigid(z_s, nearest, Mc_s, Mc_e):
    M_s = np.stack([Mc_s[int(a)] for a in nearest]); M_e = np.stack([Mc_e[int(a)] for a in nearest])
    offset = tan_rows(z_s, M_s); theta = np.arccos(np.clip(np.sum(z_s * M_s, 1), -1, 1))
    off_t = pt_transport(offset, M_s, M_e)
    nrm = np.linalg.norm(off_t, axis=1, keepdims=True)
    dirhat = np.where(nrm > 1e-9, off_t / (nrm + 1e-12), 0.0)
    return nr(M_e * np.cos(theta)[:, None] + dirhat * np.sin(theta)[:, None])

def law_owndyn(idxs, z0, gs, delta_steps):
    z = z0.copy(); n_h = len(idxs)
    for step in range(delta_steps):
        frame_idx = min(gs * 10 + 12 + step // 5, Z.shape[0] - 1)
        gate_idx = min(gs + step // 50, len(HOME) - 1)
        lab = HOME[gate_idx]; zc = nr(Z[frame_idx].astype(np.float32))
        cores, Mc, nc = core_state(lab, zc)
        if cores.size:
            coreM = np.stack([Mc[int(a)] for a in cores]); logw_core = np.log(np.array([nc[int(a)] for a in cores], float))
        else:
            coreM = np.zeros((0, z.shape[1])); logw_core = np.zeros(0)
        pop = np.concatenate([coreM, z]); lw = np.concatenate([logw_core, np.zeros(n_h)])
        a_step = attr(z, pop, lw, self_start=len(coreM))
        s_step = salt_sub(z, idxs)
        z = nr(z + dt * (s_step + a_step))
    return z

max_g = min(len(G) - 1, 113)
res = collections.defaultdict(list)  # (law, horizon, sizebin) -> [(cos, relerr)]

for gs in range(g0, max_g - 5, gstep):
    lab_s = HOME[gs]; fr_s = gs * 10 + 12; z_s_full = nr(Z[fr_s].astype(np.float32))
    cores_s, Mc_s, nc_s = core_state(lab_s, z_s_full)
    if cores_s.size == 0: continue
    is_core_s = np.isin(lab_s, cores_s); halo_idx = np.where(~is_core_s)[0]
    if len(halo_idx) == 0: continue
    coreM_s = np.stack([Mc_s[int(a)] for a in cores_s])
    sim = z_s_full[halo_idx] @ coreM_s.T; nn = np.argmax(sim, 1); nearest = cores_s[nn]

    for delta in horizons:
        dg = delta // 50
        ge = gs + dg
        if ge >= max_g: continue
        fr_e = fr_s + delta // 5
        if fr_e >= Z.shape[0]: continue
        lab_e = HOME[ge]; z_e_full = nr(Z[fr_e].astype(np.float32))
        cores_e, Mc_e, nc_e = core_state(lab_e, z_e_full)
        valid = np.isin(nearest, cores_e) if cores_e.size else np.zeros(len(nearest), bool)
        if valid.sum() == 0: continue
        v_idx = halo_idx[valid]; v_nearest = nearest[valid]
        z_s_v = z_s_full[v_idx]

        preds = {}
        if "i" in laws_wanted:
            preds["i_rigid"] = law_rigid(z_s_v, v_nearest, Mc_s, Mc_e)
        if "iii" in laws_wanted:
            preds["iii_owndyn"] = law_owndyn(v_idx, z_s_v, gs, delta)
        preds["baseline_frozen"] = z_s_v  # naive: halo doesn't move at all (sanity floor)

        # true halo array at eval time, for building the "true" swapped-in comparator + FULL population
        is_core_e = np.isin(lab_e, cores_e); halo_idx_e_all = np.where(~is_core_e)[0]

        for lawname, predpos in preds.items():
            z_test = z_e_full.copy(); z_test[v_idx] = predpos
            for a in cores_e:
                idx = np.where(lab_e == a)[0]; n = len(idx)
                if n < 10: continue
                M = nr(z_e_full[idx].mean(0, keepdims=True))[0]; zg = z_e_full[idx]; sa = salt_sub(zg, idx)
                order = np.concatenate([idx, np.setdiff1d(np.arange(N), idx)])
                full = tan_rows((sa + attr(zg, z_e_full[order], np.zeros(N), self_start=0)).mean(0, keepdims=True), M[None])[0]
                others = [b for b in cores_e if b != a]
                P = np.stack([Mc_e[int(b)] for b in others]) if others else np.zeros((0, z_e_full.shape[1]))
                W = np.array([nc_e[int(b)] for b in others], float)
                halo_pop = z_test[halo_idx_e_all]
                pop = np.concatenate([zg, P, halo_pop])
                lw = np.concatenate([np.zeros(n), np.log(W + 1e-9), np.zeros(len(halo_pop))])
                est = tan_rows((sa + attr(zg, pop, lw, self_start=0)).mean(0, keepdims=True), M[None])[0]
                key = "n10-29" if n < 30 else "n30-99" if n < 100 else "n100+"
                pair = (cos1(est, full), float(np.linalg.norm(est - full) / (np.linalg.norm(full) + 1e-12)))
                res[(lawname, delta, key)].append(pair)
                if n >= 30: res[(lawname, delta, "n30+")].append(pair)

out = {"tape": f.split("/")[-1]}
for (lawname, delta, key), v in sorted(res.items()):
    a = np.array(v)
    out[f"{lawname}_h{delta}_{key}"] = dict(count=len(v), cos_median=float(np.median(a[:, 0])), cos_p10=float(np.quantile(a[:, 0], 0.1)), relerr_median=float(np.median(a[:, 1])))
print(json.dumps(out, indent=1))
