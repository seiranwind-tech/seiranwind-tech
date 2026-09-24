"""W11 macro rollouts (river groups, fixed partition blocks of 10 gates).
State: M_a (centre), latent A_a (attraction group-mean velocity). Salt is the closed mean-field S_a(M) = mean_{w in a} salt_w(M_a).
 L0 AR3 universal (baseline, coef from train tape)
 L1 salt-MF only                  M+ = nr(M + 5dt*S(M))
 L2 salt-MF + frozen A            A = A(t0) estimated from last frame
 L3 salt-MF + A with AR1 decay   A+ = rho*A
Everything computed with the engine only as static function library (teeth)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E
nr = E.normalize_rows
def ang(a, b): return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))
f, block, perm, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); AR3 = [1.1639, 0.5491, -0.7167]; RHO = float(sys.argv[5]) if len(sys.argv) > 5 else 0.99
eng = make_engine(block, perm, seed); dt = eng.config.dt
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]; T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]; h = 12; g0 = 60; B = 10
home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
for g in range(1, len(G)):
    prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
    s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
    if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
    back = cur == home; away[back] = 0; away[~back] += 1; sw = away >= h; home[sw] = cur[sw]; away[sw] = 0; HOME.append(home.copy())
def gmean(x, inv, K): S = np.zeros((K, x.shape[1]), np.float64); np.add.at(S, inv, x); return S / np.bincount(inv, minlength=K)[:, None]
def saltMF(M, inv, K):
    zm = M[inv].astype(np.float32); inner = eng._fsalt(zm); s = 0.5 * (inner - zm * np.sum(zm * inner, 1, keepdims=True)); return gmean(s, inv, K)
Hs = [1, 10, 50]; out = {m: {k: [[], []] for k in Hs} for m in ["L0_AR3", "L1_saltMF", "L2_saltMF+Afrozen", "L3_saltMF+A_AR1"]}
for gA in range(g0, len(G) - B + 1, B):
    lab = HOME[gA]; u, inv = np.unique(lab, return_inverse=True); K = len(u); cnt = np.bincount(inv); keep = cnt >= 5
    fr0 = gA * 10 + 9; frs = np.arange(fr0, min(fr0 + 10 * B, len(Z)))
    Ms = np.stack([nr(gmean(Z[x].astype(np.float32), inv, K)) for x in frs])      # [F,K,d]
    for t0 in range(3, len(frs) - 50, 10):
        M0 = Ms[t0]; truth = {k: Ms[t0 + k] for k in Hs}; disp = {k: ang(M0, truth[k])[keep] for k in Hs}
        # L0
        Mc = M0.copy(); ds = [Ms[t0 - i] - Ms[t0 - i - 1] for i in range(3)]
        for k in range(1, 51):
            dn = sum(AR3[i] * ds[i] for i in range(3)); Mc = nr(Mc + dn); ds = [dn] + ds[:-1]
            if k in Hs: out["L0_AR3"][k][0].append(ang(Mc, truth[k])[keep]); out["L0_AR3"][k][1].append(disp[k])
        A0 = (Ms[t0] - Ms[t0 - 1]) - 5 * dt * saltMF(Ms[t0 - 1], inv, K)
        for m in ["L1_saltMF", "L2_saltMF+Afrozen", "L3_saltMF+A_AR1"]:
            Mc = M0.copy(); A = A0.copy() if m != "L1_saltMF" else 0 * A0
            for k in range(1, 51):
                if m == "L3_saltMF+A_AR1": A = RHO * A
                Mc = nr(Mc + 5 * dt * saltMF(Mc, inv, K) + A)
                if k in Hs: out[m][k][0].append(ang(Mc, truth[k])[keep]); out[m][k][1].append(disp[k])
R = {"tape": f.split("/")[-1], "rho": RHO}
for m, v in out.items():
    R[m] = {f"{5*k}steps": dict(err=float(np.mean(np.concatenate(e))), disp=float(np.mean(np.concatenate(d))), skill=float(1 - np.mean(np.concatenate(e)) / np.mean(np.concatenate(d)))) for k, (e, d) in v.items()}
print(json.dumps(R, indent=1))
