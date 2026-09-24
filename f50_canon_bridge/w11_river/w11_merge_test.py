"""W11: are macro merges decided by macro centres alone?  merge(a,b) at gate g  <=>  sim(M_a, M_b) >= 0.98 ?"""
import sys, json, numpy as np
def nr(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
f = sys.argv[1]; h = 12
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]; T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]; g0 = 20
home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
for g in range(1, len(G)):
    prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
    s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
    if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
    back = cur == home; away[back] = 0; away[~back] += 1; sw = away >= h; home[sw] = cur[sw]; away[sw] = 0; HOME.append(home.copy())
rows = []
for g in range(g0, len(G)):
    for name, lab in [("raw", H[g - 1]), ("river", HOME[g - 1])]:
        u, inv = np.unique(lab, return_inverse=True); cnt = np.bincount(inv)
        S = np.zeros((len(u), Z.shape[2]), np.float32); np.add.at(S, inv, Z[g * 10 + 8].astype(np.float32)); M = nr(S)
        maj_after = np.array([np.bincount(H[g][lab == x]).argmax() for x in u])       # track most members hold after gate g
        sim = M @ M.T; iu = np.triu_indices(len(u), 1)
        merged = maj_after[iu[0]] == maj_after[iu[1]]
        big = (cnt[iu[0]] >= 3) & (cnt[iu[1]] >= 3)
        rows.append((name, g, sim[iu][big], merged[big]))
R = {"tape": f.split("/")[-1]}
for name in ["raw", "river"]:
    s = np.concatenate([r[2] for r in rows if r[0] == name]); m = np.concatenate([r[3] for r in rows if r[0] == name])
    best = None
    for thr in [0.96, 0.97, 0.975, 0.98, 0.985, 0.99]:
        pred = s >= thr; tp = int((pred & m).sum()); fp = int((pred & ~m).sum()); fn = int((~pred & m).sum())
        R[f"{name}_thr{thr}"] = dict(tp=tp, fp=fp, fn=fn)
    R[f"{name}_merged_pairs"] = int(m.sum()); R[f"{name}_merged_sim_min"] = float(s[m].min()) if m.any() else None
    R[f"{name}_nonmerged_sim_max"] = float(s[~m].max())
print(json.dumps(R, indent=1))
