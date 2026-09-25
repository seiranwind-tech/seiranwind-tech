"""W11: universal isotropic AR(p) law for river-group centre increments.
d_{t+1} = sum_i a_i d_{t-i}  (d = M_{t+1}-M_t, 5-step frames, coefficients shared by ALL groups and tapes)
Fit on train tape, rollout test on other tapes: horizons 1..50 frames (5..250 steps), within fixed-partition blocks."""
import sys, json, numpy as np
def nr(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
def ang(a, b): return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))
def river_series(f, h=12, B=10, g0=60, minsize=5):
    D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]; T, N = trk.shape
    G = np.arange(49, T, 50); H = trk[G]
    home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
    for g in range(1, len(G)):
        prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
        s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
        if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
        back = cur == home; away[back] = 0; away[~back] += 1; sw = away >= h; home[sw] = cur[sw]; away[sw] = 0; HOME.append(home.copy())
    out = []
    for gA in range(g0, len(G) - B + 1, B):
        u, inv = np.unique(HOME[gA], return_inverse=True); cnt = np.bincount(inv)
        fr = np.arange(gA * 10 + 9, min((gA + B) * 10 + 9, len(Z)))
        M = np.zeros((len(fr), len(u), Z.shape[2]), np.float32)
        for i, x in enumerate(fr): np.add.at(M[i], inv, Z[x].astype(np.float32))
        M = nr(M)[:, cnt >= minsize]; out.append((M, cnt[cnt >= minsize]))
    return out
