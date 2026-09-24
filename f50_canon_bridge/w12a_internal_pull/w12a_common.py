"""Shared helpers for W12A: A_a evolution law fitting/eval.
Reuses the exact river-partition and mean-field-salt conventions of w11_rollout.py / w11_ar.py.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E
nr = E.normalize_rows


def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))


def river_home(trk, h=12):
    G = np.arange(49, trk.shape[0], 50)
    H = trk[G]
    N = trk.shape[1]
    home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
    for g in range(1, len(G)):
        prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
        s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
        if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
        back = cur == home; away[back] = 0; away[~back] += 1
        sw = away >= h; home[sw] = cur[sw]; away[sw] = 0
        HOME.append(home.copy())
    return G, HOME


def gmean(x, inv, K):
    S = np.zeros((K, x.shape[1]), np.float64)
    np.add.at(S, inv, x)
    return S / np.bincount(inv, minlength=K)[:, None]


def make_saltMF(eng):
    def saltMF(M, inv, K):
        zm = M[inv].astype(np.float32)
        inner = eng._fsalt(zm)
        s = 0.5 * (inner - zm * np.sum(zm * inner, 1, keepdims=True))
        return gmean(s, inv, K)
    return saltMF


def env_pointmass(Mc, cnt, eng):
    """Tangent attraction pull of each core centre toward the OTHER cores, treated as point masses
    (M_b, n_b), weighted by the engine's own softmax kernel (bandwidth eng.bandwidth), self excluded.
    Halo is not included (dropped) -- kept cheap; see PROTOCOL for the caveat."""
    bw = eng.bandwidth; gain = eng.config.attract_gain
    sim = Mc @ Mc.T
    np.fill_diagonal(sim, -1e30)
    logw = np.log(np.maximum(cnt, 1).astype(np.float64))
    L = sim / bw + logw[None, :]
    L -= L.max(1, keepdims=True)
    w = np.exp(L); w /= w.sum(1, keepdims=True) + 1e-12
    raw = gain * nr(w @ Mc - Mc)
    tan = raw - np.sum(raw * Mc, 1, keepdims=True) * Mc
    return tan


def blocks(G, HOME, Z, g0=60, B=10, minsize=5):
    """Yield (Ms_full[F,Kfull,d], inv[N], Kfull, cnt_full[Kfull], keep[Kfull]) per fixed-partition
    block, late regime only. Ms_full/inv/Kfull cover ALL river groups (needed so eng._fsalt sees the
    real per-well teeth via zm = M[inv]); keep marks which groups are cores (n>=minsize)."""
    for gA in range(g0, len(G) - B + 1, B):
        lab = HOME[gA]; u, inv = np.unique(lab, return_inverse=True); K = len(u)
        cnt = np.bincount(inv); keep = cnt >= minsize
        if keep.sum() < 2:
            continue
        fr0 = gA * 10 + 9; frs = np.arange(fr0, min(fr0 + 10 * B, len(Z)))
        Ms = np.stack([nr(gmean(Z[x].astype(np.float32), inv, K)) for x in frs])  # [F,K,d]
        yield Ms, inv, K, cnt, keep
