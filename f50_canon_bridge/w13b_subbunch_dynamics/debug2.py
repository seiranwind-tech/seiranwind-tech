import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w12a_internal_pull"))
from w12a_common import make_engine, river_home, gmean, nr, ang

S = os.environ["SCR"]
f = os.path.join(S, "tapes", "tape_0_1357_11.npz")
eng = make_engine(0, 1357, 11); dt = eng.config.dt
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]
G, HOME = river_home(trk, h=12)

def saltMF(M, inv, K):
    zm = M[inv].astype(np.float32)
    inner = eng._fsalt(zm)
    s = 0.5 * (inner - zm * np.sum(zm * inner, 1, keepdims=True))
    return gmean(s, inv, K)

g0, B = 60, 10
E5 = []; D5 = []; E50 = []; D50 = []
nblocks = 0
for gA in range(g0, len(G) - B + 1, B):
    lab = HOME[gA]; u, inv = np.unique(lab, return_inverse=True); Kfull = len(u)
    cnt = np.bincount(inv); keep = cnt >= 5
    if keep.sum() < 2: continue
    nblocks += 1
    if nblocks > 4: break
    fr0 = gA * 10 + 9
    frs = np.arange(fr0, min(fr0 + 10 * B, len(Z)))
    Ms = np.stack([nr(gmean(Z[x].astype(np.float32), inv, Kfull)) for x in frs])
    for t0 in range(3, len(frs) - 50, 10):
        M0 = Ms[t0]; truth5 = Ms[t0 + 1]; truth50 = Ms[t0 + 10]
        Mc = M0.copy()
        for k in range(1, 11):
            Mc = nr(Mc + 5 * dt * saltMF(Mc, inv, Kfull))
            if k == 1:
                E5.append(ang(Mc, truth5)[keep]); D5.append(ang(M0, truth5)[keep])
            if k == 10:
                E50.append(ang(Mc, truth50)[keep]); D50.append(ang(M0, truth50)[keep])

E5 = np.concatenate(E5); D5 = np.concatenate(D5); E50 = np.concatenate(E50); D50 = np.concatenate(D50)
print("n blocks used", nblocks)
print("5step  err", E5.mean(), "disp", D5.mean(), "skill", 1 - E5.mean()/D5.mean())
print("50step err", E50.mean(), "disp", D50.mean(), "skill", 1 - E50.mean()/D50.mean())
