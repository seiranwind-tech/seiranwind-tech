import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w12a_internal_pull"))
from w12a_common import make_engine, river_home, gmean, nr, ang

S = os.environ["SCR"]
f = os.path.join(S, "tapes", "tape_0_1357_11.npz")
eng = make_engine(0, 1357, 11); dt = eng.config.dt
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]
G, HOME = river_home(trk, h=12)
gA = 60
lab = HOME[gA]; u, inv = np.unique(lab, return_inverse=True); Kfull = len(u)
cnt = np.bincount(inv); keep = np.where(cnt >= 10)[0]
print("Kfull", Kfull, "n cores(n>=10)", len(keep), cnt[keep][:10])
fr0 = gA * 10 + 9
frs = np.arange(fr0, min(fr0 + 100, len(Z)))
Ms = np.stack([nr(gmean(Z[x].astype(np.float32), inv, Kfull)) for x in frs])
t0 = 3
M0 = Ms[t0]
truth5 = Ms[t0 + 1]
disp5 = ang(M0[keep], truth5[keep])
print("disp5 median", np.median(disp5))

def saltMF(M, inv, K):
    zm = M[inv].astype(np.float32)
    inner = eng._fsalt(zm)
    s = 0.5 * (inner - zm * np.sum(zm * inner, 1, keepdims=True))
    return gmean(s, inv, K)

Mc = M0.copy()
Mc = nr(Mc + 5 * dt * saltMF(Mc, inv, Kfull))
err5 = ang(Mc[keep], truth5[keep])
print("err5 (L1 saltMF) median", np.median(err5), "mean", np.mean(err5))
print("skill", 1 - np.mean(err5)/np.mean(disp5))
