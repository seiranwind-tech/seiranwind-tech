"""W11 force balance at group level: magnitudes of S (salt), A (attraction), D (actual); cos(S,-A);
fit D ~ c*S_MF(M) (effective mobility) and D ~ c1*S_MF + c2*(Mbar_nbr projection)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E
nr = E.normalize_rows
f, block, perm, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
eng = make_engine(block, perm, seed); dt = eng.config.dt
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]; T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]; h = 12; g0 = 60
home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
for g in range(1, len(G)):
    prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
    s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
    if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
    back = cur == home; away[back] = 0; away[~back] += 1; sw = away >= h; home[sw] = cur[sw]; away[sw] = 0; HOME.append(home.copy())
def gmean(x, inv, K): S = np.zeros((K, x.shape[1]), np.float64); np.add.at(S, inv, x); return S / np.bincount(inv, minlength=K)[:, None]
def tan(v, M): return v - np.sum(v * M, 1, keepdims=True) * M
rows = {k: [] for k in ["S", "A", "Dm", "Smf", "cosSA", "n", "spread"]}
Xs, Ys = [], []
for g in range(g0, len(G) - 1, 3):
    fr = g * 10 + 12; lab = HOME[g]; u, inv = np.unique(lab, return_inverse=True); K = len(u); cnt = np.bincount(inv); keep = cnt >= 5
    z = nr(Z[fr].astype(np.float32)); z1 = nr(Z[fr + 1].astype(np.float32)); M = nr(gmean(z, inv, K))
    inner = eng._fsalt(z); s = 0.5 * (inner - z * np.sum(z * inner, 1, keepdims=True)); a = eng.config.attract_gain * nr(eng._attraction(z))
    zm = M[inv].astype(np.float32); im = eng._fsalt(zm); sm = 0.5 * (im - zm * np.sum(zm * im, 1, keepdims=True))
    S_, A_, Smf = tan(gmean(s, inv, K), M)[keep], tan(gmean(a, inv, K), M)[keep], tan(gmean(sm, inv, K), M)[keep]
    Dm = tan(gmean(z1 - z, inv, K) / (5 * dt), M)[keep]
    rows["S"] += np.linalg.norm(S_, axis=1).tolist(); rows["A"] += np.linalg.norm(A_, axis=1).tolist(); rows["Dm"] += np.linalg.norm(Dm, axis=1).tolist()
    rows["Smf"] += np.linalg.norm(Smf, axis=1).tolist(); rows["cosSA"] += (np.sum(S_ * -A_, 1) / (np.linalg.norm(S_, axis=1) * np.linalg.norm(A_, axis=1))).tolist()
    rows["n"] += cnt[keep].tolist(); rows["spread"] += (1 - gmean(np.sum(z * M[inv], 1, keepdims=True), inv, K)[keep, 0]).tolist()
    Xs.append(Smf); Ys.append(Dm)
X = np.concatenate(Xs); Y = np.concatenate(Ys); c = float(np.sum(X * Y) / np.sum(X * X))
R = {k: float(np.median(v)) for k, v in rows.items()}
R["fit_D_eq_c_SMF"] = dict(c=c, R2=float(1 - np.sum((Y - c * X) ** 2) / np.sum(Y ** 2)), cos_median=float(np.median(np.sum(X * Y, 1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1)))))
# per-group mobility: does c depend on n / spread?
ci = np.sum(X * Y, 1) / np.sum(X * X, 1); n = np.array(rows["n"]); sp = np.array(rows["spread"])
R["c_per_group_quantiles"] = np.quantile(ci, [0.1, 0.5, 0.9]).tolist(); R["corr_c_logn"] = float(np.corrcoef(ci, np.log(n))[0, 1]); R["corr_c_spread"] = float(np.corrcoef(ci, sp)[0, 1])
print(json.dumps(R, indent=1))
