"""W11 mean-field closure test. For river groups (h=12) at sampled frames f:
  EXACT  G_a   = mean_{w in a} F_w(Z_f)            F_w(z) = 0.5*tangent salt_w(z) + 0.3*nr(attr_w(z))   (wave negligible, measured)
  MF     G^MF_a= mean_{w in a} F_w(M-field)        every member placed at its group centre M_a (population = point masses)
  ACTUAL D_a   = mean_w (Z_{f+1}-Z_f) / (5 dt)
Reports cos / R2 between the three, tangent to M_a. Engine used only as a static function library (teeth, skew, bandwidth)."""
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
def F(z):
    inner = eng._fsalt(z); salt = 0.5 * (inner - z * np.sum(z * inner, 1, keepdims=True))
    return salt, eng.config.attract_gain * nr(eng._attraction(z))
def gmean(x, inv, K): S = np.zeros((K, x.shape[1]), np.float64); np.add.at(S, inv, x); return S / np.bincount(inv, minlength=K)[:, None]
def tan(v, M): return v - np.sum(v * M, 1, keepdims=True) * M
acc = {k: [] for k in ["cos_exact_actual", "cos_mf_exact", "cos_mf_actual", "cos_mfsalt_salt", "cos_mfattr_attr"]}; sq = {k: [0.0, 0.0] for k in ["exact_vs_actual", "mf_vs_actual", "mf_vs_exact", "salt_mf", "attr_mf"]}
eng.interaction_stage = getattr(eng, "interaction_stage")
for g in range(g0, len(G) - 1, 3):
    fr = g * 10 + 12                                         # a frame between gates (partition fixed)
    lab = HOME[g]; u, inv = np.unique(lab, return_inverse=True); K = len(u); cnt = np.bincount(inv)
    z = nr(Z[fr].astype(np.float32)); z1 = nr(Z[fr + 1].astype(np.float32))
    M = nr(gmean(z, inv, K)); keep = cnt >= 5
    s_e, a_e = F(z); Ge = gmean(s_e + a_e, inv, K)
    zm = M[inv].astype(np.float32); s_m, a_m = F(zm); Gm = gmean(s_m + a_m, inv, K)
    Da = gmean((z1 - z), inv, K) / (5 * dt)
    Ge, Gm, Da = tan(Ge, M)[keep], tan(Gm, M)[keep], tan(Da, M)[keep]
    Se, Sm = tan(gmean(s_e, inv, K), M)[keep], tan(gmean(s_m, inv, K), M)[keep]; Ae, Am = tan(gmean(a_e, inv, K), M)[keep], tan(gmean(a_m, inv, K), M)[keep]
    c = lambda a, b: np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    acc["cos_exact_actual"] += c(Ge, Da).tolist(); acc["cos_mf_exact"] += c(Gm, Ge).tolist(); acc["cos_mf_actual"] += c(Gm, Da).tolist()
    acc["cos_mfsalt_salt"] += c(Sm, Se).tolist(); acc["cos_mfattr_attr"] += c(Am, Ae).tolist()
    for k, (p, y) in {"exact_vs_actual": (Ge, Da), "mf_vs_actual": (Gm, Da), "mf_vs_exact": (Gm, Ge), "salt_mf": (Sm, Se), "attr_mf": (Am, Ae)}.items():
        sq[k][0] += float(np.sum((p - y) ** 2)); sq[k][1] += float(np.sum(y ** 2))
R = {"tape": f.split("/")[-1], "dt": dt}
R.update({k: float(np.median(v)) for k, v in acc.items()}); R.update({"R2_" + k: 1 - a / b for k, (a, b) in sq.items()})
print(json.dumps(R, indent=1))
