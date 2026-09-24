"""W11: is river-group centre motion driven by the autonomous zout clock?
Engine: zout evolves ONLY from zout (blade + salt(zout), wall clip) -> exogenous per-well drive.
wave summed over a 5-step frame telescopes: sum dt*0.6*(zout_t - zop_t)/wall = dt*0.12*(ZO[f+1]-ZO[f]).
Test per group a: D_a = mean_w (Z[f+1]-Z[f])  vs  W_a = dt*0.12 * mean_w (ZO[f+1]-ZO[f]).
Then rollouts: M_{f+1} = nr(M_f + beta*W_a(f) + residual law), residual law = 0 or AR on residual."""
import sys, json, numpy as np
def nr(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
def ang(a, b): return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
f = sys.argv[1]; h = 12; B = 10
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]; ZO = D["ZO"]; T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]; g0 = 60
home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
for g in range(1, len(G)):
    prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
    s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
    if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
    back = cur == home; away[back] = 0; away[~back] += 1; sw = away >= h; home[sw] = cur[sw]; away[sw] = 0; HOME.append(home.copy())
blocks = []
for gA in range(g0, len(G) - B + 1, B):
    u, inv = np.unique(HOME[gA], return_inverse=True); cnt = np.bincount(inv); keep = cnt >= 5
    fr = np.arange(gA * 10 + 9, min((gA + B) * 10 + 9, len(Z)))
    SZ = np.zeros((len(fr), len(u), Z.shape[2]), np.float32); SO = np.zeros_like(SZ)
    for i, x in enumerate(fr): np.add.at(SZ[i], inv, Z[x].astype(np.float32)); np.add.at(SO[i], inv, ZO[x].astype(np.float32))
    SZ /= cnt[None, :, None]; SO /= cnt[None, :, None]
    blocks.append((SZ[:, keep], SO[:, keep], cnt[keep]))
# 1) explained fraction
num = den = 0.0; cs = []; betas_x = []; betas_y = []
for SZ, SO, c in blocks:
    Dm = np.diff(SZ, axis=0); W = DT * 0.12 * np.diff(SO, axis=0)
    betas_x.append(W.reshape(-1)); betas_y.append(Dm.reshape(-1))
    cs.append(np.sum(Dm * W, -1) / (np.linalg.norm(Dm, axis=-1) * np.linalg.norm(W, axis=-1) + 1e-12))
x = np.concatenate(betas_x); y = np.concatenate(betas_y); beta = float(x @ y / (x @ x))
R = {"tape": f.split("/")[-1], "dt": DT, "beta_fit": beta, "R2_wave_only_beta1": float(1 - np.sum((y - x) ** 2) / np.sum(y ** 2)),
     "R2_wave_only_betafit": float(1 - np.sum((y - beta * x) ** 2) / np.sum(y ** 2)), "cos_D_W_median": float(np.median(np.concatenate([c.ravel() for c in cs])))}
# 2) rollouts with exogenous drive: M_{k+1} = nr(M_k + b*W_k + a*(prev residual))
res_x, res_y = [], []
for SZ, SO, c in blocks:
    Dm = np.diff(SZ, axis=0); W = DT * 0.12 * np.diff(SO, axis=0); r = Dm - beta * W
    res_x.append(r[:-1].reshape(-1)); res_y.append(r[1:].reshape(-1))
rx = np.concatenate(res_x); ry = np.concatenate(res_y); a1 = float(rx @ ry / (rx @ rx)); R["residual_AR1_coef"] = a1
R["residual_frac_of_D"] = float(np.sum((y - beta * x) ** 2) / np.sum(y ** 2))
for mode in ["hold", "drive", "drive+resAR1"]:
    E = {k: ([], []) for k in [1, 10, 50]}
    for SZ, SO, c in blocks:
        M = nr(SZ); W = DT * 0.12 * np.diff(SO, axis=0)
        for t0 in range(1, len(M) - 50, 10):
            Mc = SZ[t0].copy(); rprev = (SZ[t0] - SZ[t0 - 1]) - beta * DT * 0.12 * (SO[t0] - SO[t0 - 1])
            for k in range(1, 51):
                if mode == "hold": step = 0
                elif mode == "drive": step = beta * W[t0 + k - 1]
                else: rprev = a1 * rprev; step = beta * W[t0 + k - 1] + rprev
                Mc = Mc + step
                if k in E: E[k][0].append(ang(nr(Mc), M[t0 + k])); E[k][1].append(ang(M[t0], M[t0 + k]))
    R[mode] = {f"{5*k}steps": dict(err=float(np.mean(np.concatenate(E[k][0]))), disp=float(np.mean(np.concatenate(E[k][1]))),
               skill=float(1 - np.mean(np.concatenate(E[k][0])) / np.mean(np.concatenate(E[k][1])))) for k in E}
print(json.dumps(R, indent=1))
