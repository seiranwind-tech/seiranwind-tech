"""W11 autonomy of macro ('river', h gates hysteresis) centre motion.
Horizon test from frame f: predict M(f+k) using only macro state at f (and past macro frames).
 P0 hold; P1 const velocity (velocity = mean over last v frames); P2 const velocity + const turn (rotate in the (M,V) plane).
Skill = 1 - err/disp, where disp = angle actually moved. Also the same for raw track centres."""
import sys, json, numpy as np
f = sys.argv[1]; h = int(sys.argv[2]) if len(sys.argv) > 2 else 12
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"].astype(np.float32); T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]; g0 = 60
def nr(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
def succ(g):
    prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
    return {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
for g in range(1, len(G)):
    s = succ(g)
    if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
    back = H[g] == home; away[back] = 0; away[~back] += 1; sw = away >= h; home[sw] = H[g][sw]; away[sw] = 0; HOME.append(home.copy())
HOME = np.array(HOME)
def group_series(labfn, gA, gB):
    """for gate interval [gA,gB): the partition fixed at gate gA; centres of each group for all frames in the interval (+ history 10 frames)"""
    lab = labfn(gA); u, inv = np.unique(lab, return_inverse=True); cnt = np.bincount(inv)
    fr = np.arange(gA * 10 + 10 - 10, gB * 10 + 10)          # frames: gate gA is frame gA*10+9; include 10 frames history
    M = np.zeros((len(fr), len(u), Z.shape[2]), np.float32)
    for i, x in enumerate(fr): np.add.at(M[i], inv, Z[x])
    return u, cnt, nr(M)
def ang(a, b): return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))
res = {}
for name, labfn in [("raw", lambda g: H[g]), (f"river_h{h}", lambda g: HOME[g])]:
    # horizons k frames (5 steps each) within a fixed-partition block of B gates -> no membership change inside block
    for B in [1, 5]:
        E = {k: {"disp": [], "P0": [], "P1": [], "P2": [], "w": []} for k in [1, 10, 10 * B] if k <= 10 * B}
        for gA in range(g0, len(G) - B, B):
            u, cnt, M = group_series(labfn, gA, gA + B); big = cnt >= 5
            if not big.any(): continue
            M = M[:, big]; o = 10                       # index of frame at gate gA (start)
            V = nr(M[o] - M[o - 10]) ; vang = np.radians(ang(M[o], M[o - 10])) / 10          # per-frame speed, direction
            Vp = nr(M[o - 5] - M[o - 10 - 0 if o - 15 < 0 else o - 15]) if o >= 15 else None
            for k in E:
                if o + k >= len(M): continue
                tgt = M[o + k]; d = ang(M[o], tgt)
                # P1: move along great circle through M[o] in direction V by speed*k
                Vt = nr(V - np.sum(V * M[o], -1, keepdims=True) * M[o]); th = (vang * k)[:, None]
                p1 = nr(np.cos(th) * M[o] + np.sin(th) * Vt)
                # P2: velocity from last 5 frames only (fresher)
                V5 = M[o] - M[o - 5]; V5t = nr(V5 - np.sum(V5 * M[o], -1, keepdims=True) * M[o]); th5 = (np.radians(ang(M[o], M[o - 5])) / 5 * k)[:, None]
                p2 = nr(np.cos(th5) * M[o] + np.sin(th5) * V5t)
                E[k]["disp"].append(d); E[k]["P0"].append(d); E[k]["P1"].append(ang(p1, tgt)); E[k]["P2"].append(ang(p2, tgt)); E[k]["w"].append(cnt[big])
        for k, e in E.items():
            if not e["disp"]: continue
            w = np.concatenate(e["w"]).astype(float); disp = np.concatenate(e["disp"])
            r = {"steps": 5 * k, "disp_mean_deg": float(np.average(disp, weights=w))}
            for p in ["P1", "P2"]:
                err = np.concatenate(e[p]); r[p + "_err_deg"] = float(np.average(err, weights=w)); r[p + "_skill"] = float(1 - np.average(err, weights=w) / np.average(disp, weights=w))
            res[f"{name}_block{B}_k{k}"] = r
print(json.dumps(res, indent=1))
