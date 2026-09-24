"""W11: taxonomy of the remaining macro (river) events and their macro-level predictability."""
import sys, json, numpy as np
def nr(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
f = sys.argv[1]; h = int(sys.argv[2]) if len(sys.argv) > 2 else 12
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]; T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]; g0 = 60
home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]; EV = []
for g in range(1, len(G)):
    prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
    s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
    if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
    back = cur == home; away[back] = 0; away[~back] += 1; sw = np.where(away >= h)[0]
    for w in sw: EV.append((g, int(w), int(home[w]), int(cur[w])))
    home[sw] = cur[sw]; away[sw] = 0; HOME.append(home.copy())
HOME = np.array(HOME)
R = {"tape": f.split("/")[-1], "h": h}
late = [e for e in EV if e[0] >= g0]; R["macro_events_late"] = len(late)
first_seen = {}
for g in range(len(G)):
    for t in np.unique(H[g]).tolist(): first_seen.setdefault(t, g)
to_new = [e for e in late if first_seen[e[3]] > e[0] - h - 1]        # destination born within the away period = wanderer's own birth-track
R["to_own_born_track"] = len(to_new); R["to_existing_group"] = len(late) - len(to_new)
# destination group size (in HOME) and centre similarity rank of destination among home's neighbours
ranks, sims, rsims, dsize = [], [], [], []
for g, w, a, b in late:
    lab = HOME[g - 1]; u, inv = np.unique(lab, return_inverse=True)
    S = np.zeros((len(u), Z.shape[2]), np.float32); np.add.at(S, inv, Z[g * 10 + 9 - 10 * h].astype(np.float32)); M = nr(S)
    ia = np.searchsorted(u, a) if a in u else None; ib = np.searchsorted(u, b) if b in u else None
    if ia is None or ib is None or ia >= len(u) or u[ia] != a or ib >= len(u) or u[ib] != b: continue
    sim = M @ M[ia]; sim[ia] = -9; order = np.argsort(-sim); ranks.append(int(np.where(order == ib)[0][0])); sims.append(float(sim[ib]))
    rsims.append(float(np.median(sim[sim > -9]))); dsize.append(int((lab == b).sum()))
R["dest_existing_rank_among_groups_by_centre_sim(0=nearest)"] = dict(n=len(ranks), frac_rank0=float(np.mean(np.array(ranks) == 0)) if ranks else None,
    frac_rank_le2=float(np.mean(np.array(ranks) <= 2)) if ranks else None, sim_dest_median=float(np.median(sims)) if sims else None, sim_typical_median=float(np.median(rsims)) if rsims else None)
# permanent new macro groups (singleton tracks that became home of their well and never dissolved)
R["macro_K_first_last_late"] = [int(len(np.unique(HOME[g0]))), int(len(np.unique(HOME[-1])))]
print(json.dumps(R, indent=1))
