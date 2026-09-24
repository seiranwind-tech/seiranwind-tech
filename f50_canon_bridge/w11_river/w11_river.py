"""W11 river coarse-graining (causal). A well keeps its HOME group while it wanders;
home switches only after the well has been away h consecutive gates (hysteresis).
Merged tracks are carried to their successor. Compares raw L1.5 (track) vs macro (home) layer:
  events per gate, K, centre roughness, and autonomy of macro centre motion."""
import sys, json, numpy as np
f = sys.argv[1]; hs = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 8, 12, 20]
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]; T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]; g0 = 60                        # analysis from gate 60 (step 3000) on
def nr(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
def successor(g):   # map of tracks present at g-1 but absent at g -> track most of their wells hold at g
    prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist()); m = {}
    for t in np.unique(prev).tolist():
        if t not in alive: m[t] = int(np.bincount(cur[prev == t]).argmax())
    return m
SUCC = [None] + [successor(g) for g in range(1, len(G))]
def centres_of(lab, frame):   # observed centre of each group from zin frame
    z = Z[frame].astype(np.float32); u, inv = np.unique(lab, return_inverse=True)
    S = np.zeros((len(u), z.shape[1]), np.float32); np.add.at(S, inv, z); return u, nr(S)
out = {"tape": f.split("/")[-1]}
def roughness(labels_at_frame):
    """per 5-step frame: angle moved by each group centre (deg); jump = angle when membership changed"""
    ang, prev = [], None
    for fr in range(g0 * 10, len(Z)):
        lab = labels_at_frame(fr); u, C = centres_of(lab, fr); d = dict(zip(u.tolist(), C))
        if prev is not None:
            common = [k for k in d if k in prev]
            if common:
                a = np.degrees(np.arccos(np.clip(np.sum(np.array([d[k] for k in common]) * np.array([prev[k] for k in common]), 1), -1, 1)))
                ang.append(a)
        prev = d
    a = np.concatenate(ang); return dict(mean_deg=float(a.mean()), p99_deg=float(np.quantile(a, 0.99)), max_deg=float(a.max()))
# raw layer: track labels hold between gates (labels change only at gates)
raw_lab = lambda fr: trk[fr * 5 + 4]
out["raw"] = dict(events_per_gate=float((H[g0:] != H[g0 - 1:-1]).sum(1).mean()),
                  K_mean=float(np.mean([len(np.unique(h)) for h in H[g0:]])), centre_roughness=roughness(raw_lab))
for h in hs:
    home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]; ev = []
    for g in range(1, len(G)):
        s = SUCC[g]
        if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
        # a home that is not alive at g and has no successor (all members wandering) keeps its id
        cur = H[g]; back = cur == home; away[back] = 0; away[~back] += 1
        sw = away >= h; nsw = int(sw.sum()); home[sw] = cur[sw]; away[sw] = 0
        HOME.append(home.copy()); ev.append(nsw)
    HOME = np.array(HOME)
    home_lab = lambda fr: HOME[min((fr * 5 + 4) // 50, len(G) - 1)] if (fr * 5 + 4) >= 49 else trk[fr * 5 + 4]
    wander = float(np.mean(HOME[g0:] != H[g0:]))
    out[f"river_h{h}"] = dict(events_per_gate=float(np.mean(ev[g0 - 1:])), K_mean=float(np.mean([len(np.unique(x)) for x in HOME[g0:]])),
                              frac_wells_wandering=wander, centre_roughness=roughness(home_lab))
print(json.dumps(out, indent=1))
