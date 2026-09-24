"""W12B: build per-gate pair data (boundary-transfer flux) + per-gate founding counts,
from a single discovery/prospective tape, using the river HOME partition (h=12),
exactly as in w11_river.py / w11_macro_events.py (read-only, tape-replay).

Output: one .npz with arrays for pair-rows (gate, a, b, sim, n_a, n_b, rank, spread_a, target)
and founding-rows (gate, a, n_a, spread_a, target=#foundings from a at that gate).
"""
import sys, json, numpy as np

def nr(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def build(f, h=12, g0=60):
    D = np.load(f, allow_pickle=True, mmap_mode="r")
    trk = D["trk"]; Z = D["Z"]; T, N = trk.shape
    G = np.arange(49, T, 50); H = trk[G]
    # --- replicate w11_river.py's HOME construction (causal, hysteresis h) ---
    def successor(g):
        prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist()); m = {}
        for t in np.unique(prev).tolist():
            if t not in alive:
                m[t] = int(np.bincount(cur[prev == t]).argmax())
        return m
    SUCC = [None] + [successor(g) for g in range(1, len(G))]
    home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
    EV = []  # (gate, well, a=old home, b=new home) recorded at the gate the switch happens
    for g in range(1, len(G)):
        s = SUCC[g]
        if s:
            home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
        cur = H[g]; back = cur == home; away[back] = 0; away[~back] += 1
        sw = np.where(away >= h)[0]
        for w in sw:
            EV.append((g, int(w), int(home[w]), int(cur[w])))
        home[sw] = cur[sw]; away[sw] = 0
        HOME.append(home.copy())
    HOME = np.array(HOME)

    first_seen = {}
    for g in range(len(G)):
        for t in np.unique(H[g]).tolist():
            first_seen.setdefault(t, g)

    late_EV = [e for e in EV if e[0] >= g0]
    # split: to an existing HOME[g-1] group vs a founding (wanderer's own birth track)
    to_new = set()
    for e in late_EV:
        g, w, a, b = e
        if first_seen[b] > g - h - 1:
            to_new.add(e)

    pair_rows = []   # gate, a, b, sim, n_a, n_b, rank, spread_a, target
    found_rows = []  # gate, a, n_a, spread_a, target(#foundings)

    # index late events by gate for fast target lookup
    from collections import defaultdict
    ev_by_gate = defaultdict(list)
    for e in late_EV:
        ev_by_gate[e[0]].append(e)

    for g in range(g0, len(G)):
        lab = HOME[g - 1]
        u, inv = np.unique(lab, return_inverse=True)
        frame = 10 * (g - 1) + 9
        z = Z[frame].astype(np.float32)
        S = np.zeros((len(u), z.shape[1]), np.float32)
        np.add.at(S, inv, z)
        cnt = np.bincount(inv, minlength=len(u))
        M = nr(S)
        # spread: 1 - mean cos to group centre
        cos_to_c = np.sum(nr(z) * M[inv], axis=1)
        spread = np.zeros(len(u), np.float32)
        np.add.at(spread, inv, 1.0 - cos_to_c)
        spread = spread / np.maximum(cnt, 1)

        keep = cnt >= 2
        idx_keep = np.where(keep)[0]
        if len(idx_keep) < 2:
            continue
        Mk = M[idx_keep]; SIM = Mk @ Mk.T
        np.fill_diagonal(SIM, -9)
        order = np.argsort(-SIM, axis=1)
        rankmat = np.empty_like(order)
        for i in range(len(idx_keep)):
            rankmat[i, order[i]] = np.arange(len(idx_keep))

        # targets for this gate: existing-group transfers only (exclude foundings)
        gate_events = ev_by_gate.get(g, [])
        transfer_target = {}  # (a,b) -> count
        found_count = defaultdict(int)
        for e in gate_events:
            gg, w, a, b = e
            if e in to_new:
                found_count[a] += 1
            else:
                transfer_target[(a, b)] = transfer_target.get((a, b), 0) + 1

        uid = u[idx_keep]
        pos = {int(t): i for i, t in enumerate(uid.tolist())}
        for i, a in enumerate(uid.tolist()):
            n_a = int(cnt[idx_keep[i]]); spread_a = float(spread[idx_keep[i]])
            found_rows.append((g, a, n_a, spread_a, found_count.get(a, 0)))
            for j, b in enumerate(uid.tolist()):
                if i == j:
                    continue
                n_b = int(cnt[idx_keep[j]])
                sim = float(SIM[i, j]); rank = int(rankmat[i, j])
                target = transfer_target.get((a, b), 0)
                pair_rows.append((g, a, b, sim, n_a, n_b, rank, spread_a, target))

    pair_rows = np.array(pair_rows, dtype=np.float64)
    found_rows = np.array(found_rows, dtype=np.float64)
    return pair_rows, found_rows, dict(tape=f.split("/")[-1], h=h, g0=g0,
                                        n_gates=len(G) - g0, n_late_events=len(late_EV),
                                        n_transfers=len(late_EV) - len(to_new), n_foundings=len(to_new))

if __name__ == "__main__":
    f = sys.argv[1]; out = sys.argv[2]
    pair_rows, found_rows, meta = build(f)
    np.savez_compressed(out, pair_rows=pair_rows, found_rows=found_rows)
    meta["pair_rows"] = int(pair_rows.shape[0]); meta["found_rows"] = int(found_rows.shape[0])
    print(json.dumps(meta, indent=1))
