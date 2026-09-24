"""W11 replay: tape-level taxonomy of label events and the 'river' coarse-graining.
Part A: raw events per gate; fate of each birth (returns to parent? how fast?).
Part B: per-well excursion analysis: leave track a, come back to a within L gates."""
import sys, json, numpy as np
f = sys.argv[1]; D = np.load(f, allow_pickle=True); trk = D["trk"]; T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]                  # track at each gate (after reassign+merge)
pre = trk[G - 1]                                      # track just before gate
R = {"tape": f.split("/")[-1], "T": int(T), "gates": int(len(G))}
K = np.array([len(np.unique(h)) for h in H]); R["K_first_last"] = [int(K[0]), int(K[-1])]
R["K_every_10_gates"] = K[::10].tolist()
changes = (H != pre).sum(1); R["label_changes_per_gate_mean"] = float(changes.mean())
R["label_changes_per_gate_late(second_half)"] = float(changes[len(G)//2:].mean())
# births: track ids that appear for the first time at a gate
seen = set(np.unique(trk[0]).tolist()); births = []
for g, t in enumerate(G):
    u = np.unique(H[g]); new = [x for x in u.tolist() if x not in seen]
    for x in new:
        m = np.where(H[g] == x)[0]; births.append(dict(g=g, track=x, n=len(m), parent=int(np.bincount(pre[t - 49 if False else g][m] if False else pre[g][m]).argmax()) if False else int(np.bincount(pre[g][m]).argmax()), wells=m.tolist()))
    seen.update(u.tolist())
R["births_total"] = len(births); R["births_second_half"] = int(sum(b["g"] >= len(G)//2 for b in births))
# fate of each birth: when the born track dies, where are its founding wells? back in parent's (surviving) track?
fates = []
for b in births:
    g0 = b["g"]; w = np.array(b["wells"]); alive = [g for g in range(g0, len(G)) if (H[g] == b["track"]).any()]
    life = len(alive)
    gend = g0 + life
    if gend >= len(G): fates.append(dict(life=life, fate="alive_at_end")); continue
    after = H[gend][w]
    # parent track may itself have merged; follow parent's wells' current track
    par_wells = np.where(pre[g0] == b["parent"])[0]; par_wells = np.setdiff1d(par_wells, w)
    par_now = np.bincount(H[gend][par_wells]).argmax() if len(par_wells) else -1
    frac_home = float(np.mean(after == par_now)) if par_now >= 0 else 0.0
    fates.append(dict(life=life, fate="home" if frac_home >= 0.5 else "elsewhere", frac_home=frac_home, n=b["n"]))
lifes = np.array([x["life"] for x in fates]) if fates else np.array([0])
R["birth_fate_counts"] = {k: int(sum(x["fate"] == k for x in fates)) for k in ["home", "elsewhere", "alive_at_end"]}
R["birth_life_gates_quantiles_p50_p90_p99"] = np.quantile(lifes, [0.5, 0.9, 0.99]).tolist()
R["birth_life_eq1_gate_frac"] = float(np.mean(lifes == 1))
R["birth_size_quantiles_p50_p90_max"] = np.quantile([b["n"] for b in births] or [0], [0.5, 0.9, 1.0]).tolist()
# Part B: per-well excursions: well leaves "home" group (defined via parent's wells' track) and returns within L gates
# simpler, label-free: co-membership return. For each well at gate g, companions S = wells sharing its track.
# after an event (well's track changes), does the well share a track with >=50% of S again within L gates?
L = 20; ex = dict(events=0, return_le_L=0, lags=[])
for g in range(len(G) - 1):
    moved = np.where(H[g + 1] != H[g])[0]
    # exclude pure merges (whole group relabelled together): event = well separated from its companions
    for w in moved[:: max(1, len(moved) // 400)]:   # subsample for speed
        S = np.where(H[g] == H[g][w])[0]; S = S[S != w]
        if len(S) == 0: continue
        if np.mean(H[g + 1][S] == H[g + 1][w]) >= 0.5: continue      # moved together with companions -> relabel/merge, not excursion
        ex["events"] += 1
        for l in range(2, L + 1):
            if g + l >= len(G): break
            if np.mean(H[g + l][S] == H[g + l][w]) >= 0.5: ex["return_le_L"] += 1; ex["lags"].append(l - 1); break
R["excursions_subsampled"] = dict(events=ex["events"], returned_within_20_gates=ex["return_le_L"],
     return_frac=ex["return_le_L"] / max(ex["events"], 1), lag_p50_p90=np.quantile(ex["lags"] or [0], [0.5, 0.9]).tolist())
print(json.dumps(R, indent=1))
