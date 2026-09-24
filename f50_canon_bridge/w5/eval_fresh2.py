import sys, json, numpy as np
sys.path[:0] = ["w2", "w4"]
import bridge_eval
from cross_align import windows, first_cross, window_agree, dt_stats, row_metrics
d = {k: v for k, v in np.load(sys.argv[1]).items()}
f3 = json.load(open("w5/F3_FREEZE.json"))["families"]
g = f3["F3_POWER_1P"]["gamma"]; x = d["so"] * d["n"] ** g
preds = {f: bridge_eval.predict(f, d) for f in ["F1_LEGAL", "F2_AUG"]}
preds["F3_POWER_1P"] = 0.9 - d["yo"] + f3["F3_POWER_1P"]["beta"] * x
c = f3["F3_POWER_REG"]["coef"]; preds["F3_POWER_REG"] = c[0] + c[1] * d["yo"] + c[2] * x + c[3] * d["sb"] * d["n"] ** g
o, st = windows(d); ph = d["phase"]; tb = first_cross(d["G_birth"] > 0, ph, o, st)
out = {}
for k, F in preds.items():
    tq = first_cross(F > 0, ph, o, st)
    out[k] = dict(p49=row_metrics(d["G_birth"][ph == 49], F[ph == 49]), windows=window_agree(tb, tq), dt=dt_stats(tb, tq))
# time-to-birth with legal-state bridge: T = F3/(-dF3 smoothed) evaluated vs actual
key = d["seed"] * 10**7 + d["basin"]; oo = np.lexsort((d["step"], key)); k_, s_ = key[oo], d["step"][oo]
F = -preds["F3_POWER_REG"][oo]; G = -d["G_birth"][oo]    # positive = safe
same = (k_[1:] == k_[:-1]) & (s_[1:] == s_[:-1] + 1)
v = np.full(len(F), np.nan); v[1:][same] = -(F[1:] - F[:-1])[same]
vs = v.copy()
for i in range(1, len(vs)):
    if same[i - 1] and not np.isnan(vs[i - 1]) and not np.isnan(v[i]): vs[i] = 0.2 * v[i] + 0.8 * vs[i - 1]
nxt = np.full(len(G), np.nan); last = np.nan
for i in range(len(G) - 1, -1, -1):
    if i < len(G) - 1 and not same[i]: last = np.nan
    if G[i] < 0: last = s_[i]
    nxt[i] = last - s_[i] if not np.isnan(last) else np.nan
sel = (F > 0) & (F < 0.02) & (vs > 1e-5) & ~np.isnan(nxt) & (nxt <= 100)
r = (F[sel] / vs[sel]) / np.maximum(nxt[sel], 1)
out["T_birth_legal_F3REG"] = dict(n=int(sel.sum()), median_ratio=float(np.median(r)), frac_within_x2=float(np.mean((r > 0.5) & (r < 2))))
print(json.dumps(out, indent=1))
json.dump(out, open("w5/FRESH2_RESULT.json", "w"), indent=1)
