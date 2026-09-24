"""W5 — birth precursor curves (DEV only, diagnostic). Hidden quantities used only as targets/axes."""
import sys, json, numpy as np
d = {k: v for k, v in np.load(sys.argv[1]).items()}
an = np.sqrt(2 * np.log(d["n"])); ymin = 0.9 - d["G_shape"]; qmin = 0.9 - d["G_birth"]
ok = (d["so"] > 1e-6) & (d["n"] >= 3)
U = (d["yo"] - ymin)[ok] / (d["so"] * an)[ok]
out = {}
qs = np.percentile(U, [5, 25, 50, 75, 95])
out["U_quantiles_5_25_50_75_95"] = qs.round(3).tolist()
# U vs n (does the EVT scaling hold across sizes?)
nb = [(3, 5), (5, 10), (10, 20), (20, 50), (50, 200), (200, 5000)]
out["U_median_by_n"] = {f"{a}-{b}": float(np.median(U[(d['n'][ok] >= a) & (d['n'][ok] < b)])) if ((d['n'][ok] >= a) & (d['n'][ok] < b)).any() else None for a, b in nb}
# linear collapse: (yo - ymin) = beta * so*an
x, y = (d["so"] * an)[ok], (d["yo"] - ymin)[ok]
beta = float((x @ y) / (x @ x)); r = np.corrcoef(x, y)[0, 1]
out["collapse_beta"] = beta; out["collapse_corr"] = float(r)
# better scaling exponent: y = beta * so * n^gamma ?  fit log
m = (x > 0) & (y > 0)
A = np.c_[np.ones(m.sum()), np.log(d["so"][ok][m]), np.log(d["n"][ok][m])]
co, *_ = np.linalg.lstsq(A, np.log(y[m]), rcond=None)
out["loglaw_ymean_minus_ymin = e^a * so^b * n^c"] = co.round(4).tolist()
# time-to-birth: per basin series, G = qmin-0.9 (positive safe), v = -dG
key = d["seed"] * 10**7 + d["basin"]; o = np.lexsort((d["step"], key))
k, st, G = key[o], d["step"][o], (qmin - 0.9)[o]
same = (k[1:] == k[:-1]) & (st[1:] == st[:-1] + 1)
v = np.full(len(G), np.nan); v[1:][same] = -(G[1:] - G[:-1])[same]
# smooth v with EMA per series
vs = v.copy(); a = 0.2
for i in range(1, len(vs)):
    if same[i - 1] and not np.isnan(vs[i - 1]) and not np.isnan(v[i]): vs[i] = a * v[i] + (1 - a) * vs[i - 1]
# actual steps to first G<0 (birth-side crossing) within same series
nxt = np.full(len(G), np.nan); last = np.nan
for i in range(len(G) - 1, -1, -1):
    if i < len(G) - 1 and not same[i]: last = np.nan
    if G[i] < 0: last = st[i]
    nxt[i] = last - st[i] if not np.isnan(last) else np.nan
sel = (G > 0) & (G < 0.02) & (vs > 1e-5) & ~np.isnan(nxt) & (nxt <= 100)
Tp = G[sel] / vs[sel]; Ta = nxt[sel]
out["T_birth_pred_vs_actual"] = dict(n=int(sel.sum()), corr=float(np.corrcoef(Tp, Ta)[0, 1]), corr_log=float(np.corrcoef(np.log1p(Tp), np.log1p(Ta))[0, 1]),
    median_ratio=float(np.median(Tp / np.maximum(Ta, 1))), frac_within_x2=float(np.mean((Tp / np.maximum(Ta, 1) > 0.5) & (Tp / np.maximum(Ta, 1) < 2))))
# pre-birth aligned profile of G, v and U (steps -50..0 before first crossing)
crossing = np.where((G[1:] < 0) & (G[:-1] >= 0) & same)[0] + 1
prof = {}
Uall = np.full(len(G), np.nan); Uo = np.where(ok, (d["yo"] - ymin) / np.maximum(d["so"] * an, 1e-9), np.nan)[o]
for off in [-50, -40, -30, -20, -10, -5, -2, -1, 0]:
    j = crossing + off; good = (j >= 0) & (k[np.clip(j, 0, len(k)-1)] == k[crossing]) & (st[np.clip(j,0,len(k)-1)] == st[crossing] + off)
    jj = j[good]
    prof[off] = dict(G=float(np.nanmedian(G[jj])), v=float(np.nanmedian(vs[jj])), U=float(np.nanmedian(Uo[jj])), R=float(np.median(d["R"][o][jj])), so=float(np.median(d["so"][o][jj])), N=int(good.sum()))
out["aligned_profile"] = prof
print(json.dumps(out, indent=1))
