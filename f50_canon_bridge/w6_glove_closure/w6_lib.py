"""W6 GloVe L1.5 closure audit — shared library. Hidden columns (G_birth, G_shape) are targets/diagnostics only."""
import sys, os, json, hashlib, numpy as np
_R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(_R, "w4"), os.path.join(_R, "w2"), os.path.join(_R, "w3")]
from cross_align import windows, first_cross, window_agree, dt_stats, row_metrics

NBINS = [(3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, 10**9)]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def load(p): return {k: v for k, v in np.load(p).items()}
def nbin_label(a, b): return f"{a}-{b}" if b < 10**9 else f">{a-1}"

def tail(d):
    """L_tail = ybar_own - y_min (y_min hidden, from G_shape)."""
    return d["yo"] - (0.9 - d["G_shape"])

def fit_powerlaw(d, m):
    L = tail(d); m = m & (L > 0) & (d["so"] > 1e-7) & (d["n"] >= 3)
    A = np.c_[np.ones(m.sum()), np.log(d["so"][m]), np.log(d["n"][m])]
    co, *_ = np.linalg.lstsq(A, np.log(L[m]), rcond=None)
    # constrained law L = a*so*n^gamma  (so exponent fixed to 1)
    A2 = np.c_[np.ones(m.sum()), np.log(d["n"][m])]
    c2, *_ = np.linalg.lstsq(A2, np.log(L[m] / d["so"][m]), rcond=None)
    return dict(n=int(m.sum()), free_loga=float(co[0]), free_so_exp=float(co[1]), free_gamma=float(co[2]),
                a_at_so1=float(np.exp(c2[0])), gamma_at_so1=float(c2[1]))

def compare_laws(d, m):
    L = tail(d); m = m & (L > 0) & (d["so"] > 1e-7) & (d["n"] >= 3)
    x1 = (d["so"] * np.sqrt(2 * np.log(d["n"])))[m]
    g = fit_powerlaw(d, m)["gamma_at_so1"]; x2 = (d["so"] * d["n"] ** g)[m]; y = L[m]
    out = {}
    for name, x in [("sqrt2lnn", x1), ("n_gamma", x2)]:
        a = float(x @ y / (x @ x)); r = y - a * x
        out[name] = dict(a=a, rmse=float(np.sqrt(np.mean(r ** 2))), corr=float(np.corrcoef(x, y)[0, 1]),
                         log_resid_sd=float(np.std(np.log(y / (a * x)))))
    return out, g

def B_predict(fam, d):
    x = lambda g: d["so"] * d["n"] ** g
    t = fam["type"]
    if t == "phys":   # 0.9 - yo + beta*so*n^gamma
        return 0.9 - d["yo"] + fam["beta"] * x(fam["gamma"])
    if t == "reg":
        c = fam["coef"]; return c[0] + c[1] * d["yo"] + c[2] * x(fam["gamma"])
    if t == "phys_rescue":
        return 0.9 - d["yo"] + fam["beta"] * x(fam["gamma"]) + fam["lam"] * rescue_scalar(fam["rescue"], d, fam["gamma"])
    raise ValueError(t)

def rescue_scalar(name, d, g):
    if name == "rival_gap": return d["yb"] - d["yo"]            # legal first-moment rival proximity (<=0 usually)
    if name == "cob": return d["cob"]                            # legal centre-centre cosine
    if name == "sb_tail": return d["sb"] * d["n"] ** g           # projected rival spread (augmentation family)
    if name == "rho": return d["rho"]                            # projected own/rival correlation
    raise ValueError(name)

def series(d):
    key = d["seed"].astype(np.int64) * 10**7 + d["basin"]; o = np.lexsort((d["step"], key))
    k, s = key[o], d["step"][o]
    same = np.r_[False, (k[1:] == k[:-1]) & (s[1:] == s[:-1] + 1)]   # same[i]: row i continues row i-1
    return o, k, s, same

def past_diff(x, same, w):
    """x[t]-x[t-w] using only past rows of the same contiguous series; nan if unavailable. Returns per-step rate."""
    out = np.full(len(x), np.nan)
    run = np.zeros(len(x), int)
    for i in range(1, len(x)): run[i] = run[i - 1] + 1 if same[i] else 0
    ok = run >= w; idx = np.where(ok)[0]
    out[idx] = (x[idx] - x[idx - w]) / w
    return out

def steps_to_birth(G, s, same):
    """hidden: steps until G>0 first occurs in the same contiguous series (evaluator only)."""
    nxt = np.full(len(G), np.nan); last = np.nan
    for i in range(len(G) - 1, -1, -1):
        if i + 1 < len(G) and not same[i + 1]: last = np.nan
        if G[i] > 0: last = s[i]
        nxt[i] = last - s[i] if not np.isnan(last) else np.nan
    return nxt

def timing_eval(d, fam, w, horizon=100):
    o, k, s, same = series(d)
    G = B_predict(fam, d)[o]; Gt = d["G_birth"][o]
    yo, so, n = d["yo"][o], d["so"][o], d["n"][o].astype(float)
    g, b = fam["gamma"], fam["beta"]
    ydot = past_diff(yo, same, w); sdot = past_diff(so, same, w)
    Gdot = -ydot + b * n ** g * sdot                            # n constant within interval
    T = steps_to_birth(Gt, s, same)
    sel = (G < 0) & (Gdot > 1e-7) & (Gt <= 0) & ~np.isnan(T) & (T <= horizon) & (T >= 1)
    That = -G[sel] / Gdot[sel]; Ta = T[sel]; r = That / Ta
    err = np.abs(That - Ta)
    cal = {}
    for lo, hi in [(1, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]:
        mm = (That >= lo) & (That < hi)
        cal[f"{lo}-{hi}"] = dict(n=int(mm.sum()), median_actual=float(np.median(Ta[mm])) if mm.any() else None)
    return dict(window=w, n=int(sel.sum()), median_ratio=float(np.median(r)) if sel.any() else None,
                median_abs_err=float(np.median(err)) if sel.any() else None, p90_abs_err=float(np.percentile(err, 90)) if sel.any() else None,
                within_x2=float(np.mean((r > 0.5) & (r < 2))) if sel.any() else None,
                median_abs_logratio=float(np.median(np.abs(np.log(r)))) if sel.any() else None, calibration=cal)

def event_metrics(d, F):
    ph = d["phase"]; o, st = windows(d)
    tb = first_cross(d["G_birth"] > 0, ph, o, st); tq = first_cross(F > 0, ph, o, st)
    r49 = ph == 49
    res = dict(redraw=row_metrics(d["G_birth"][r49], F[r49]), windows=window_agree(tb, tq), dt=dt_stats(tb, tq))
    res["per_seed"] = {int(sd): row_metrics(d["G_birth"][r49 & (d["seed"] == sd)], F[r49 & (d["seed"] == sd)]) for sd in np.unique(d["seed"])}
    res["per_nbin"] = {}
    for a, b in NBINS:
        mm = r49 & (d["n"] >= a) & (d["n"] <= b)
        if mm.sum() > 20 and (d["G_birth"][mm] > 0).any():
            res["per_nbin"][nbin_label(a, b)] = dict(rows=int(mm.sum()), events=int((d["G_birth"][mm] > 0).sum()), **row_metrics(d["G_birth"][mm], F[mm]))
    return res

def residual_audit(d, F):
    """Classify redraw-step errors by the hidden own-tail vs joint (own/best) margins."""
    r = d["phase"] == 49; Gb, Gs = d["G_birth"][r], d["G_shape"][r]; f = F[r]
    fp = (f > 0) & (Gb <= 0); fn = (f <= 0) & (Gb > 0)
    return dict(FP=int(fp.sum()), FN=int(fn.sum()),
                FP_rescued_by_rival=int((fp & (Gs > 0)).sum()),      # own tail really crossed 0.9 but best centre saved it
                FP_own_tail_overestimated=int((fp & (Gs <= 0)).sum()),
                FN_own_tail_crossed_underpredicted=int((fn & (Gs > 0)).sum()),
                all_redraw_rescue_rate=float(((Gs > 0) & (Gb <= 0)).sum() / max((Gs > 0).sum(), 1)))

def fiber(d, keys, bins=5, mask=None):
    m = np.ones(len(d["G_birth"]), bool) if mask is None else mask
    cell = np.zeros(m.sum(), np.int64)
    for k in keys:
        v = d[k][m]; q = np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]); cell = cell * bins + np.searchsorted(q, v)
    u, inv = np.unique(cell, return_inverse=True)
    pos = np.bincount(inv, d["G_birth"][m] > 0); tot = np.bincount(inv); p = pos / tot; mixed = (p > 0) & (p < 1)
    return dict(mixed_frac=float(tot[mixed].sum() / tot.sum()), cells=int(len(u)))
