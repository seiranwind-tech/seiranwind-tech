"""W6 DEV stage: A (tail law), B (freeze B0/B1/B2 [+B3 single rescue scalar]), D (timing window). DEV blocks 0-3 only."""
import sys, json, numpy as np
sys.path.insert(0, "."); from w6_lib import *
d = load("data/DEV_table.npz"); ALL = np.ones(len(d["n"]), bool)
# ---------- W6-A ----------
A = {}
A["overall"] = fit_powerlaw(d, ALL); A["law_compare"], g = compare_laws(d, ALL)
A["per_seed"] = {int(s): fit_powerlaw(d, d["seed"] == s) for s in np.unique(d["seed"])}
A["per_nbin"] = {}
for a, b in NBINS:
    m = (d["n"] >= a) & (d["n"] <= b)
    if m.sum() > 200: A["per_nbin"][nbin_label(a, b)] = fit_powerlaw(d, m) | {"rows": int(m.sum())}
t1, t2 = np.quantile(d["step"], [1 / 3, 2 / 3])
A["per_time"] = {"early": fit_powerlaw(d, d["step"] <= t1), "middle": fit_powerlaw(d, (d["step"] > t1) & (d["step"] <= t2)), "late": fit_powerlaw(d, d["step"] > t2)}
L = tail(d); ok = (L > 0) & (d["so"] > 1e-7) & (d["n"] >= 3)
U = np.where(ok, L / np.maximum(d["so"] * d["n"] ** g, 1e-12), np.nan)
U2 = np.where(ok, L / np.maximum(d["so"] * np.sqrt(2 * np.log(np.maximum(d["n"], 2))), 1e-12), np.nan)
A["U_gamma_median_by_nbin"] = {nbin_label(a, b): float(np.nanmedian(U[(d["n"] >= a) & (d["n"] <= b)])) for a, b in NBINS if ((d["n"] >= a) & (d["n"] <= b) & ok).sum() > 50}
A["U_sqrt2lnn_median_by_nbin"] = {nbin_label(a, b): float(np.nanmedian(U2[(d["n"] >= a) & (d["n"] <= b)])) for a, b in NBINS if ((d["n"] >= a) & (d["n"] <= b) & ok).sum() > 50}
def cv(dct): v = np.array(list(dct.values())); return float(v.std() / v.mean())
A["collapse_cv_across_nbins"] = {"U_gamma": cv(A["U_gamma_median_by_nbin"]), "U_sqrt2lnn": cv(A["U_sqrt2lnn_median_by_nbin"])}
A["gamma_spaCy_W5"] = 0.3447280636528175; A["gamma_GloVe"] = g
A["gamma_in_0.33_0.36"] = bool(0.33 <= g <= 0.36)
A["dev_table_sha"] = sha("data/DEV_table.npz")
json.dump(A, open("TAIL_LAW_DEV.json", "w"), indent=1)
# ---------- W6-B ----------
x = d["so"] * d["n"] ** g; r = d["G_birth"] - (0.9 - d["yo"])
beta = float(x @ r / (x @ x))
X = np.c_[np.ones(len(x)), d["yo"], x]; c, *_ = np.linalg.lstsq(X, d["G_birth"], rcond=None)
fams = {"B0_DIRECT_TRANSFER_spaCy": dict(type="phys", beta=0.706525245438733, gamma=0.3447280636528175, note="W5 spaCy F3_POWER_1P, not refit"),
        "B1_GLOVE_REFIT": dict(type="phys", beta=beta, gamma=g),
        "B2_GLOVE_REG": dict(type="reg", gamma=g, coef=c.tolist())}
# DEV residual audit + single rescue scalar selection (fit blocks 0-2, validate block 3)
audit = {k: residual_audit(d, B_predict(f, d)) for k, f in fams.items()}
tr, va = d["seed"] <= 2, d["seed"] == 3
sub = lambda m: {k: v[m] for k, v in d.items()}
dtr, dva = sub(tr), sub(va)
cand = {}
for name in ["rival_gap", "cob", "sb_tail", "rho"]:
    Z = np.c_[dtr["so"] * dtr["n"] ** g, rescue_scalar(name, dtr, g)]
    co, *_ = np.linalg.lstsq(Z, dtr["G_birth"] - (0.9 - dtr["yo"]), rcond=None)
    fam = dict(type="phys_rescue", beta=float(co[0]), gamma=g, lam=float(co[1]), rescue=name)
    ra = residual_audit(dva, B_predict(fam, dva))
    cand[name] = dict(lam=float(co[1]), beta=float(co[0]), val_errors=ra["FP"] + ra["FN"], val=ra)
base_va = residual_audit(dva, B_predict(dict(type="phys", beta=float((dtr["so"] * dtr["n"] ** g) @ (dtr["G_birth"] - 0.9 + dtr["yo"]) / np.sum((dtr["so"] * dtr["n"] ** g) ** 2)), gamma=g), dva))
best = min(cand, key=lambda k: cand[k]["val_errors"])
selection = dict(baseline_B1_val_errors=base_va["FP"] + base_va["FN"], candidates=cand, selected=best,
                 selected_improves=cand[best]["val_errors"] < base_va["FP"] + base_va["FN"])
if selection["selected_improves"]:
    Z = np.c_[x, rescue_scalar(best, d, g)]; co, *_ = np.linalg.lstsq(Z, r, rcond=None)
    fams["B3_GLOVE_RESCUE"] = dict(type="phys_rescue", beta=float(co[0]), gamma=g, lam=float(co[1]), rescue=best)
dev_metrics = {k: event_metrics(d, B_predict(f, d))["redraw"] for k, f in fams.items()}
fibers = {"(yo,so,n)": fiber(d, ["yo", "so", "n"], mask=d["phase"] == 49)}
if "B3_GLOVE_RESCUE" in fams:
    d["_resc"] = rescue_scalar(best, d, g); fibers[f"(yo,so,n,{best})"] = fiber(d, ["yo", "so", "n", "_resc"], mask=d["phase"] == 49); del d["_resc"]
json.dump(dict(families=fams, threshold=0.0, dev_table_sha=sha("data/DEV_table.npz"), dev_redraw_metrics=dev_metrics,
               dev_residual_audit=audit, rescue_selection=selection, dev_fiber=fibers), open("CANDIDATE_FREEZE.json", "w"), indent=1)
# ---------- W6-D ----------
tim = {w: timing_eval(d, fams["B1_GLOVE_REFIT"], w) for w in [1, 3, 5, 10]}
wbest = min(tim, key=lambda w: tim[w]["median_abs_logratio"])
json.dump(dict(family="B1_GLOVE_REFIT", selected_window=wbest, dev=tim, rule="Gdot=-dybar/dt + beta*n^gamma*dsigma/dt (past-only, n const); predict T=-Gamma/Gdot when Gamma<0,Gdot>0"),
          open("TIMING_FREEZE.json", "w"), indent=1)
print(json.dumps(dict(gamma=g, law_compare=A["law_compare"], collapse=A["collapse_cv_across_nbins"], per_seed_gamma={k: round(v["gamma_at_so1"], 4) for k, v in A["per_seed"].items()},
      per_nbin_gamma={k: round(v["gamma_at_so1"], 4) for k, v in A["per_nbin"].items()}, per_time={k: round(v["gamma_at_so1"], 4) for k, v in A["per_time"].items()},
      U_gamma=A["U_gamma_median_by_nbin"], U_sqrt=A["U_sqrt2lnn_median_by_nbin"], fams=fams, dev_metrics=dev_metrics, audit=audit, rescue=dict((k, v["val_errors"]) for k, v in cand.items()) | {"baseline": selection["baseline_B1_val_errors"], "selected": best}, fiber=fibers,
      timing={w: (t["median_ratio"], t["within_x2"], t["median_abs_logratio"]) for w, t in tim.items()}, wbest=wbest), indent=1, default=str))
