"""W6-C/D/E on a sealed held-out split. Reads frozen CANDIDATE_FREEZE.json / TIMING_FREEZE.json only; no refitting."""
import sys, json, numpy as np
sys.path.insert(0, "."); from w6_lib import *
split, tab = sys.argv[1], sys.argv[2]
lock = dict(l.split()[::-1] for l in open("FREEZE_LOCK.txt"))
for f in ["CANDIDATE_FREEZE.json", "TIMING_FREEZE.json"]:
    assert sha(f) == lock[f], f"{f} changed after freeze"
cf = json.load(open("CANDIDATE_FREEZE.json")); tf = json.load(open("TIMING_FREEZE.json"))
d = load(tab); fams = cf["families"]
out = dict(split=split, table_sha=sha(tab), freeze_sha={k: lock[k] for k in lock}, rows=int(len(d["n"])),
           redraw_rows=int((d["phase"] == 49).sum()), redraw_events=int((d["G_birth"][d["phase"] == 49] > 0).sum()))
out["families"] = {}
for k, f in fams.items():
    F = B_predict(f, d)
    out["families"][k] = dict(events=event_metrics(d, F), residual=residual_audit(d, F))
out["timing"] = timing_eval(d, fams[tf["family"]], tf["selected_window"])
out["fiber"] = {"(yo,so,n)": fiber(d, ["yo", "so", "n"], mask=d["phase"] == 49)}
if "B3_GLOVE_RESCUE" in fams:
    d["_resc"] = rescue_scalar(fams["B3_GLOVE_RESCUE"]["rescue"], d, fams["B3_GLOVE_RESCUE"]["gamma"])
    out["fiber"]["(yo,so,n,rescue)"] = fiber(d, ["yo", "so", "n", "_resc"], mask=d["phase"] == 49)
# tail-law check on held-out (does the DEV gamma describe held-out tails?)
g = fams["B1_GLOVE_REFIT"]["gamma"]; out["tail_law_heldout"] = fit_powerlaw(d, np.ones(len(d["n"]), bool))
json.dump(out, open(f"{split}_RESULTS.json", "w"), indent=1)
for k, v in out["families"].items():
    e = v["events"]; r = e["redraw"]; w = e["windows"]; t = e["dt"]
    print(f"{k:26s} redraw P {r['precision']:.3f} R {r['recall']:.3f} sign {r['sign_agree']:.3f} corr {r['corr']:.3f} | win P {w['precision']:.3f} R {w['recall']:.3f} | med|dt| {t.get('median_abs')} p90 {t.get('p90_abs')} le10 {t.get('frac_le10',0):.2f} | FP {v['residual']['FP']} (rescue {v['residual']['FP_rescued_by_rival']}) FN {v['residual']['FN']}")
print("timing", {k: out["timing"][k] for k in ["window", "n", "median_ratio", "median_abs_err", "p90_abs_err", "within_x2"]})
print("fiber", out["fiber"], "tail gamma heldout", round(out["tail_law_heldout"]["gamma_at_so1"], 4))
