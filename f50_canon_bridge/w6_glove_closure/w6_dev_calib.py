"""DEV-only addendum (declared before any held-out read): B1c = B1 form with beta calibrated at threshold 0
by minimising redraw-step FP+FN. Grid chosen on blocks 0-2, checked on block 3, final beta from all DEV."""
import sys, json, numpy as np
sys.path.insert(0, "."); from w6_lib import *
d = load("data/DEV_table.npz"); cf = json.load(open("CANDIDATE_FREEZE.json"))
g = cf["families"]["B1_GLOVE_REFIT"]["gamma"]; r = d["phase"] == 49
x = (d["so"] * d["n"] ** g); base = 0.9 - d["yo"]; G = d["G_birth"]
grid = np.linspace(0.2, 0.8, 241)
def err(m, b): f = base[m] + b * x[m]; return int(((f > 0) != (G[m] > 0)).sum())
tr, va = r & (d["seed"] <= 2), r & (d["seed"] == 3)
b_tr = float(grid[np.argmin([err(tr, b) for b in grid])])
b_all = float(grid[np.argmin([err(r, b) for b in grid])])
cf["families"]["B1c_GLOVE_THRESH_CAL"] = dict(type="phys", beta=b_all, gamma=g,
    note="beta calibrated at threshold 0 on DEV redraw rows (FP+FN), DEV-only", beta_train_blocks012=b_tr,
    val_block3_errors=err(va, b_tr), val_block3_errors_B1=err(va, cf["families"]["B1_GLOVE_REFIT"]["beta"]),
    val_block3_errors_B0=int(((0.9 - d["yo"][va] + 0.706525245438733 * (d["so"] * d["n"] ** 0.3447280636528175)[va] > 0) != (G[va] > 0)).sum()))
cf["dev_redraw_metrics"]["B1c_GLOVE_THRESH_CAL"] = event_metrics(d, B_predict(cf["families"]["B1c_GLOVE_THRESH_CAL"], d))["redraw"]
cf["dev_residual_audit"]["B1c_GLOVE_THRESH_CAL"] = residual_audit(d, B_predict(cf["families"]["B1c_GLOVE_THRESH_CAL"], d))
json.dump(cf, open("CANDIDATE_FREEZE.json", "w"), indent=1)
print(json.dumps(cf["families"]["B1c_GLOVE_THRESH_CAL"], indent=1), cf["dev_redraw_metrics"]["B1c_GLOVE_THRESH_CAL"])
