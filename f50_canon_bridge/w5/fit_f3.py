"""W5 F3 family (preregistered here, fit on DEV only, then frozen).
F3_POWER_1P : Gamma_hat = 0.9 - yo + beta * so * n**gamma          (gamma, beta from DEV log/lin fit)
F3_POWER_REG: Gamma_hat = a + b*yo + c*so*n**gamma + e*sb*n**gamma  (OLS on DEV, gamma fixed from DEV)
Threshold 0. Hidden quantities (G_birth, G_shape) used only as DEV targets."""
import json, hashlib, numpy as np
d = {k: v for k, v in np.load("w1/DEV_table.npz").items()}
ok = (d["so"] > 1e-6) & (d["n"] >= 3)
ymin = 0.9 - d["G_shape"]; y = (d["yo"] - ymin)
m = ok & (y > 0)
A = np.c_[np.ones(m.sum()), np.log(d["so"][m]), np.log(d["n"][m])]
co, *_ = np.linalg.lstsq(A, np.log(y[m]), rcond=None); gamma = float(co[2])
x = d["so"] * d["n"] ** gamma
beta_shape = float((x[ok] @ y[ok]) / (x[ok] @ x[ok]))
# 1-parameter law targeted at G_birth: choose beta minimising squared error of Gamma_hat vs G_birth
r = d["G_birth"] - (0.9 - d["yo"]); beta_birth = float((x @ r) / (x @ x))
X = np.c_[np.ones(len(x)), d["yo"], x, d["sb"] * d["n"] ** gamma]
reg, *_ = np.linalg.lstsq(X, d["G_birth"], rcond=None)
freeze = dict(families={
    "F3_POWER_1P": dict(eq="0.9 - yo + beta*so*n**gamma", gamma=gamma, beta=beta_birth, beta_shape_fit=beta_shape),
    "F3_POWER_REG": dict(eq="a + b*yo + c*so*n**gamma + e*sb*n**gamma", gamma=gamma, coef=reg.tolist())},
    threshold=0.0, dev_table_sha=hashlib.sha256(open("w1/DEV_table.npz", "rb").read()).hexdigest(),
    loglaw=co.tolist())
json.dump(freeze, open("w5/F3_FREEZE.json", "w"), indent=1); print(json.dumps(freeze, indent=1))
