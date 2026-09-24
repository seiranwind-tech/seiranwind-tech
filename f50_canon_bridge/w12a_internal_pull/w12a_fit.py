"""W12A: fit the A_a evolution law.
Model (frame-to-frame, 5 engine steps per frame):
    M+ = nr(M + 5*dt*S_MF(M) + A)
    A+ = rho*A + beta*Env(M) + gamma*S_MF(M)
where Env(M) is the tangent point-mass attraction pull toward the OTHER cores (softmax weighted by
n_b, engine bandwidth), and S_MF is the closed salt mean field. Fit is isotropic (one scalar per
regressor, shared by all 300 dims and all cores/tapes), by pooled least squares (normal equations
accumulated tape-by-tape so only one tape is ever resident).
Usage: python3 w12a_fit.py out.json tape1:block:perm:seed [tape2:block:perm:seed ...]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w12a_common import make_engine, river_home, make_saltMF, env_pointmass, blocks, nr

out_path = sys.argv[1]
specs = sys.argv[2:]

XtX = np.zeros((3, 3), np.float64)
XtY = np.zeros(3, np.float64)
n_samples = 0
per_tape = {}

for spec in specs:
    f, block, perm, seed = spec.split(":")
    block, perm, seed = int(block), int(perm), int(seed)
    eng = make_engine(block, perm, seed); dt = eng.config.dt
    saltMF = make_saltMF(eng)
    D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]
    G, HOME = river_home(trk, h=12)
    tape_n = 0
    for Ms, inv, Kfull, cnt, keep in blocks(G, HOME, Z, g0=60, B=10, minsize=5):
        F = Ms.shape[0]
        Smf_full = np.stack([saltMF(Ms[t], inv, Kfull) for t in range(F)])      # [F,Kfull,d]
        Smf = Smf_full[:, keep, :]; Mk = Ms[:, keep, :]; cntk = cnt[keep]; K = keep.sum()
        A = np.zeros((F - 1, K, Ms.shape[2]), np.float64)
        for t in range(F - 1):
            A[t] = (Mk[t + 1] - Mk[t]) - 5 * dt * Smf[t]
        Env = np.stack([env_pointmass(Mk[t], cntk, eng) for t in range(F - 1)])  # [F-1,K,d]
        # regressors at time t predict A[t] (t=0..F-2); target is A[t+1] i.e. A[1:]
        Xr = np.stack([A[:-1].reshape(-1), Env[:-1].reshape(-1), Smf[:-2].reshape(-1)], -1)
        Yr = A[1:].reshape(-1)
        XtX += Xr.T @ Xr; XtY += Xr.T @ Yr; n_samples += Xr.shape[0]
        tape_n += Xr.shape[0]
    per_tape[os.path.basename(f)] = tape_n
    del D, Z, trk
    print(f"done {f}: {tape_n} samples so far total={n_samples}", file=sys.stderr)

def solve_subset(cols):
    idx = np.array(cols)
    sub = np.linalg.solve(XtX[np.ix_(idx, idx)] + 1e-6 * np.eye(len(idx)), XtY[idx])
    full = np.zeros(3); full[idx] = sub
    return full.tolist()

names = {"AR1": [0], "ENV": [1], "BAL": [2], "AR1+ENV": [0, 1], "AR1+BAL": [0, 2], "ENV+BAL": [1, 2], "AR1+ENV+BAL": [0, 1, 2]}
subsets = {k: solve_subset(v) for k, v in names.items()}
coef = np.array(subsets["AR1+ENV+BAL"])
res = dict(coef_rho_beta_gamma=coef.tolist(), subsets=subsets, n_samples=n_samples, per_tape=per_tape, specs=specs)
with open(out_path, "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
