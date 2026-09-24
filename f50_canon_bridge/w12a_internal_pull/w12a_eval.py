"""W12A: rollout evaluation of the A_a law(s) vs the AR3 baseline, on one or more tapes.
Model (matches w11_rollout.py conventions -- ALL river groups co-evolve via salt-MF so that
eng._fsalt sees the correct per-well teeth; only CORE groups (n>=5) carry a latent A, all other
groups get A=0, i.e. pure salt-MF advection, same as the template's L1):
    M+ = nr(M + 5*dt*S_MF(M) + A)                       (A=0 off cores)
    A+ = rho*A + beta*Env(M_core) + gamma*S_MF(M_core)   (cores only; Env = other-core point-mass pull)
Usage: python3 w12a_eval.py coef.json[:subset_name] tape:block:perm:seed [tape:block:perm:seed ...]
  subset_name in {AR1,ENV,BAL,AR1+ENV,AR1+BAL,ENV+BAL,AR1+ENV+BAL}; default AR1+ENV+BAL.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w12a_common import make_engine, river_home, make_saltMF, env_pointmass, blocks, nr, ang

coef_spec = sys.argv[1]
specs = sys.argv[2:]
KNOWN = ("AR1", "ENV", "BAL", "AR1+ENV", "AR1+BAL", "ENV+BAL", "AR1+ENV+BAL")
if ":" in coef_spec and coef_spec.split(":")[-1] in KNOWN:
    path, sub = coef_spec.rsplit(":", 1)
else:
    path, sub = coef_spec, "AR1+ENV+BAL"
C = json.load(open(path))
rho, beta, gamma = C["subsets"][sub] if "subsets" in C else C["coef_rho_beta_gamma"]

AR3 = [1.1639, 0.5491, -0.7167]
Hs = [1, 10, 50]
out = {}

for spec in specs:
    f, block, perm, seed = spec.split(":")
    block, perm, seed = int(block), int(perm), int(seed)
    eng = make_engine(block, perm, seed); dt = eng.config.dt
    saltMF = make_saltMF(eng)
    D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]
    G, HOME = river_home(trk, h=12)
    res = {m: {k: [[], []] for k in Hs} for m in ["L0_AR3", "L_law"]}
    for Ms, inv, Kfull, cnt, keep in blocks(G, HOME, Z, g0=60, B=10, minsize=5):
        F = Ms.shape[0]
        Mk_all = Ms[:, keep, :]  # for AR3 baseline + truth, core-only
        for t0 in range(3, F - 50, 10):
            M0k = Mk_all[t0]; truth = {k: Mk_all[t0 + k] for k in Hs}; disp = {k: ang(M0k, truth[k]) for k in Hs}
            # --- L0: universal AR3 baseline (core-only state, matches w11_rollout.py) ---
            Mc = M0k.copy(); ds = [Mk_all[t0 - i] - Mk_all[t0 - i - 1] for i in range(3)]
            for k in range(1, 51):
                dn = sum(AR3[i] * ds[i] for i in range(3)); Mc = nr(Mc + dn); ds = [dn] + ds[:-1]
                if k in Hs:
                    res["L0_AR3"][k][0].append(ang(Mc, truth[k])); res["L0_AR3"][k][1].append(disp[k])
            # --- L_law: full-partition salt-MF advection + latent A law on cores ---
            Mc_full = Ms[t0].copy()
            A_full = np.zeros_like(Mc_full)
            A_full[keep] = (Ms[t0][keep] - Ms[t0 - 1][keep]) - 5 * dt * saltMF(Ms[t0 - 1], inv, Kfull)[keep]
            for k in range(1, 51):
                s_mf_full = saltMF(Mc_full, inv, Kfull)
                Mc_full = nr(Mc_full + 5 * dt * s_mf_full + A_full)
                env = env_pointmass(Mc_full[keep], cnt[keep], eng)
                A_full[keep] = rho * A_full[keep] + beta * env + gamma * s_mf_full[keep]
                if k in Hs:
                    Mck = Mc_full[keep]
                    res["L_law"][k][0].append(ang(Mck, truth[k])); res["L_law"][k][1].append(disp[k])

    R = {"tape": os.path.basename(f), "coef": [rho, beta, gamma], "subset": sub}
    for m, v in res.items():
        R[m] = {f"{5*k}steps": dict(err=float(np.mean(np.concatenate(e))), disp=float(np.mean(np.concatenate(d))),
                                     skill=float(1 - np.mean(np.concatenate(e)) / np.mean(np.concatenate(d)))) for k, (e, d) in v.items()}
    out[os.path.basename(f)] = R
    print(json.dumps(R, indent=1))
    del D, Z, trk

print(json.dumps(out, indent=1))
