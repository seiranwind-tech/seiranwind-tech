"""W12A: rollout evaluation of one or more A_a laws vs the AR3 baseline, on one or more tapes.
Model (matches w11_rollout.py conventions -- ALL river groups co-evolve via salt-MF so that
eng._fsalt sees the correct per-well teeth; only CORE groups (n>=5) carry a latent A, all other
groups get A=0, i.e. pure salt-MF advection, same as the template's L1):
    M+ = nr(M + 5*dt*S_MF(M) + A)                       (A=0 off cores)
    A+ = rho*A + beta*Env(M_core) + gamma*S_MF(M_core)   (cores only; Env = other-core point-mass pull)
Usage: python3 w12a_eval.py "name1:rho1,beta1,gamma1;name2:rho2,beta2,gamma2;..." \
           tape:block:perm:seed [tape:block:perm:seed ...] [--stride N] [--maxcores K]
The AR3 baseline and the tape load are shared across all named laws.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w12a_common import make_engine, river_home, make_saltMF, env_pointmass, blocks, nr, ang

laws_arg = sys.argv[1]
args = sys.argv[2:]
stride = 1; maxcores = None; specs = []
i = 0
while i < len(args):
    if args[i] == "--stride": stride = int(args[i + 1]); i += 2
    elif args[i] == "--maxcores": maxcores = int(args[i + 1]); i += 2
    else: specs.append(args[i]); i += 1

LAWS = {}
for chunk in laws_arg.split(";"):
    name, coefs = chunk.split(":")
    LAWS[name] = tuple(float(x) for x in coefs.split(","))

AR3 = [1.1639, 0.5491, -0.7167]
Hs = [1, 10, 50]
rng = np.random.default_rng(0)
out = {}

for spec in specs:
    f, block, perm, seed = spec.split(":")
    block, perm, seed = int(block), int(perm), int(seed)
    eng = make_engine(block, perm, seed); dt = eng.config.dt
    saltMF = make_saltMF(eng)
    D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]
    G, HOME = river_home(trk, h=12)
    res = {m: {k: [[], []] for k in Hs} for m in (["L0_AR3"] + list(LAWS.keys()))}
    bidx = 0
    for Ms, inv, Kfull, cnt, keep in blocks(G, HOME, Z, g0=60, B=10, minsize=5):
        bidx += 1
        if stride > 1 and (bidx - 1) % stride != 0:
            continue
        if maxcores is not None and keep.sum() > maxcores:
            idxk = np.where(keep)[0]
            drop = rng.choice(idxk, size=len(idxk) - maxcores, replace=False)
            keep = keep.copy(); keep[drop] = False
        F = Ms.shape[0]
        Mk_all = Ms[:, keep, :]
        for t0 in range(3, F - 50, 10):
            M0k = Mk_all[t0]; truth = {k: Mk_all[t0 + k] for k in Hs}; disp = {k: ang(M0k, truth[k]) for k in Hs}
            # --- L0: universal AR3 baseline (core-only state, matches w11_rollout.py) ---
            Mc = M0k.copy(); ds = [Mk_all[t0 - i2] - Mk_all[t0 - i2 - 1] for i2 in range(3)]
            for k in range(1, 51):
                dn = sum(AR3[i2] * ds[i2] for i2 in range(3)); Mc = nr(Mc + dn); ds = [dn] + ds[:-1]
                if k in Hs:
                    res["L0_AR3"][k][0].append(ang(Mc, truth[k])); res["L0_AR3"][k][1].append(disp[k])
            # --- named A_a laws: full-partition salt-MF advection + latent A on cores ---
            A0 = (Ms[t0][keep] - Ms[t0 - 1][keep]) - 5 * dt * saltMF(Ms[t0 - 1], inv, Kfull)[keep]
            for name, (rho, beta, gamma) in LAWS.items():
                Mc_full = Ms[t0].copy()
                A_full = np.zeros_like(Mc_full); A_full[keep] = A0
                for k in range(1, 51):
                    s_mf_full = saltMF(Mc_full, inv, Kfull)
                    Mc_full = nr(Mc_full + 5 * dt * s_mf_full + A_full)
                    env = env_pointmass(Mc_full[keep], cnt[keep], eng)
                    A_full[keep] = rho * A_full[keep] + beta * env + gamma * s_mf_full[keep]
                    if k in Hs:
                        Mck = Mc_full[keep]
                        res[name][k][0].append(ang(Mck, truth[k])); res[name][k][1].append(disp[k])

    R = {"tape": os.path.basename(f), "laws": LAWS}
    for m, v in res.items():
        R[m] = {f"{5*k}steps": dict(err=float(np.mean(np.concatenate(e))), disp=float(np.mean(np.concatenate(d))),
                                     skill=float(1 - np.mean(np.concatenate(e)) / np.mean(np.concatenate(d)))) for k, (e, d) in v.items()}
    out[os.path.basename(f)] = R
    print(json.dumps(R, indent=1)); sys.stdout.flush()
    del D, Z, trk

print(json.dumps(out, indent=1))
