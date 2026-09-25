import sys, json, glob, numpy as np
sys.path.insert(0, "/home/user/seiranwind-tech/f50_canon_bridge/w11_river")
from w11_lib import river_series, nr, ang

A = np.array([1.1639, 0.5491, -0.7167])

def sigma_of_n(n, sigma0, alpha):
    return sigma0 * np.array(n, float) ** (alpha / 2.0)

def simulate(M, cnt, sigma0, alpha, rho, ensembles, horizons, rng):
    # M: [F, K, D], cnt: [K]
    F, K, D = M.shape
    p = 3
    results = {k: {"sim_disp": [], "true_disp": [], "err_vs_ar": []} for k in horizons}
    maxk = max(horizons)
    for t0 in range(p, F - maxk, 10):
        d0 = [M[t0 - i] - M[t0 - i - 1] for i in range(p)]  # [p][K,D], most recent first
        sig = sigma_of_n(cnt, sigma0, alpha)  # [K]
        # stochastic ensemble
        Mc = np.repeat(M[t0][None], ensembles, axis=0)          # [E,K,D]
        ds = [np.repeat(x[None], ensembles, axis=0) for x in d0]  # each [E,K,D]
        xi_prev = np.zeros((ensembles, K, D))
        # deterministic AR3-only track
        Md = M[t0].copy(); dsd = [x.copy() for x in d0]
        for k in range(1, maxk + 1):
            dn = A[0]*ds[0] + A[1]*ds[1] + A[2]*ds[2]
            eps = rng.standard_normal((ensembles, K, D))
            xi = rho * xi_prev + np.sqrt(max(1 - rho**2, 0)) * sig[None, :, None] * eps
            Mc = nr(Mc + dn + xi); ds = [dn] + ds[:-1]; xi_prev = xi
            dnd = A[0]*dsd[0] + A[1]*dsd[1] + A[2]*dsd[2]
            Md = nr(Md + dnd); dsd = [dnd] + dsd[:-1]
            if k in horizons:
                true_M = M[t0 + k]  # [K,D]
                sim_disp = ang(Mc, true_M[None]).reshape(-1)  # not used directly; see below
                results[k]["sim_disp"].append(ang(Mc, M[t0][None]))          # [E,K]
                results[k]["true_disp"].append(ang(true_M, M[t0]))          # [K]
                results[k]["err_vs_ar"].append(dict(
                    sim_err=ang(Mc, Md[None]),                                # [E,K] sim vs deterministic AR3
                    true_err=ang(true_M, Md)))                                # [K] true vs deterministic AR3
    return results
