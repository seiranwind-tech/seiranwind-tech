import sys, json, glob, numpy as np
sys.path.insert(0, "/home/user/seiranwind-tech/f50_canon_bridge/w11_river")
from w11_lib import river_series, nr, ang

A = np.array([1.1639, 0.5491, -0.7167])

def residuals(files):
    rs, ns, series = [], [], []
    for f in files:
        for M, cnt in river_series(f):
            d = np.diff(M, axis=0)  # [F-1, K, D]
            F1, K, D = d.shape
            for k in range(K):
                dk = d[:, k, :]
                rk = []
                for t in range(2, len(dk) - 1):
                    pred = A[0]*dk[t] + A[1]*dk[t-1] + A[2]*dk[t-2]
                    r = dk[t+1] - pred
                    rk.append(r); rs.append(r); ns.append(cnt[k])
                if len(rk) >= 2:
                    series.append(np.array(rk))
    return np.array(rs), np.array(ns), series

if __name__ == "__main__":
    fit_files = sorted(glob.glob(sys.argv[1] + "/*.npz"))
    rs, ns, series = residuals(fit_files)
    var_per_sample = (rs**2).mean(axis=1)  # mean over D dims, per residual sample
    # power law sigma^2(n) = sigma0^2 * n^alpha, fit by averaging within unique n then log-log LSQ
    uns = np.unique(ns)
    mv = np.array([var_per_sample[ns == u].mean() for u in uns])
    w = np.array([np.sum(ns == u) for u in uns])
    X = np.log(uns); Yv = np.log(mv)
    Wm = np.diag(w)
    Xd = np.stack([np.ones_like(X), X], -1)
    beta = np.linalg.lstsq(Xd * w[:, None]**0.5, Yv * w**0.5, rcond=None)[0]
    log_sigma0sq, alpha = beta
    sigma0 = float(np.sqrt(np.exp(log_sigma0sq)))
    # lag-1 autocorrelation, pooled across series & dims
    num, den = 0.0, 0.0
    for s in series:
        num += np.sum(s[:-1] * s[1:]); den += np.sum(s[:-1]**2)
    rho = float(num / den)
    # angle-per-step reference (at mean n)
    D = rs.shape[1]
    n_ref = float(np.median(ns))
    sigma_ref = sigma0 * n_ref**(alpha/2)
    ang_per_step_deg = float(np.degrees(sigma_ref * np.sqrt(D)))
    out = dict(sigma0=sigma0, alpha=float(alpha), rho=rho, D=int(D), n_ref=n_ref,
               ang_per_step_deg_at_n_ref=ang_per_step_deg, n_samples=int(len(rs)),
               n_series=len(series), coef=A.tolist())
    print(json.dumps(out, indent=1))
