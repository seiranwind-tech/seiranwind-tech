import sys, json, glob, numpy as np
from scipy.stats import ks_2samp
sys.path.insert(0, "/home/user/seiranwind-tech/f50_canon_bridge/w11_river")
from w11_lib import river_series, nr, ang
from w14_sim import simulate, A

def evaluate(tape_files, sigma0, alpha, rho, ensembles=50, horizons=(10, 50), seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for f in tape_files:
        agg = {k: dict(sim_disp=[], true_disp=[], sim_err=[], true_err=[]) for k in horizons}
        for M, cnt in river_series(f):
            if M.shape[1] == 0:
                continue
            r = simulate(M, cnt, sigma0, alpha, rho, ensembles, horizons, rng)
            for k in horizons:
                if not r[k]["sim_disp"]:
                    continue
                agg[k]["sim_disp"].append(np.concatenate([x.ravel() for x in r[k]["sim_disp"]]))
                agg[k]["true_disp"].append(np.concatenate([x.ravel() for x in r[k]["true_disp"]]))
                agg[k]["sim_err"].append(np.concatenate([e["sim_err"].ravel() for e in r[k]["err_vs_ar"]]))
                agg[k]["true_err"].append(np.concatenate([e["true_err"].ravel() for e in r[k]["err_vs_ar"]]))
        res = {}
        for k in horizons:
            if not agg[k]["sim_disp"]:
                continue
            sim_disp = np.concatenate(agg[k]["sim_disp"]); true_disp = np.concatenate(agg[k]["true_disp"])
            sim_err = np.concatenate(agg[k]["sim_err"]); true_err = np.concatenate(agg[k]["true_err"])
            p90_band = np.percentile(sim_err, 90)
            coverage = float(np.mean(true_err <= p90_band))
            ks = ks_2samp(sim_disp, true_disp)
            res[f"{5*k}steps"] = dict(
                sim_disp_mean=float(sim_disp.mean()), true_disp_mean=float(true_disp.mean()),
                sim_disp_p10=float(np.percentile(sim_disp, 10)), sim_disp_p50=float(np.percentile(sim_disp, 50)),
                sim_disp_p90=float(np.percentile(sim_disp, 90)),
                true_disp_p10=float(np.percentile(true_disp, 10)), true_disp_p50=float(np.percentile(true_disp, 50)),
                true_disp_p90=float(np.percentile(true_disp, 90)),
                coverage=coverage, ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue),
                n_sim=int(len(sim_disp)), n_true=int(len(true_disp)))
        out[f.split("/")[-1]] = res
    return out

if __name__ == "__main__":
    params = json.load(open(sys.argv[1]))
    files = sorted(glob.glob(sys.argv[2] + "/*.npz"))
    r = evaluate(files, params["sigma0"], params["alpha"], params["rho"])
    print(json.dumps(r, indent=1))
