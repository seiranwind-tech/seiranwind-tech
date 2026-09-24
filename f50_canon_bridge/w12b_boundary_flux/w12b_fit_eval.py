"""W12B: fit Poisson rate laws for boundary transfers (pair-level) and foundings (per-a),
compare against baselines, and report held-out metrics.

Usage: python3 w12b_fit_eval.py <fit_npz1> [<fit_npz2> ...] -- <val_npz1> [<val_npz2> ...]
Fits on the union of fit tapes, evaluates on each val tape separately and pooled.
"""
import sys, json, numpy as np

# ---------- IRLS Poisson GLM (log link) ----------
def poisson_glm_fit(Xd, y, n_iter=100, ridge=2.0):
    n, p = Xd.shape
    beta = np.zeros(p)
    for it in range(n_iter):
        eta = Xd @ beta
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        W = mu
        z = eta + (y - mu) / np.maximum(mu, 1e-9)
        XtW = Xd.T * W
        A = XtW @ Xd + ridge * np.eye(p)
        b = XtW @ z
        beta_new = np.linalg.solve(A, b)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new; break
        beta = beta_new
    return beta

def poisson_loglik(y, mu):
    from scipy.special import gammaln
    mu = np.maximum(mu, 1e-12)
    return np.sum(y * np.log(mu) - mu - gammaln(y + 1))

def poisson_deviance(y, mu):
    mu = np.maximum(mu, 1e-12)
    term = np.where(y > 0, y * np.log(y / mu) - (y - mu), mu)
    return 2.0 * np.sum(term)

def auc_score(y_bin, score):
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = y_bin.sum(); n0 = len(y_bin) - n1
    if n1 == 0 or n0 == 0:
        return None
    sum_ranks_pos = ranks[y_bin == 1].sum()
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)

# ---------- feature builders ----------
def pair_features(rows):
    # cols: gate, a, b, sim, n_a, n_b, rank, spread_a, target
    sim = rows[:, 3]; n_a = rows[:, 4]; n_b = rows[:, 5]; rank = rows[:, 6]; spread_a = rows[:, 7]
    X = np.column_stack([np.ones(len(rows)), sim, np.log(n_a), np.log(n_b), rank, spread_a])
    y = rows[:, 8]
    return X, y

def pair_features_nn(rows):
    # nearest-neighbour-only feature set: intercept + is_rank0
    rank = rows[:, 6]
    X = np.column_stack([np.ones(len(rows)), (rank == 0).astype(np.float64)])
    y = rows[:, 8]
    return X, y

def found_features(rows):
    # cols: gate, a, n_a, spread_a, target
    n_a = rows[:, 2]; spread_a = rows[:, 3]
    X = np.column_stack([np.ones(len(rows)), np.log(n_a), spread_a])
    y = rows[:, 4]
    return X, y

def load(paths):
    prs, frs = [], []
    for p in paths:
        d = np.load(p)
        prs.append(d["pair_rows"]); frs.append(d["found_rows"])
    return np.concatenate(prs, axis=0), np.concatenate(frs, axis=0)

def evaluate_pair(beta_full, beta_const, beta_nn, rows, label):
    Xf, y = pair_features(rows)
    Xc = np.column_stack([np.ones(len(rows))])
    Xn, _ = pair_features_nn(rows)
    mu_full = np.exp(np.clip(Xf @ beta_full, -30, 30))
    mu_const = np.exp(np.clip(Xc @ beta_const, -30, 30))
    mu_nn = np.exp(np.clip(Xn @ beta_nn, -30, 30))
    out = {"tape_group": label, "n_pairs": int(len(rows)), "n_events": float(y.sum())}
    out["loglik_full"] = poisson_loglik(y, mu_full)
    out["loglik_const"] = poisson_loglik(y, mu_const)
    out["loglik_nn"] = poisson_loglik(y, mu_nn)
    out["deviance_full"] = poisson_deviance(y, mu_full)
    out["deviance_const"] = poisson_deviance(y, mu_const)
    out["deviance_nn"] = poisson_deviance(y, mu_nn)
    y_bin = (y > 0).astype(int)
    out["auc_full"] = auc_score(y_bin, mu_full)
    out["auc_nn_baseline"] = auc_score(y_bin, mu_nn)
    out["auc_sim_only"] = auc_score(y_bin, rows[:, 3])
    # destination top-1 accuracy: for each (gate,a) with >=1 transfer, does argmax mu over b match?
    gates = rows[:, 0].astype(int); avals = rows[:, 1].astype(int)
    key = gates.astype(np.int64) * 1_000_000 + avals.astype(np.int64)
    uniq_keys, inv = np.unique(key, return_inverse=True)
    n_groups = len(uniq_keys)
    best_full = np.full(n_groups, -1.0); best_full_idx = np.full(n_groups, -1)
    best_nn = np.full(n_groups, -1.0); best_nn_idx = np.full(n_groups, -1)
    has_event = np.zeros(n_groups, dtype=bool); true_idx = np.full(n_groups, -1)
    for i in range(len(rows)):
        g = inv[i]
        if mu_full[i] > best_full[g]:
            best_full[g] = mu_full[i]; best_full_idx[g] = i
        if mu_nn[i] > best_nn[g]:
            best_nn[g] = mu_nn[i]; best_nn_idx[g] = i
        if y[i] > 0:
            has_event[g] = True; true_idx[g] = i
    mask = has_event
    if mask.sum() > 0:
        correct_full = np.sum(best_full_idx[mask] == true_idx[mask])
        correct_nn = np.sum(best_nn_idx[mask] == true_idx[mask])
        out["dest_top1_acc_full"] = float(correct_full / mask.sum())
        out["dest_top1_acc_nn"] = float(correct_nn / mask.sum())
        out["n_pairs_with_event_for_top1"] = int(mask.sum())
    else:
        out["dest_top1_acc_full"] = None; out["dest_top1_acc_nn"] = None
    # calibration: total events per gate, predicted vs actual
    ug = np.unique(gates)
    pred_tot = np.array([mu_full[gates == g].sum() for g in ug])
    act_tot = np.array([y[gates == g].sum() for g in ug])
    out["calib_total_pred_mean"] = float(pred_tot.mean())
    out["calib_total_actual_mean"] = float(act_tot.mean())
    out["calib_corr"] = float(np.corrcoef(pred_tot, act_tot)[0, 1]) if len(ug) > 2 else None
    return out

def evaluate_found(beta_full, beta_const, rows, label):
    Xf, y = found_features(rows)
    Xc = np.column_stack([np.ones(len(rows))])
    mu_full = np.exp(np.clip(Xf @ beta_full, -30, 30))
    mu_const = np.exp(np.clip(Xc @ beta_const, -30, 30))
    out = {"tape_group": label, "n_rows": int(len(rows)), "n_events": float(y.sum())}
    out["loglik_full"] = poisson_loglik(y, mu_full)
    out["loglik_const"] = poisson_loglik(y, mu_const)
    out["deviance_full"] = poisson_deviance(y, mu_full)
    out["deviance_const"] = poisson_deviance(y, mu_const)
    return out

if __name__ == "__main__":
    args = sys.argv[1:]
    sep = args.index("--")
    fit_paths = args[:sep]; val_paths = args[sep + 1:]

    fit_pairs, fit_found = load(fit_paths)
    print(f"# fit pair rows: {len(fit_pairs)}, fit found rows: {len(fit_found)}", file=sys.stderr)

    Xf, yf = pair_features(fit_pairs)
    beta_full = poisson_glm_fit(Xf, yf)
    Xc = np.column_stack([np.ones(len(fit_pairs))])
    beta_const = poisson_glm_fit(Xc, yf)
    Xn, _ = pair_features_nn(fit_pairs)
    beta_nn = poisson_glm_fit(Xn, yf)

    Xff, yff = found_features(fit_found)
    beta_found_full = poisson_glm_fit(Xff, yff)
    Xfc = np.column_stack([np.ones(len(fit_found))])
    beta_found_const = poisson_glm_fit(Xfc, yff)

    results = {
        "beta_pair_full": beta_full.tolist(),
        "beta_pair_full_names": ["intercept", "sim", "log_n_a", "log_n_b", "rank", "spread_a"],
        "beta_pair_const": beta_const.tolist(),
        "beta_pair_nn": beta_nn.tolist(),
        "beta_found_full": beta_found_full.tolist(),
        "beta_found_full_names": ["intercept", "log_n_a", "spread_a"],
        "beta_found_const": beta_found_const.tolist(),
        "fit_tapes": [p.split("/")[-1] for p in fit_paths],
        "val_tapes": [p.split("/")[-1] for p in val_paths],
        "fit_metrics_pair": evaluate_pair(beta_full, beta_const, beta_nn, fit_pairs, "FIT_pooled"),
        "fit_metrics_found": evaluate_found(beta_found_full, beta_found_const, fit_found, "FIT_pooled"),
        "val_metrics_pair_per_tape": [],
        "val_metrics_found_per_tape": [],
    }

    for vp in val_paths:
        d = np.load(vp)
        vr, vf = d["pair_rows"], d["found_rows"]
        results["val_metrics_pair_per_tape"].append(evaluate_pair(beta_full, beta_const, beta_nn, vr, vp.split("/")[-1]))
        results["val_metrics_found_per_tape"].append(evaluate_found(beta_found_full, beta_found_const, vf, vp.split("/")[-1]))

    if val_paths:
        val_pairs, val_found = load(val_paths)
        results["val_metrics_pair_pooled"] = evaluate_pair(beta_full, beta_const, beta_nn, val_pairs, "VAL_pooled")
        results["val_metrics_found_pooled"] = evaluate_found(beta_found_full, beta_found_const, val_found, "VAL_pooled")

    print(json.dumps(results, indent=1))
