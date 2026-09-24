"""W12B: one-shot prospective evaluation. Frozen coefficients only (copied verbatim from
PROTOCOL_W12B.md) -- no refit, no retuning. Evaluates on the two prospective tapes' pair/found rows."""
import json, numpy as np
import w12b_fit_eval as m

# --- frozen coefficients (verbatim from PROTOCOL_W12B.md, fit on the 3 discovery tapes) ---
beta_pair_full = np.array([-6.666873839249263, 3.1154287382879398, -0.19302865198682742,
                            0.3426163685962038, -1.9604929327574852, 0.2780582468019149])
beta_pair_const = np.array([-6.693530729010572])
beta_pair_nn = np.array([-8.411610903211939, 4.947461842965106])
beta_found_full = np.array([-4.811817608442928, 0.22058266098410076, 0.39798275112628034])
beta_found_const = np.array([-3.998979949446851])

PASS_AUC_MIN = 0.90
PASS_TOP1_MARGIN = 0.05

PTAPES = {
    "tape_2_8080_32_zo": "work/pairs_p2_8080_32.npz",
    "tape_3_4242_31_zo": "work/pairs_p3_4242_31.npz",
}

results = {"frozen_coefficients": {
    "beta_pair_full": beta_pair_full.tolist(),
    "beta_pair_full_names": ["intercept", "sim", "log_n_a", "log_n_b", "rank", "spread_a"],
    "beta_pair_const": beta_pair_const.tolist(),
    "beta_pair_nn": beta_pair_nn.tolist(),
    "beta_found_full": beta_found_full.tolist(),
    "beta_found_const": beta_found_const.tolist(),
}, "pass_criterion": {
    "criterion_1": "loglik(full) > loglik(const) AND loglik(full) > loglik(nn), pooled and per-tape",
    "criterion_2": f"AUC(full) >= {PASS_AUC_MIN} on each prospective tape",
    "criterion_3": f"dest_top1_acc(full) >= dest_top1_acc(nn) - {PASS_TOP1_MARGIN} on each prospective tape",
}, "per_tape": {}, "verdict": {}}

all_pair_rows, all_found_rows = [], []
overall_pass = True
for name, path in PTAPES.items():
    d = np.load(path)
    pr, fr = d["pair_rows"], d["found_rows"]
    all_pair_rows.append(pr); all_found_rows.append(fr)
    pair_eval = m.evaluate_pair(beta_pair_full, beta_pair_const, beta_pair_nn, pr, name)
    found_eval = m.evaluate_found(beta_found_full, beta_found_const, fr, name)

    c1 = (pair_eval["loglik_full"] > pair_eval["loglik_const"]) and (pair_eval["loglik_full"] > pair_eval["loglik_nn"])
    c2 = (pair_eval["auc_full"] is not None) and (pair_eval["auc_full"] >= PASS_AUC_MIN)
    top1_full = pair_eval["dest_top1_acc_full"]; top1_nn = pair_eval["dest_top1_acc_nn"]
    c3 = (top1_full is None) or (top1_nn is None) or (top1_full >= top1_nn - PASS_TOP1_MARGIN)
    tape_pass = bool(c1 and c2 and c3)
    overall_pass = overall_pass and tape_pass

    results["per_tape"][name] = {
        "pair_metrics": pair_eval,
        "found_metrics": found_eval,
        "criterion_1_loglik_beats_baselines": bool(c1),
        "criterion_2_auc_ge_threshold": bool(c2),
        "criterion_3_top1_within_margin_of_nn": bool(c3),
        "tape_pass": tape_pass,
    }

pooled_pair = np.concatenate(all_pair_rows, axis=0)
pooled_found = np.concatenate(all_found_rows, axis=0)
pooled_pair_eval = m.evaluate_pair(beta_pair_full, beta_pair_const, beta_pair_nn, pooled_pair, "POOLED_prospective")
pooled_found_eval = m.evaluate_found(beta_found_full, beta_found_const, pooled_found, "POOLED_prospective")
c1_pool = (pooled_pair_eval["loglik_full"] > pooled_pair_eval["loglik_const"]) and (pooled_pair_eval["loglik_full"] > pooled_pair_eval["loglik_nn"])
results["pooled"] = {
    "pair_metrics": pooled_pair_eval,
    "found_metrics": pooled_found_eval,
    "criterion_1_loglik_beats_baselines_pooled": bool(c1_pool),
}

results["verdict"] = {
    "overall_pass": bool(overall_pass and c1_pool),
    "note": "overall_pass requires both prospective tapes to individually pass criteria 1-3, and the pooled log-likelihood to beat both baselines.",
}

print(json.dumps(results, indent=1))
