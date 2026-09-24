"""Canonical-line (F40 V3.9R3) retrain of the SAME W3 controller
architecture: two-layer recurrent HOLD/SWITCH controller over candidate
predictors of the next-interval basin/track centre movement W. Trains on
DEV seeds (blocks) 0,1,2 and freezes to CONTROLLER_FREEZE.json. Evaluates
on seed (block) 3, writes DEV_controller_trace.npz (full seeds 0-3,
deployed/frozen controller, strictly causal), and prints sha256 of
engine/table/freeze/run-script/trace.

STRICT: only LEGAL aggregates + the 'moves' (realised W) list are read.
G_birth/G_shape/so/sb/rho/events(births)/lineage/threshold are never
touched.

Canonical-line moves-to-redraw_index convention differs from the PU line:
redraw_index = (step-1)//REDRAW (table's own convention; the reassignment
step has phase==49), so moves are mapped with STEP_OFFSET=1 (see
controller_run.build_moves_map's docstring). Verified against
w1/DEV_table.npz's own redraw_index/phase columns (99.85% of a random
sample of rows resolve to a recorded move).
"""
import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controller_run as cr

W1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w1")
TABLE = os.path.join(W1, "DEV_table.npz")
EVENTS = os.path.join(W1, "DEV_events.json")
ENGINE = os.path.join(W1, "canon_recorder.py")  # actual F40 V3.9R3 engine file is external (legacy/V39R3_FROZEN/); its cited sha256 is recorded in the freeze/report as text, this is the present read-only recorder script
FREEZE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CONTROLLER_FREEZE.json")
TRACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DEV_controller_trace.npz")
RUN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controller_run.py")

TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = [3]
STEP_OFFSET = 1  # canonical line: redraw_index = (step-1)//REDRAW; moves' step is the phase==49 table step
CITED_ENGINE_SHA256 = "4a71aeb4a66a8a071a6b6f9a9b065687d237909aea5e89a537ee6d80d2aa3b1d"  # f40_v39r3_engine.py, per OBSERVABLE_CONSTITUTION.md / canon_recorder.py ENG_SHA (engine file itself not present in this bridge dir)

ETA_A = 0.6   # EMA smoothing of candidate A (EMA-of-past-W) across interval boundaries
ETA_M = 0.9   # slow-memory EMA for candidate M (dmu*50 momentum)
ETA_H1 = 0.7  # layer-1 recurrent memory
ETA_H2 = 0.7  # layer-2 recurrent memory
MARGIN = 0.0  # switch-label margin: challenger must strictly beat champion's error
L2 = 1e-3
LR1, ITERS1 = 0.3, 400
LR2, ITERS2 = 0.3, 400


def softmax_batch(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_softmax(X, y, n_classes, iters, lr, l2):
    n, f = X.shape
    Theta = np.zeros((n_classes, f))
    Y = np.eye(n_classes)[y]
    for it in range(iters):
        logits = X @ Theta.T
        P = softmax_batch(logits)
        grad = (P - Y).T @ X / n + l2 * Theta
        Theta -= lr * grad
    return Theta


def main():
    print("loading DEV table/events (moves only) ...")
    data, moves = cr.load_table_events(TABLE, EVENTS)
    mv = cr.build_moves_map(moves, step_offset=STEP_OFFSET)
    n = len(data["seed"])
    print("rows:", n, "moves:", len(moves))

    order_all = cr.proc_order(data)
    tau = cr.tau_of_age(data["age"])
    dtau = cr.compute_dtau(tau, order_all, data)

    # ---- global default (W_prior): mean realised W over train-seed moves ----
    train_W = [m["W"] for m in moves if m["seed"] in TRAIN_SEEDS]
    W_prior = float(np.mean(train_W))
    print("W_prior (train mean W):", W_prior)

    # ---- Pass0: theta-free causal candidate recurrences (all seeds) ----
    A_pred, D_raw, M_pred = cr.compute_pass0_impl(data, order_all, mv, ETA_A, ETA_M, W_prior)

    # ---- label_W: this row's CURRENT interval's realised W (label/eval only, never a controller input) ----
    ri = data["redraw_index"]
    seed = data["seed"]; basin = data["basin"]
    label_W = np.full(n, np.nan)
    for i in range(n):
        w = mv.get((int(seed[i]), int(basin[i]), int(ri[i])))
        if w is not None:
            label_W[i] = w
    has_label = ~np.isnan(label_W)
    print("rows with a valid interval-W label:", has_label.sum())

    train_mask = has_label & np.isin(seed, TRAIN_SEEDS)
    eval_mask = has_label & np.isin(seed, EVAL_SEEDS)
    print("train-labeled rows:", train_mask.sum(), "eval-labeled rows:", eval_mask.sum())

    # ---- fit k_D (closed-form least squares: D_pred = k_D * D_raw) ----
    k_D = float(np.sum(D_raw[train_mask] * label_W[train_mask]) /
                (np.sum(D_raw[train_mask] ** 2) + cr.EPS))
    D_pred = k_D * D_raw
    print("k_D:", k_D)

    preds3 = (A_pred, D_pred, M_pred)
    err = np.stack([np.abs(A_pred - label_W), np.abs(D_pred - label_W), np.abs(M_pred - label_W)], axis=1)

    # ---- Layer 1 training: label1 = argmin candidate error ----
    label1 = np.argmin(err, axis=1)
    X1_raw = cr.build_X1(data, tau)
    mean1 = X1_raw[train_mask].mean(axis=0)
    std1 = X1_raw[train_mask].std(axis=0) + 1e-6
    X1s = cr.build_X1_std(X1_raw, mean1, std1)

    print("fitting layer-1 softmax ranker ...")
    Theta1 = fit_softmax(X1s[train_mask], label1[train_mask], 3, ITERS1, LR1, L2)
    train_acc1 = (np.argmax(X1s[train_mask] @ Theta1.T, axis=1) == label1[train_mask]).mean()
    print("layer-1 train top-1 accuracy (matches argmin-error candidate):", train_acc1)

    # ---- Pass A (provisional, train seeds only): builds champion trace to label layer 2 ----
    order_train = cr.proc_order(data, seed_filter=TRAIN_SEEDS)
    freeze_partial = dict(eta_h1=ETA_H1, eta_h2=ETA_H2)
    print("running Pass A (provisional switch rule) on train seeds ...")
    passA = cr.run_pass(data, order_train, preds3, tau, dtau, freeze_partial, mode="provisional",
                         Theta1=Theta1, mean1=mean1, std1=std1)

    # ---- Layer 2 labels: SWITCH=1 iff challenger beats champion by margin at label_W ----
    champ = passA["champion_id"]; chall = passA["challenger_id"]; valid = passA["challenger_valid"]
    preds_mat = np.stack(preds3, axis=1)
    champ_err = np.abs(preds_mat[np.arange(n), np.clip(champ, 0, 2)] - label_W)
    chall_err = np.abs(preds_mat[np.arange(n), np.clip(chall, 0, 2)] - label_W)
    label2 = ((chall_err + MARGIN) < champ_err).astype(int)

    l2_mask = train_mask & valid
    print("layer-2 trainable rows (challenger valid & labeled):", l2_mask.sum(),
          "switch-positive fraction:", label2[l2_mask].mean() if l2_mask.sum() else float("nan"))

    X2_raw = np.stack([passA["p1"][:, 0], passA["p1"][:, 1], passA["p1"][:, 2],
                        passA["gap"], passA["norm_pred_gap"], tau, dtau,
                        np.log1p(np.clip(passA["champion_age"], 0, None)), passA["h2"]], axis=1)
    mean2 = X2_raw[l2_mask].mean(axis=0)
    std2 = X2_raw[l2_mask].std(axis=0) + 1e-6
    X2s = cr.standardize(X2_raw, mean2, std2)
    X2s = np.concatenate([np.ones((n, 1)), X2s], axis=1)

    print("fitting layer-2 HOLD/SWITCH learner ...")
    Theta2 = fit_softmax(X2s[l2_mask], label2[l2_mask], 2, ITERS2, LR2, L2)
    train_acc2 = (np.argmax(X2s[l2_mask] @ Theta2.T, axis=1) == label2[l2_mask]).mean()
    print("layer-2 train accuracy:", train_acc2)

    # ---- freeze ----
    freeze = dict(
        line="canonical (F40 V3.9R3, CANON-BRIDGE-R1)",
        candidates=cr.CANDIDATES,
        eta_A=ETA_A, eta_M=ETA_M, eta_h1=ETA_H1, eta_h2=ETA_H2,
        k_D=k_D, W_prior=W_prior, margin=MARGIN, step_offset=STEP_OFFSET,
        legal_cols=cr.LEGAL_COLS,
        mean1=mean1.tolist(), std1=std1.tolist(), Theta1=Theta1.tolist(),
        mean2=mean2.tolist(), std2=std2.tolist(), Theta2=Theta2.tolist(),
        x2_feature_names=["p1_A", "p1_D", "p1_M", "gap_top_minus_champ",
                           "norm_pred_gap", "tau", "dtau", "log1p_champion_age", "h2"],
        train_seeds=TRAIN_SEEDS, eval_seeds=EVAL_SEEDS,
        sha256_recorder_script=cr.sha256_file(ENGINE),
        sha256_dev_table=cr.sha256_file(TABLE),
        sha256_dev_events=cr.sha256_file(EVENTS),
        cited_engine_sha256_f40_v39r3=CITED_ENGINE_SHA256,
    )
    with open(FREEZE_PATH, "w") as f:
        json.dump(freeze, f, indent=2)
    print("wrote", FREEZE_PATH)

    # ---- Pass B (deploy, frozen) on the FULL DEV table (seeds 1-6) ----
    print("running frozen controller trace over full DEV table ...")
    result = cr.trace(TABLE, EVENTS, FREEZE_PATH)
    np.savez_compressed(TRACE_PATH, **result)
    print("wrote", TRACE_PATH)

    # ---- eval metrics on seeds 5-6 ----
    chosen = result["chosen_pred"]
    rmse = lambda p, m: float(np.sqrt(np.mean((p[m] - label_W[m]) ** 2)))
    print("\n=== EVAL seed 3 (labeled rows only) ===")
    print("n eval rows:", int(eval_mask.sum()))
    print("RMSE chosen (controller):", rmse(chosen, eval_mask))
    print("RMSE A (EMA-of-past-W):  ", rmse(A_pred, eval_mask))
    print("RMSE D (rotation):       ", rmse(D_pred, eval_mask))
    print("RMSE M (momentum):       ", rmse(M_pred, eval_mask))
    n_switch = int(np.sum(result["champion_id"][eval_mask] != np.roll(result["champion_id"], 1)[eval_mask]))
    print("fraction challenger_valid (eval):", float(result["challenger_valid"][eval_mask].mean()))
    ps = result["P_SWITCH"][eval_mask]
    print("P_SWITCH mean/std/min/max (eval):", np.nanmean(ps), np.nanstd(ps), np.nanmin(ps), np.nanmax(ps))

    print("\nsha256 CONTROLLER_FREEZE.json:", cr.sha256_file(FREEZE_PATH))
    print("sha256 controller_run.py:     ", cr.sha256_file(RUN_PATH))
    print("sha256 DEV_controller_trace.npz:", cr.sha256_file(TRACE_PATH))


if __name__ == "__main__":
    main()
