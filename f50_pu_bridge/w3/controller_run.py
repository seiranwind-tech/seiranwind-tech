"""W3 -- ChatGPT-line analog: two-layer recurrent HOLD/SWITCH controller
over candidate predictors of the next-interval basin-centroid movement W.

Layer 1 (ranker): softmax over CANDIDATES using LEGAL aggregate features,
  with recurrent memory h1 = eta_h1*h1 + (1-eta_h1)*p1.
Layer 2 (HOLD/SWITCH): softmax([P_HOLD,P_SWITCH]) = softmax(Theta2 . X2),
  X2 = [p1(3), p_top-p_champ, normalized prediction gap, tau, dtau,
        log1p(champion_age), h2], h2 = eta_h2*h2 + (1-eta_h2)*(p_top-p_champ).
Rule: do_switch = (challenger != champion) AND P_SWITCH > P_HOLD.

Everything here is STRICTLY CAUSAL: at any row, only LEGAL aggregates of
that row and of past rows of the same (seed,basin) lineage, and realised W
of PAST (already-closed) intervals, are used. The realised W of the
interval currently in progress is never read by the controller; it is used
only externally, to build DEV training labels / evaluation metrics.

This module is self-contained (numpy only) and is shared by the training
driver (train_controller.py) and by trace(), the frozen replay entrypoint.
"""
import json
import hashlib
import os
import numpy as np

REDRAW = 50
CANDIDATES = ["A", "D", "M"]  # 0=EMA-of-past-W, 1=direction/rotation, 2=magnitude-momentum
EPS = 1e-8

LEGAL_COLS = ["n", "age", "R", "skew", "yo", "yb", "cob", "dR", "dskew", "dmu"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FREEZE = os.path.join(HERE, "CONTROLLER_FREEZE.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def softmax_row(logits):
    m = logits.max()
    e = np.exp(logits - m)
    return e / e.sum()


def standardize(X, mean, std):
    return (X - mean) / (std + EPS)


def load_freeze(freeze_path=None):
    path = freeze_path or DEFAULT_FREEZE
    with open(path) as f:
        return json.load(f)


def load_table_events(table_npz_path, events_json_path):
    """Load table columns and ONLY the 'moves' list (realised W). Never
    reads 'events' (births) -- enforced by simply not indexing that key."""
    d = np.load(table_npz_path)
    data = {k: d[k] for k in d.files}
    with open(events_json_path) as f:
        ev = json.load(f)
    moves = ev["moves"]  # ONLY this list is used; 'events' (births) is ignored
    return data, moves


def build_moves_map(moves):
    mv = {}
    for m in moves:
        r = m["step"] // REDRAW
        mv[(int(m["seed"]), int(m["basin"]), int(r))] = float(m["W"])
    return mv


def proc_order(data, seed_filter=None):
    """Return row indices ordered by (seed, basin, step) ascending, i.e.
    contiguous per (seed,basin) lineage in causal step order. Optionally
    restrict to a set/list of seeds."""
    seed = data["seed"]
    if seed_filter is not None:
        sf = set(int(s) for s in seed_filter)
        mask = np.array([int(s) in sf for s in seed], dtype=bool)
    else:
        mask = np.ones(len(seed), dtype=bool)
    idx = np.where(mask)[0]
    basin = data["basin"][idx]
    step = data["step"][idx]
    sd = data["seed"][idx]
    order_local = np.lexsort((step, basin, sd))
    return idx[order_local]


def tau_of_age(age):
    return np.clip(age.astype(np.float64) / 500.0, 0.0, 1.0)


def compute_dtau(tau, order, data):
    """dtau causal within each (seed,basin) group in proc order; 0 at the
    first row of a lineage."""
    n = len(order)
    dtau = np.zeros(n, dtype=np.float64)
    seed_o = data["seed"][order]
    basin_o = data["basin"][order]
    tau_o = tau[order]
    new_group = np.ones(n, dtype=bool)
    new_group[1:] = (seed_o[1:] != seed_o[:-1]) | (basin_o[1:] != basin_o[:-1])
    dtau_o = np.zeros(n)
    dtau_o[1:] = tau_o[1:] - tau_o[:-1]
    dtau_o[new_group] = 0.0
    dtau[order] = dtau_o
    return dtau


def compute_pass0_impl(data, order, mv, eta_A, eta_M, W_prior):
    """Sequential (single pass, theta-free) causal recurrences:
      A_pred = EMA of realised W across PAST closed intervals of this basin
      D_raw  = running sum of |dskew| since the start of the CURRENT interval
      M_pred = slow EMA of (dmu*50), continuous across interval boundaries
    Returns arrays in ORIGINAL table row order.
    """
    n = len(order)
    A_pred = np.empty(n)
    D_raw = np.empty(n)
    M_pred = np.empty(n)

    seed_o = data["seed"][order]
    basin_o = data["basin"][order]
    ri_o = data["redraw_index"][order]
    dskew_o = data["dskew"][order]
    dmu_o = data["dmu"][order]

    cur_seed = None
    cur_basin = None
    ema_W = None
    ema_M = 0.0
    cum_dskew = 0.0
    cur_ri = None

    for i in range(n):
        s = int(seed_o[i]); b = int(basin_o[i]); r = int(ri_o[i])
        if s != cur_seed or b != cur_basin:
            # new lineage
            cur_seed, cur_basin = s, b
            ema_W = None
            ema_M = 0.0
            cum_dskew = 0.0
            cur_ri = r
        elif r != cur_ri:
            # interval boundary crossed: fold in the just-closed interval's realised W
            key = (s, b, cur_ri)
            w_prev = mv.get(key)
            if w_prev is not None:
                ema_W = w_prev if ema_W is None else (eta_A * ema_W + (1 - eta_A) * w_prev)
            cum_dskew = 0.0
            cur_ri = r

        cum_dskew += abs(float(dskew_o[i]))
        ema_M = eta_M * ema_M + (1 - eta_M) * (float(dmu_o[i]) * 50.0)

        A_pred[i] = ema_W if ema_W is not None else W_prior
        D_raw[i] = cum_dskew
        M_pred[i] = ema_M

    out_A = np.empty(n); out_A[order] = A_pred
    out_D = np.empty(n); out_D[order] = D_raw
    out_M = np.empty(n); out_M[order] = M_pred
    return out_A, out_D, out_M


def build_X1(data, tau):
    cols = [data[c].astype(np.float64) for c in LEGAL_COLS] + [tau]
    return np.stack(cols, axis=1)


def build_X1_std(X1_raw, mean1, std1):
    Xs = standardize(X1_raw, np.array(mean1), np.array(std1))
    bias = np.ones((Xs.shape[0], 1))
    return np.concatenate([bias, Xs], axis=1)


def run_pass(data, order, preds3, tau, dtau, freeze, mode, Theta1, mean1, std1,
             Theta2=None, mean2=None, std2=None):
    """Sequential layer-1 + layer-2 causal recurrence over `order`.
    preds3 = (A_pred, D_pred, M_pred) arrays in ORIGINAL row order.
    mode: 'provisional' -> do_switch via raw softmax comparison (p_top>p_champ), Theta2 unused for the decision (still computed if given, for reporting NaN otherwise).
          'deploy'       -> do_switch via trained Theta2 (P_SWITCH>P_HOLD).
    Returns dict of arrays in ORIGINAL row order (only for indices in `order`; others left as provided fill via caller).
    """
    n = len(data["seed"])  # full row count of `data`; `order` may be a subset (e.g. train seeds)
    n_proc = len(order)
    X1_raw = build_X1(data, tau)  # original order, all rows present in data
    X1s_full = build_X1_std(X1_raw, mean1, std1)  # original order

    A_pred, D_pred, M_pred = preds3

    P_HOLD = np.full(n, np.nan)
    P_SWITCH = np.full(n, np.nan)
    champion_id = np.full(n, -1, dtype=np.int64)
    challenger_id = np.full(n, -1, dtype=np.int64)
    challenger_valid = np.zeros(n, dtype=bool)
    Gamma_G = np.full(n, np.nan)
    p1_out = np.full((n, 3), np.nan)
    gap_out = np.full(n, np.nan)
    norm_gap_out = np.full(n, np.nan)
    h2_out = np.full(n, np.nan)
    champ_age_out = np.full(n, -1, dtype=np.int64)

    eta_h1 = freeze["eta_h1"]
    eta_h2 = freeze["eta_h2"]

    seed_o = data["seed"][order]
    basin_o = data["basin"][order]

    cur_seed = None
    cur_basin = None
    h1 = None
    h2 = 0.0
    champ = None
    champ_age = 0

    Theta1 = np.array(Theta1)
    if Theta2 is not None:
        Theta2 = np.array(Theta2)

    for pos in range(n_proc):
        i = order[pos]
        s = int(seed_o[pos]); b = int(basin_o[pos])
        new_lineage = (s != cur_seed or b != cur_basin)
        if new_lineage:
            cur_seed, cur_basin = s, b
            h2 = 0.0
            champ = None
            champ_age = 0

        x1 = X1s_full[i]
        logits1 = Theta1 @ x1
        p1 = softmax_row(logits1)
        if new_lineage or h1 is None:
            h1 = p1.copy()
        else:
            h1 = eta_h1 * h1 + (1 - eta_h1) * p1

        if champ is None:
            champ = int(np.argmax(h1))
            champ_age = 0

        chall = int(np.argmax(h1))
        chall_valid = (chall != champ)
        p_top = h1[chall]
        p_champ = h1[champ]
        gap = p_top - p_champ

        preds = (A_pred[i], D_pred[i], M_pred[i])
        norm_gap = abs(preds[chall] - preds[champ]) / (abs(preds[champ]) + EPS)

        h2 = eta_h2 * h2 + (1 - eta_h2) * gap if not new_lineage else gap

        champ_age_log = np.log1p(champ_age)
        x2_raw = np.array([p1[0], p1[1], p1[2], gap, norm_gap, tau[i], dtau[i], champ_age_log, h2])

        if mode == "deploy":
            x2s = standardize(x2_raw, np.array(mean2), np.array(std2))
            x2s = np.concatenate([[1.0], x2s])
            logits2 = Theta2 @ x2s
            p2 = softmax_row(logits2)
            p_hold, p_switch = p2[0], p2[1]
            do_switch = chall_valid and (p_switch > p_hold)
        else:  # provisional (label-generation pass; Theta2 not used for the decision)
            p_hold, p_switch = np.nan, np.nan
            do_switch = chall_valid and (p_top > p_champ)

        P_HOLD[i] = p_hold
        P_SWITCH[i] = p_switch
        champion_id[i] = champ
        challenger_id[i] = chall
        challenger_valid[i] = chall_valid
        Gamma_G[i] = p_switch - 0.5 if mode == "deploy" else np.nan
        p1_out[i] = p1
        gap_out[i] = gap
        norm_gap_out[i] = norm_gap
        h2_out[i] = h2
        champ_age_out[i] = champ_age

        if do_switch:
            champ = chall
            champ_age = 0
        else:
            champ_age += 1

    return dict(P_HOLD=P_HOLD, P_SWITCH=P_SWITCH, champion_id=champion_id,
                challenger_id=challenger_id, challenger_valid=challenger_valid,
                Gamma_G=Gamma_G, p1=p1_out, gap=gap_out, norm_pred_gap=norm_gap_out,
                h2=h2_out, champion_age=champ_age_out)


def trace(table_npz_path, events_json_path, freeze_path=None):
    """Replay the FROZEN controller over any table with the same schema.
    Strictly causal: only LEGAL aggregates and PAST realised W (from the
    'moves' list) are used; realised W of the current interval is never
    read. Returns a dict of numpy arrays aligned to the table row order.
    """
    freeze = load_freeze(freeze_path)
    data, moves = load_table_events(table_npz_path, events_json_path)
    mv = build_moves_map(moves)

    order = proc_order(data)  # all seeds present in the table
    tau = tau_of_age(data["age"])
    dtau = compute_dtau(tau, order, data)

    A_pred, D_raw, M_pred = compute_pass0_impl(
        data, order, mv, freeze["eta_A"], freeze["eta_M"], freeze["W_prior"])
    D_pred = freeze["k_D"] * D_raw

    out = run_pass(
        data, order, (A_pred, D_pred, M_pred), tau, dtau, freeze, mode="deploy",
        Theta1=freeze["Theta1"], mean1=freeze["mean1"], std1=freeze["std1"],
        Theta2=freeze["Theta2"], mean2=freeze["mean2"], std2=freeze["std2"])

    n = len(data["seed"])
    chosen_pred = np.empty(n)
    preds_mat = np.stack([A_pred, D_pred, M_pred], axis=1)
    chosen_pred = preds_mat[np.arange(n), out["champion_id"]]

    result = dict(
        seed=data["seed"], step=data["step"], basin=data["basin"],
        redraw_index=data["redraw_index"], phase=data["phase"],
        P_HOLD=out["P_HOLD"], P_SWITCH=out["P_SWITCH"],
        champion_id=out["champion_id"], challenger_id=out["challenger_id"],
        challenger_valid=out["challenger_valid"], Gamma_G=out["Gamma_G"],
        A_pred=A_pred, D_pred=D_pred, M_pred=M_pred, chosen_pred=chosen_pred,
        tau=tau, dtau=dtau, champion_age=out["champion_age"],
    )
    return result


if __name__ == "__main__":
    import sys
    r = trace(sys.argv[1], sys.argv[2])
    print({k: v.shape for k, v in r.items()})
