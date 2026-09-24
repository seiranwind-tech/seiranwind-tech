import numpy as np, json
from fit_eval import load_subsample, feats_F1, feats_F2, fit_ols

def top4_importance(X, names, coef):
    # skip intercept (index 0)
    std = X.std(0)
    imp = np.abs(coef) * std
    order = np.argsort(-imp)
    order = [i for i in order if names[i] != "1"][:4]
    return order

def fiber_ambiguity(raw_cols_dict, y, n_bins=5):
    # raw_cols_dict: name -> 1D array (the coordinate values to bin on)
    names = list(raw_cols_dict)
    edges = []
    binned = []
    for nm in names:
        v = raw_cols_dict[nm]
        qs = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        qs = np.unique(qs)
        if len(qs) < 2:
            qs = np.array([v.min(), v.max() + 1e-9])
        idx = np.clip(np.digitize(v, qs[1:-1], right=False), 0, len(qs) - 2)
        binned.append(idx)
    binned = np.stack(binned, axis=1)  # rows x k
    # cell key
    keys = binned[:, 0].astype(np.int64)
    mult = 1
    for j in range(1, binned.shape[1]):
        mult *= (n_bins)
        keys = keys * n_bins + binned[:, j]
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    pos = (y > 0).astype(float)
    sum_pos = np.bincount(inv, weights=pos, minlength=len(uniq))
    n_cell = counts.astype(float)
    p = sum_pos / n_cell
    both_signs = (sum_pos > 0) & (sum_pos < n_cell)
    frac_rows_in_ambiguous_cells = float(n_cell[both_signs].sum() / n_cell.sum())
    eps = 1e-12
    ent = -(p*np.log2(p+eps) + (1-p)*np.log2(1-p+eps))
    ent[np.isclose(p, 0) | np.isclose(p, 1)] = 0.0
    mean_entropy = float(np.sum(ent * n_cell) / n_cell.sum())
    return dict(n_cells=int(len(uniq)), n_rows=int(n_cell.sum()),
                frac_rows_in_ambiguous_cells=frac_rows_in_ambiguous_cells,
                mean_conditional_sign_entropy=mean_entropy, coords=names, n_bins=n_bins)

def main():
    sub = load_subsample("w1/DEV_table.npz")
    y = sub['G_birth']
    seed = sub['seed']
    fit_mask = np.isin(seed, [0, 1, 2])

    X1, names1 = feats_F1(sub)
    coef1 = fit_ols(X1[fit_mask], y[fit_mask])
    top1 = top4_importance(X1, names1, coef1)
    print("F1 top4:", [names1[i] for i in top1])

    X2, names2 = feats_F2(sub)
    coef2 = fit_ols(X2[fit_mask], y[fit_mask])
    top2 = top4_importance(X2, names2, coef2)
    print("F2 top4:", [names2[i] for i in top2])

    coord_map = {
        "yo": sub['yo'], "yb": sub['yb'], "skew": sub['skew'], "cob": sub['cob'],
        "dmu": sub['dmu'], "so": sub['so'], "sb": sub['sb'], "rho": sub['rho'],
        "sqrt(2*log(n))*sqrt(1-R**2)": np.sqrt(2*np.log(sub['n']))*np.sqrt(np.clip(1-sub['R']**2,0,None)),
        "sqrt(2*log(n))*so": np.sqrt(2*np.log(sub['n']))*sub['so'],
        "sqrt(2*log(n))*sb": np.sqrt(2*np.log(sub['n']))*sub['sb'],
    }

    coords1 = {names1[i]: coord_map[names1[i]] for i in top1}
    coords2 = {names2[i]: coord_map[names2[i]] for i in top2}

    fib1 = fiber_ambiguity(coords1, y, n_bins=5)
    fib2 = fiber_ambiguity(coords2, y, n_bins=5)
    print("F1 fiber:", fib1)
    print("F2 fiber:", fib2)

    out = dict(F1_LEGAL=fib1, F2_AUG=fib2)
    with open("w2/_fiber_report.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
