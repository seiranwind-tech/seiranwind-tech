import numpy as np, json, hashlib

RNG_SEED = 12345

def load_subsample(path):
    d = np.load(path)
    cols = {k: d[k] for k in d.files}
    n = len(cols['seed'])
    rng = np.random.default_rng(RNG_SEED)
    phase49 = cols['phase'] == 49
    other = ~phase49
    keep_other = rng.random(n) < 0.20
    mask = phase49 | (other & keep_other)
    sub = {k: v[mask] for k, v in cols.items()}
    return sub

def feats_F0(d):
    n, R, yo = d['n'], d['R'], d['yo']
    a_n = np.sqrt(2*np.log(n))
    sp = np.sqrt(np.clip(1 - R**2, 0, None))
    X = np.stack([np.ones_like(yo), yo, a_n*sp], axis=1)
    names = ["1", "yo", "sqrt(2*log(n))*sqrt(1-R**2)"]
    return X, names

def feats_F1(d):
    n, R, yo, yb, skew, cob, dmu = d['n'], d['R'], d['yo'], d['yb'], d['skew'], d['cob'], d['dmu']
    a_n = np.sqrt(2*np.log(n))
    sp = np.sqrt(np.clip(1 - R**2, 0, None))
    X = np.stack([np.ones_like(yo), yo, yb, a_n*sp, skew, cob, dmu], axis=1)
    names = ["1", "yo", "yb", "sqrt(2*log(n))*sqrt(1-R**2)", "skew", "cob", "dmu"]
    return X, names

def feats_F2(d):
    X1, names1 = feats_F1(d)
    n = d['n']; so, sb, rho = d['so'], d['sb'], d['rho']
    a_n = np.sqrt(2*np.log(n))
    X = np.concatenate([X1, np.stack([a_n*so, a_n*sb, rho], axis=1)], axis=1)
    names = names1 + ["sqrt(2*log(n))*so", "sqrt(2*log(n))*sb", "rho"]
    return X, names

FAMS = {"F0_LOW": feats_F0, "F1_LEGAL": feats_F1, "F2_AUG": feats_F2}

def fit_ols(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef

def metrics(y, yhat):
    pred_event = yhat > 0
    true_event = y > 0
    sign_agree = np.mean((yhat > 0) == (y > 0))
    tp = np.sum(pred_event & true_event)
    fp = np.sum(pred_event & ~true_event)
    fn = np.sum(~pred_event & true_event)
    tn = np.sum(~pred_event & ~true_event)
    precision = tp / (tp + fp) if (tp+fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp+fn) > 0 else float('nan')
    corr = float(np.corrcoef(yhat, y)[0, 1])
    rmse = float(np.sqrt(np.mean((yhat - y)**2)))
    false_early = int(fp)   # predicted birth, none occurred
    false_late = int(fn)    # missed/late birth
    return dict(n=int(len(y)), sign_agreement=float(sign_agree), precision=float(precision),
                recall=float(recall), corr=corr, rmse=rmse, false_early=false_early,
                false_late=false_late, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

def main():
    sub = load_subsample("w1/DEV_table.npz")
    seed = sub['seed']; phase = sub['phase']; y = sub['G_birth']
    fit_mask = np.isin(seed, [1, 2, 3, 4])
    val_mask = np.isin(seed, [5, 6])

    report = {}
    for fam, ffun in FAMS.items():
        X, names = ffun(sub)
        Xf, yf = X[fit_mask], y[fit_mask]
        coef = fit_ols(Xf, yf)
        Xv, yv = X[val_mask], y[val_mask]
        yhat_v = Xv @ coef
        m_all = metrics(yv, yhat_v)
        p49 = phase[val_mask] == 49
        m_p49 = metrics(yv[p49], yhat_v[p49])
        report[fam] = dict(names=names, coef=coef.tolist(), val_all=m_all, val_phase49=m_p49)
        print(fam, "coef=", dict(zip(names, coef)))
        print("  val_all:", m_all)
        print("  val_phase49:", m_p49)

    with open("w2/_split_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
