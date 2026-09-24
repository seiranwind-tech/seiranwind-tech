"""W2 frozen L1.5 observable-bridge evaluator.

Implements the preregistered families exactly as frozen in BRIDGE_FREEZE.json.
`predict(family, table_dict)` takes a dict of numpy arrays (as np.load(...).items()
or a row-subset thereof would give you) with at least the LEGAL columns
n, R, yo, yb, skew, cob, dmu, and (for F2_AUG only) the PROPOSED_AUGMENTATION
columns so, sb, rho. Returns the predicted signed margin F_hat; event_hat = F_hat > 0.
"""
import numpy as np

# Frozen coefficients (fit by OLS on DEV seeds 1-6, full precision).
COEFS = {
    "F0_LOW": {
        "names": ["1", "yo", "sqrt(2*log(n))*sqrt(1-R**2)"],
        "coef": [1.08824033800514, -1.2090485046195865, 0.12544417439152175],
    },
    "F1_LEGAL": {
        "names": ["1", "yo", "yb", "sqrt(2*log(n))*sqrt(1-R**2)", "skew", "cob", "dmu"],
        "coef": [0.2494189269759505, -0.3670112992265494, -0.04010087689791319,
                 0.1878925410677737, 1.833050384055201, 0.026036070760012013,
                 0.09085124337219859],
    },
    "F2_AUG": {
        "names": ["1", "yo", "yb", "sqrt(2*log(n))*sqrt(1-R**2)", "skew", "cob", "dmu",
                  "sqrt(2*log(n))*so", "sqrt(2*log(n))*sb", "rho"],
        "coef": [0.453079247364239, -0.5401804163014536, -0.025215508806552094,
                 0.030440828004966697, 0.33540364784826465, 0.007671907216516102,
                 0.03193147433146947, 0.9438181829566333, -0.0013379521856482551,
                 0.0009339001791915586],
    },
}

THRESHOLD = 0.0


def _spread_term(t):
    n, R = t["n"], t["R"]
    a_n = np.sqrt(2 * np.log(n))
    sp = np.sqrt(np.clip(1 - R ** 2, 0, None))
    return a_n * sp


def _feats_F0(t):
    yo = t["yo"]
    return np.stack([np.ones_like(yo), yo, _spread_term(t)], axis=1)


def _feats_F1(t):
    yo, yb, skew, cob, dmu = t["yo"], t["yb"], t["skew"], t["cob"], t["dmu"]
    return np.stack([np.ones_like(yo), yo, yb, _spread_term(t), skew, cob, dmu], axis=1)


def _feats_F2(t):
    X1 = _feats_F1(t)
    n = t["n"]; so, sb, rho = t["so"], t["sb"], t["rho"]
    a_n = np.sqrt(2 * np.log(n))
    return np.concatenate([X1, np.stack([a_n * so, a_n * sb, rho], axis=1)], axis=1)


_FEATS = {"F0_LOW": _feats_F0, "F1_LEGAL": _feats_F1, "F2_AUG": _feats_F2}


def predict(family, table_dict):
    """Return predicted signed margin F_hat (np.ndarray) for the given family.

    family: one of "F0_LOW", "F1_LEGAL", "F2_AUG"
    table_dict: dict-like of numpy arrays keyed by column name (n, R, yo, yb,
        skew, cob, dmu, and so/sb/rho for F2_AUG).
    """
    if family not in COEFS:
        raise ValueError(f"unknown family {family!r}; expected one of {list(COEFS)}")
    X = _FEATS[family](table_dict)
    coef = np.asarray(COEFS[family]["coef"])
    return X @ coef
