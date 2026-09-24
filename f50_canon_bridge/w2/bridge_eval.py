"""W2 frozen L1.5 observable-bridge evaluator (CANONICAL engine line, CANON-BRIDGE-R1).

Implements the preregistered families (same bases as the PU line's PREREG.json,
coefficients refit on the canonical DEV table) exactly as frozen in BRIDGE_FREEZE.json.
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
        "coef": [0.04466612407890401, -0.14510186954246476, 0.1253619118956314],
    },
    "F1_LEGAL": {
        "names": ["1", "yo", "yb", "sqrt(2*log(n))*sqrt(1-R**2)", "skew", "cob", "dmu"],
        "coef": [-0.04522528601147161, -0.052811738452468976, -0.049464183517215675,
                 0.12982607418357833, 0.7527409104608137, 0.04476542251996097,
                 -0.0057510635202136775],
    },
    "F2_AUG": {
        "names": ["1", "yo", "yb", "sqrt(2*log(n))*sqrt(1-R**2)", "skew", "cob", "dmu",
                  "sqrt(2*log(n))*so", "sqrt(2*log(n))*sb", "rho"],
        "coef": [0.6291478446055013, -0.7273009369982016, -0.008443823135307976,
                 0.025577981800864687, -0.06055651501978589, 0.008551490405684556,
                 -0.0007126847028858113, 1.0006987358961885, -0.06659787381185646,
                 8.227544489762552e-11],
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
