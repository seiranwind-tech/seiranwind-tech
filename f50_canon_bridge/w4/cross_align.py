"""W4 — cross-surface alignment evaluator on sealed FRESH (seeds 101-106).
Imports only frozen APIs: w2/bridge_eval.predict, w3/controller_run.trace."""
import sys, json, hashlib, numpy as np
sys.path[:0] = ["w2", "w3"]
import bridge_eval, controller_run

NOC = 99
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def windows(d):
    key = (d["seed"].astype(np.int64) * 10**9 + d["basin"].astype(np.int64) * 10**4 + d["redraw_index"])
    o = np.lexsort((d["phase"], key)); k = key[o]
    starts = np.r_[0, np.where(np.diff(k))[0] + 1]
    return o, starts

def first_cross(cond, phase, o, starts):
    v = np.where(cond[o], phase[o], NOC)
    return np.minimum.reduceat(v, starts)

def dt_stats(a, b):
    both = (a < NOC) & (b < NOC); dt = (b - a)[both].astype(float)
    if not both.any(): return dict(n_both=0)
    ad = np.abs(dt)
    return dict(n_both=int(both.sum()), median_dt=float(np.median(dt)), median_abs=float(np.median(ad)),
                p90_abs=float(np.percentile(ad, 90)), p95_abs=float(np.percentile(ad, 95)),
                frac_le1=float((ad <= 1).mean()), frac_le10=float((ad <= 10).mean()), frac_le50=float((ad <= 50).mean()),
                early=int((dt < 0).sum()), late=int((dt > 0).sum()), exact=int((dt == 0).sum()))

def window_agree(a, b):
    ca, cb = a < NOC, b < NOC
    return dict(windows=int(len(a)), crossA=int(ca.sum()), crossB=int(cb.sum()), both=int((ca & cb).sum()),
                A_only_missed_by_B=int((ca & ~cb).sum()), B_only_false=int((~ca & cb).sum()),
                window_sign_agree=float((ca == cb).mean()),
                recall=float((ca & cb).sum() / max(ca.sum(), 1)), precision=float((ca & cb).sum() / max(cb.sum(), 1)))

def row_metrics(g, f):
    return dict(sign_agree=float(((g > 0) == (f > 0)).mean()), corr=float(np.corrcoef(g, f)[0, 1]),
                precision=float(((g > 0) & (f > 0)).sum() / max((f > 0).sum(), 1)),
                recall=float(((g > 0) & (f > 0)).sum() / max((g > 0).sum(), 1)))

def fiber(d, keys, bins=5):
    cell = np.zeros(len(d["G_birth"]), np.int64)
    for k in keys:
        q = np.quantile(d[k], np.linspace(0, 1, bins + 1)[1:-1]); cell = cell * bins + np.searchsorted(q, d[k])
    u, inv = np.unique(cell, return_inverse=True)
    pos = np.bincount(inv, d["G_birth"] > 0); tot = np.bincount(inv); p = pos / tot
    mixed = (p > 0) & (p < 1); pp = np.clip(p, 1e-12, 1 - 1e-12)
    H = -(pp * np.log2(pp) + (1 - pp) * np.log2(1 - pp)) * mixed
    return dict(mixed_frac=float(tot[mixed].sum() / tot.sum()), cond_entropy_bits=float((H * tot).sum() / tot.sum()))

def main(tab, ev):
    freeze = json.load(open("w2/BRIDGE_FREEZE.json"))
    d = {k: v for k, v in np.load(tab).items()}
    tr = controller_run.trace(tab, ev)
    assert np.array_equal(tr["step"], d["step"]) and np.array_equal(tr["basin"], d["basin"]), "trace misaligned"
    fams = [f for f in freeze.get("families", {})] if isinstance(freeze.get("families"), dict) else [f["name"] for f in freeze["families"]]
    out = dict(schema="pu_bridge.w4.v1", fresh_table_sha=sha(tab), bridge_freeze_sha=sha("w2/BRIDGE_FREEZE.json"),
               controller_freeze_sha=sha("w3/CONTROLLER_FREEZE.json"), families=fams)
    o, st = windows(d); ph = d["phase"]
    tb = first_cross(d["G_birth"] > 0, ph, o, st)
    ts = first_cross(d["G_shape"] > 0, ph, o, st)
    cv = tr["challenger_valid"].astype(bool); GG = tr["Gamma_G"]
    tg = first_cross(cv & (GG > 0), ph, o, st)
    out["event_rate_rows"] = float((d["G_birth"] > 0).mean())
    out["B_shape_vs_birth"] = dict(rows=row_metrics(d["G_birth"], d["G_shape"]), windows=window_agree(tb, ts), dt=dt_stats(tb, ts))
    out["D_birth_vs_controller"] = dict(windows=window_agree(tb, tg), dt=dt_stats(tb, tg),
                                        rows_valid=row_metrics(d["G_birth"][cv], GG[cv]) if cv.any() else None,
                                        challenger_valid_frac=float(cv.mean()))
    for f in fams:
        F = bridge_eval.predict(f, d); tq = first_cross(F > 0, ph, o, st)
        out[f"A_bridge_vs_birth[{f}]"] = dict(rows=row_metrics(d["G_birth"], F), rows_phase49=row_metrics(d["G_birth"][ph == 49], F[ph == 49]),
                                               windows=window_agree(tb, tq), dt=dt_stats(tb, tq))
        out[f"C_bridge_vs_controller[{f}]"] = dict(windows=window_agree(tq, tg), dt=dt_stats(tq, tg),
                                                    rows_valid=row_metrics(F[cv], GG[cv]) if cv.any() else None)
    # event-triggered P_SWITCH around hidden birth redraws (steps), per basin series
    key = d["seed"].astype(np.int64) * 10**6 + d["basin"]; o2 = np.lexsort((d["step"], key))
    k2, s2, ps, g2, c2 = key[o2], d["step"][o2], tr["P_SWITCH"][o2], d["G_birth"][o2], cv[o2]
    ev_i = np.where((d["phase"][o2] == 49) & (g2 > 0))[0]
    offs = list(range(-100, 101, 10)); prof = {}
    for off in offs:
        j = ev_i + off; ok = (j >= 0) & (j < len(k2)); j = j[ok]; ok2 = k2[j] == k2[ev_i[ok]]
        prof[off] = float(ps[j[ok2]].mean()) if ok2.any() else None
    out["P_SWITCH_event_triggered"] = dict(baseline=float(ps.mean()), valid_baseline=float(ps[c2].mean()) if c2.any() else None, profile=prof)
    rel = {"LEGAL": ["yo", "skew", "R", "n"], "AUG": ["yo", "so", "n", "R"]}
    out["fiber"] = {k: fiber(d, v) for k, v in rel.items()}
    json.dump(out, open("w4/W4_RESULT.json", "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
