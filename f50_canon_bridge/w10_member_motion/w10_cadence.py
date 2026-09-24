"""W10-F cadence ablation. Each variant is a W9 Shadow; H_S is delivered only at steps cs % cadence == 0 (and always at gates,
where the base scan already runs). Between deliveries the observation term uses a frozen L1.5 predictor:
  HOLD   : last delivered observed centres
  EXTRAP : linear extrapolation of the last two deliveries (falls back to HOLD if K changed)."""
import sys, json, numpy as np
sys.path.insert(0, "."); from w10_common import *
CADENCES = [5, 10, 15, 20, 25, 50]; PREDICTORS = ["HOLD", "EXTRAP"]

class V:
    def __init__(s, eng, cad, pred):
        s.sh = Shadow(eng); s.cad = cad; s.pred = pred; s.last = None; s.prev = None; s.tl = s.tp = None
        s.first_div = None; s.births = 0; s.merges = 0; s.gate_birth_mismatch = 0; s.track_agree = []; s.Kdiff = []; s.angle_max = 0.0; s.handoffs = 0
    def deliver(s, o, cs):
        if s.last is not None and s.last.shape == o.shape: s.prev, s.tp = s.last, s.tl
        else: s.prev = None
        s.last, s.tl = o, cs; s.handoffs += 1
    def guess(s, cs):
        if s.last is None or s.last.shape[0] != len(s.sh.centres): return None
        if s.pred == "EXTRAP" and s.prev is not None and s.tl != s.tp:
            return nr(s.last + (s.last - s.prev) * ((cs - s.tl) / (s.tl - s.tp)))
        return s.last

def merges_wrapped(sh):
    k0 = len(sh.centres); sh.merge(); return k0 - len(sh.centres)

def run(block, perm_seed, cfg_seed, T=6000):
    eng = make_engine(block, perm_seed, cfg_seed); c = eng.config
    vs = [V(eng, cad, p) for cad in CADENCES for p in PREDICTORS if not (cad == 5 and p == "EXTRAP")]
    for _ in range(T):
        cs = eng.step_number + 2; z = eng.zin
        b0 = eng.lineage_counts["birth"]
        if cs % 50 == 0:
            for v in vs:
                sh = v.sh; cand = sh.labels[sh.nbr]; sim = np.einsum("nd,nkd->nk", z, sh.centres[cand]); ba = sim.argmax(1)
                bs = sim[np.arange(len(z)), ba]; bl = cand[np.arange(len(z)), ba]; own = np.einsum("nd,nd->n", z, sh.centres[sh.labels])
                orph = np.where((bs < sh.thr) & (own < sh.thr))[0]
                if len(orph) > c.birth_budget_per_reassign: orph = orph[np.argsort(own[orph])[:c.birth_budget_per_reassign]]
                box = {}
                def ob(L, K, z=z): o = observed(z, L, K); box["o"] = o; return o
                orig_merge = sh.merge; cnt = {"m": 0}
                def mw(orig=orig_merge, sh=sh):
                    k0 = len(sh.centres); orig(); cnt["m"] += k0 - len(sh.centres)
                sh.merge = mw
                v.last_gate_births = sh.reassign(bl, bs >= sh.thr, orph, z[orph], ob); sh.merge = orig_merge
                v.births += v.last_gate_births; v.merges += cnt["m"]; v.deliver(box["o"], cs)
        elif cs % 5 == 0:
            o_true = None
            for v in vs:
                if cs % v.cad == 0:
                    if o_true is None or o_true.shape[0] != len(v.sh.centres): o_true = observed(z, v.sh.labels, len(v.sh.centres))
                    o = observed(z, v.sh.labels, len(v.sh.centres)); v.sh.centre_update(o); v.deliver(o, cs)
                else:
                    g = v.guess(cs)
                    if g is None: v.sh.predict_only()
                    else: v.sh.centre_update(g)
        eng.step()
        eb = eng.lineage_counts["birth"] - b0
        for v in vs:
            sh = v.sh
            if cs % 50 == 0 and v.last_gate_births != eb: v.gate_birth_mismatch += 1
            sameK = len(sh.centres) == len(eng.centres)
            exact = sameK and np.array_equal(sh.tracks, eng.centre_track_ids) and np.array_equal(sh.labels, eng.labels) and np.array_equal(sh.centres, eng.centres)
            if not exact and v.first_div is None: v.first_div = cs
            if cs % 50 == 0:
                v.track_agree.append(float(np.mean(sh.tracks[sh.labels] == eng.centre_track_ids[eng.labels]))); v.Kdiff.append(abs(len(sh.centres) - len(eng.centres)))
            if sameK and np.array_equal(sh.tracks, eng.centre_track_ids):
                v.angle_max = max(v.angle_max, float(np.degrees(np.arccos(np.clip(np.sum(sh.centres * eng.centres, 1), -1, 1).min()))))
    out = []
    for v in vs:
        out.append(dict(cadence=v.cad, predictor=v.pred, first_divergence_step=v.first_div, bitwise_exact_all_steps=v.first_div is None,
                        births_shadow=v.births, births_engine=eng.lineage_counts["birth"], merges_shadow=v.merges, merges_engine=eng.lineage_counts["merge"],
                        gate_birth_count_mismatch=v.gate_birth_mismatch, gates=len(v.track_agree),
                        track_agreement_mean=float(np.mean(v.track_agree)), track_agreement_min=float(np.min(v.track_agree)),
                        K_abs_diff_mean=float(np.mean(v.Kdiff)), centre_angle_max_deg_when_aligned=v.angle_max,
                        HS_deliveries=v.handoffs))
    return dict(block=block, perm_seed=perm_seed, cfg_seed=cfg_seed, lineage=eng.lineage_counts, variants=out)

if __name__ == "__main__":
    b, ps, cs_, o = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    json.dump(run(b, ps, cs_), open(o, "w"), indent=1, default=float); print("done", o)
