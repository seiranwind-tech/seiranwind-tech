"""W7 tracker harness around the frozen engine (read-only hook before each engine.step()).
Computes exact q_w for evaluation, simulates Q_k trackers (AUTO / GUARD_global / GUARD_basin), witness/rank dynamics,
exact-gate audit and event-aware dirty fractions."""
import sys, os, json, hashlib, numpy as np
ENG_DIR = os.environ["F50_ENGINE_DIR"]
assert hashlib.sha256(open(os.path.join(ENG_DIR, "f40_v39r3_engine.py"), "rb").read()).hexdigest() == "4a71aeb4a66a8a071a6b6f9a9b065687d237909aea5e89a537ee6d80d2aa3b1d"
sys.path.insert(0, ENG_DIR); import f40_v39r3_engine as E
KS = [1, 2, 4, 8, 16]; RB = [(1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 10**9)]
NB = [(2, 5), (6, 20), (21, 50), (51, 200), (201, 10**9)]

def make_engine(block, perm_seed, cfg_seed, n=2000):
    V = np.load(os.environ["F50_VEC"]); names = open(os.environ["F50_WORDS"]).read().split("\n")
    idx = np.sort(np.random.default_rng(perm_seed).permutation(len(V))[block * n:(block + 1) * n])
    cfg = E.ScalableRunConfigV32(profile="F40_V3_9_NATIVE_L1_NONCANONICAL", glove_path="none", output_dir="/tmp/o",
                                 n_wells=n, dimension=V.shape[1], total_steps=6000, seed=cfg_seed)
    return E.ScalableF40EngineV32(cfg, [names[i] for i in idx], V[idx])

def ranks(q, lab, K):
    o = np.lexsort((q, lab)); ls = lab[o]; start = np.searchsorted(ls, np.arange(K)); r = np.empty(len(q), np.int64)
    r[o] = np.arange(len(q)) - start[ls]; return r, o, start

def rbin(r):
    for a, b in RB:
        if a - 1 <= r <= b - 1: return f"{a}" if a == b else (f"{a}-{b}" if b < 10**9 else f">={a}")
    return "?"

def nbin(n):
    for a, b in NB:
        if a <= n <= b: return f"{a}-{b}" if b < 10**9 else f">{a-1}"

def run(block, perm_seed, cfg_seed, T=6000, t0=1500):
    eng = make_engine(block, perm_seed, cfg_seed); thr = float(eng.assignment_threshold); nbr = eng.neighbors
    modes = [f"{m}_{k}" for k in KS for m in ("AUTO", "GUARD_global", "GUARD_basin", "GUARD_local")]
    R = {m: dict(readouts=0, exact_value_mismatch=0, sign_mismatch=0, refresh=0, pass_but_wrong=0, fn=0, fp=0,
                 by_nbin={}, sign_mismatch_by_nbin={}) for m in modes}
    W = dict(rank_prev_witness={"quiet": {}, "centre_update": {}, "reassign": {}}, lifetimes=[], entrants={k: [] for k in KS},
             gaps={k: [] for k in KS}, dirty_frac=[], k_event=[], births_per_reassign=[], gate_audit=dict(match=0, mismatch=0, budget_hit=0))
    tracked = None; rebuild = True; prev_z = prev_C = None; prev_track_of_well = None; wit_prev = {}; life = {}
    prev_bottom = None
    for _ in range(T):
        cs = eng.step_number + 2; redraw = cs % 50 == 0
        z = eng.zin; C = eng.centres; lab = eng.labels; tr = eng.centre_track_ids; K = len(C)
        s_own = np.einsum("nd,nd->n", z, C[lab]); s_best = np.einsum("nd,nkd->nk", z, C[lab[nbr]]).max(1)
        q = np.maximum(s_own, s_best); n_k = np.bincount(lab, minlength=K)
        r, o, start = ranks(q, lab, K)
        qmin = np.full(K, np.inf); np.minimum.at(qmin, lab, q)
        wit = o[start[np.arange(K)]]  # well id of argmin per basin (valid where n_k>0)
        rec = cs >= t0
        track_of_well = tr[lab]
        if rebuild:
            if prev_track_of_well is not None and rec:
                changed = track_of_well != prev_track_of_well
                dirty = changed | changed[nbr].any(1)
                W["dirty_frac"].append(float(dirty.mean()))
                ent = np.bincount(lab[changed], minlength=K); W["k_event"].append(int(ent.max()) if K else 0)
                for i in range(K):  # witness rank across reassign
                    t = int(tr[i]); w = wit_prev.get(t)
                    if w is not None and n_k[i] >= 2:
                        key = rbin(int(r[w])) if track_of_well[w] == t else "left"
                        dct = W["rank_prev_witness"]["reassign"]; dct[key] = dct.get(key, 0) + 1
            tracked = {k: r < k for k in KS}
            bnd = {}
            for k in KS:
                b = np.full(K, np.inf); has = n_k > k
                idx = o[start[has] + k]; b[has] = q[idx]; bnd[k] = b
            cum_g = 0.0; cum_b = np.zeros(K); cum_l = np.zeros(K); rebuild = False
            prev_bottom = {k: tracked[k].copy() for k in KS}
        else:
            dz = np.linalg.norm(z - prev_z, axis=1); dC = float(np.linalg.norm(C - prev_C, axis=1).max())
            cum_g += float(dz.max()) + dC
            mb = np.zeros(K); np.maximum.at(mb, lab, dz); cum_b += mb + dC
            dCn = np.linalg.norm(C - prev_C, axis=1)
            wc = np.maximum(dCn[lab], dCn[lab[nbr]].max(1))          # max centre move over each well's candidate set
            ml = np.zeros(K); np.maximum.at(ml, lab, wc); cum_l += mb + ml
            if rec:
                stype = "centre_update" if cs % 5 == 0 else "quiet"
                for i in np.where(n_k >= 2)[0]:
                    t = int(tr[i]); w = wit_prev.get(t)
                    if w is not None:
                        key = rbin(int(r[w])); dct = W["rank_prev_witness"][stype]; dct[key] = dct.get(key, 0) + 1
                for k in KS:
                    cur = r < k; newin = cur & ~prev_bottom[k]
                    c = np.bincount(lab[newin], minlength=K); W["entrants"][k].append(int(c.max()))
                    prev_bottom[k] = cur
        # witness lifetime bookkeeping
        for i in np.where(n_k >= 1)[0]:
            t = int(tr[i]); w = int(wit[i])
            if wit_prev.get(t) == w: life[t] = life.get(t, 0) + 1
            else:
                if t in life and rec: W["lifetimes"].append(life[t])
                life[t] = 1
            wit_prev[t] = w
        if redraw and rec:
            g_true = thr - qmin
            born = np.where((thr - q) > 0)[0]
            W["births_per_reassign"].append(int(len(born)))
            if len(born) > eng.config.birth_budget_per_reassign: W["gate_audit"]["budget_hit"] += 1
            for k in KS:
                has = n_k > k; W["gaps"][k].extend((q[o[start[has] + k]] - qmin[has]).tolist()[:50])
            for k in KS:
                tm = np.full(K, np.inf); np.minimum.at(tm, lab[tracked[k]], q[tracked[k]])
                for mode in ("AUTO", "GUARD_global", "GUARD_basin", "GUARD_local"):
                    m = R[f"{mode}_{k}"]
                    for i in np.where(n_k >= 2)[0]:
                        m["readouts"] += 1; nb = nbin(int(n_k[i])); m["by_nbin"][nb] = m["by_nbin"].get(nb, 0) + 1
                        if mode == "AUTO": val = tm[i]
                        else:
                            cum = cum_g if mode == "GUARD_global" else (cum_b[i] if mode == "GUARD_basin" else cum_l[i])
                            if tm[i] <= bnd[k][i] - cum:
                                val = tm[i]
                                if val != qmin[i]: m["pass_but_wrong"] += 1
                            else:
                                m["refresh"] += 1; val = qmin[i]
                        if val != qmin[i]: m["exact_value_mismatch"] += 1
                        if ((thr - val) > 0) != (g_true[i] > 0):
                            m["sign_mismatch"] += 1; m["sign_mismatch_by_nbin"][nb] = m["sign_mismatch_by_nbin"].get(nb, 0) + 1
                            if g_true[i] > 0: m["fn"] += 1
                            else: m["fp"] += 1
        prev_z = z.copy(); prev_C = C.copy(); prev_track_of_well = track_of_well.copy()
        births_before = eng.lineage_counts["birth"]
        eng.step()
        if redraw:
            rebuild = True
            if rec:
                eb = eng.lineage_counts["birth"] - births_before
                W["gate_audit"]["match" if eb == W["births_per_reassign"][-1] else "mismatch"] += 1
    lt = np.array(W["lifetimes"]) if W["lifetimes"] else np.array([0])
    summ = dict(block=block, perm_seed=perm_seed, cfg_seed=cfg_seed, lineage=eng.lineage_counts, trackers=R,
                gate_audit=W["gate_audit"], births_per_reassign_max=int(max(W["births_per_reassign"] or [0])),
                witness=dict(rank_prev_witness=W["rank_prev_witness"],
                             lifetime_quantiles=np.percentile(lt, [10, 50, 90, 99]).tolist(), lifetime_n=int(len(lt)),
                             entrants_max={k: int(max(v or [0])) for k, v in W["entrants"].items()},
                             entrants_mean={k: float(np.mean(v)) if v else 0 for k, v in W["entrants"].items()},
                             gap_median={k: float(np.median(v)) if v else None for k, v in W["gaps"].items()}),
                event=dict(dirty_frac_mean=float(np.mean(W["dirty_frac"])) if W["dirty_frac"] else None,
                           dirty_frac_max=float(max(W["dirty_frac"] or [0])), k_event_max=int(max(W["k_event"] or [0])),
                           k_event_median=float(np.median(W["k_event"])) if W["k_event"] else None))
    return summ

if __name__ == "__main__":
    b, ps, cs_, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    json.dump(run(b, ps, cs_), open(out, "w"), indent=1); print("done", out)
