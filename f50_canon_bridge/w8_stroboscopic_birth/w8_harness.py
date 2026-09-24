"""W8 stroboscopic birth harness: hook at each reassign gate (before engine.step()) + one step after."""
import sys, os, json, hashlib, numpy as np
ENG_DIR = os.environ["F50_ENGINE_DIR"]
assert hashlib.sha256(open(os.path.join(ENG_DIR, "f40_v39r3_engine.py"), "rb").read()).hexdigest() == "4a71aeb4a66a8a071a6b6f9a9b065687d237909aea5e89a537ee6d80d2aa3b1d"
sys.path.insert(0, ENG_DIR); import f40_v39r3_engine as E
KS = [8, 16, 32, 64]

def make_engine(block, perm_seed, cfg_seed, n=2000):
    V = np.load(os.environ["F50_VEC"]); names = open(os.environ["F50_WORDS"]).read().split("\n")
    idx = np.sort(np.random.default_rng(perm_seed).permutation(len(V))[block * n:(block + 1) * n])
    cfg = E.ScalableRunConfigV32(profile="F40_V3_9_NATIVE_L1_NONCANONICAL", glove_path="none", output_dir="/tmp/o",
                                 n_wells=n, dimension=V.shape[1], total_steps=6000, seed=cfg_seed)
    return E.ScalableF40EngineV32(cfg, [names[i] for i in idx], V[idx])

def base_scan(eng):
    """exactly the engine's _reassign_sparse arrays (read-only recompute for audit)."""
    cand = eng.labels[eng.neighbors]; sim = np.einsum("nd,nkd->nk", eng.zin, eng.centres[cand])
    best = sim.max(1); own = np.einsum("nd,nd->n", eng.zin, eng.centres[eng.labels])
    return own, best, eng.labels.copy()

def run(block, perm_seed, cfg_seed, T=6000, t0=1500):
    eng = make_engine(block, perm_seed, cfg_seed); thr = float(eng.assignment_threshold); B = eng.config.birth_budget_per_reassign
    R = dict(gates=0, basin_readouts=0, sign_mismatch=0, count_mismatch_engine_counter=0, near_miss_mismatch=0,
             births_count_mismatch=0, identity_violation=0, budget_hit=0, budget_selection_mismatch=0,
             overflow={k: 0 for k in KS}, overflow_basins={k: 0 for k in KS},
             multiplicity_hist={}, max_NB=0, max_global_orphans=0, multi_birth_basins=0, birth_basins=0,
             payload_bytes={"H1": 0, "H2": 0, **{f"H3_{k}": 0 for k in KS}, "H4": 0},
             reset_seed={"argmin_in_stale_bottom16_or_entrants": 0, "total": 0}, extra_high_dim_ops_for_handoff=0)
    pending = None
    for _ in range(T):
        cs = eng.step_number + 2; rec = cs >= t0
        if pending is not None:     # one step after a reassign: audit the reset-seed question (harness-only scan)
            own2, best2, lab2 = base_scan(eng); q2 = np.maximum(own2, best2); K2 = len(eng.centres)
            o = np.lexsort((q2, lab2)); st = np.searchsorted(lab2[o], np.arange(K2)); arg = o[st[np.bincount(lab2, minlength=K2) > 0]]
            seed_set = pending
            for w in arg:
                R["reset_seed"]["total"] += 1; R["reset_seed"]["argmin_in_stale_bottom16_or_entrants"] += int(w in seed_set)
            pending = None
        if cs % 50 == 0 and rec:
            own, best, lab = base_scan(eng); K = len(eng.centres); tr_before = eng.centre_track_ids.copy()
            q = np.maximum(own, best)                                   # ← handoff: O(N) max, no new dot products
            orphan = q < thr
            assert np.array_equal(orphan, (best < thr) & (own < thr))
            nb = np.bincount(lab[orphan], minlength=K); qmin = np.full(K, np.inf); np.minimum.at(qmin, lab, q)
            n_k = np.bincount(lab, minlength=K); live = n_k > 0
            o = np.lexsort((q, lab)); st = np.searchsorted(lab[o], np.arange(K))
            stale16 = set(o[st[i] + j] for i in np.where(live)[0] for j in range(min(16, n_k[i])))
            pred = np.where(orphan)[0]
            if len(pred) > B:
                R["budget_hit"] += 1; pred_sel = pred[np.argsort(own[pred])[:B]]
            else: pred_sel = pred
            births_before = eng.lineage_counts["birth"]; next_before = eng.next_track_id
            eng.step()
            eb = eng.lineage_counts["birth"] - births_before
            R["gates"] += 1
            if eng._birth_orphan_candidates != int(orphan.sum()): R["count_mismatch_engine_counter"] += 1
            nm = int(((best < thr) & ~(own < thr)).sum())
            if eng._birth_near_misses != nm: R["near_miss_mismatch"] += 1
            if eb != len(pred_sel): R["births_count_mismatch"] += 1
            tr_after = eng.centre_track_ids[eng.labels]
            new_wells = set(np.where(tr_after >= next_before)[0].tolist())
            if not new_wells.issubset(set(pred_sel.tolist())): R["identity_violation"] += 1
            # per-basin sign: basin i births iff any member in pred_sel
            born_basin = np.zeros(K, bool); born_basin[lab[pred_sel]] = True
            for i in np.where(live & (n_k >= 1))[0]:
                R["basin_readouts"] += 1
                if ((thr - qmin[i]) > 0) != born_basin[i]: R["sign_mismatch"] += 1
            for i in np.where(nb > 0)[0]:
                R["multiplicity_hist"][int(nb[i])] = R["multiplicity_hist"].get(int(nb[i]), 0) + 1
                R["birth_basins"] += 1; R["multi_birth_basins"] += int(nb[i] > 1)
            R["max_NB"] = max(R["max_NB"], int(nb.max()) if K else 0); R["max_global_orphans"] = max(R["max_global_orphans"], int(orphan.sum()))
            for k in KS:
                ov = nb > k; R["overflow_basins"][k] += int(ov.sum()); R["overflow"][k] += int(ov.any())
            Kl = int(live.sum()); no = int(orphan.sum())
            R["payload_bytes"]["H1"] += 4 * Kl; R["payload_bytes"]["H2"] += 6 * Kl; R["payload_bytes"]["H4"] += 4 * Kl + 8 * no
            for k in KS: R["payload_bytes"][f"H3_{k}"] += int(sum(12 * min(k, n) + 4 for n in n_k[live]))
            entrants = set(np.where(eng.centre_track_ids[eng.labels] != tr_before[lab])[0].tolist()) if len(tr_before) else set()
            pending = stale16 | entrants
            continue
        eng.step()
    R["gates_expected_fraction_ok"] = True
    return dict(block=block, perm_seed=perm_seed, cfg_seed=cfg_seed, lineage=eng.lineage_counts, results=R)

if __name__ == "__main__":
    b, ps, cs_, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    json.dump(run(b, ps, cs_), open(out, "w"), indent=1, default=int); print("done", out)
