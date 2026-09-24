"""W9 — L1.5 shadow simulator for the full L1.5 state (labels, centres, tracks, ages, basin graph).
The shadow never reads engine centres/labels/tracks after init; it receives only handoffs:
  H_S(t)  : observed basin centres  normalize(sum_{w in i} zin_w / n_i)   [K x d]   at centre-update steps
  H_T(t_r): per-well (best_label, best>=thr, orphan) from the reassign base scan + zin of orphans (new centres)
Static structure allowed: well kNN graph (fixed at init), config constants, engine bandwidth.
Compares shadow state with the engine after every step."""
import sys, os, json, hashlib, numpy as np
ENG_DIR = os.environ["F50_ENGINE_DIR"]
assert hashlib.sha256(open(os.path.join(ENG_DIR, "f40_v39r3_engine.py"), "rb").read()).hexdigest() == "4a71aeb4a66a8a071a6b6f9a9b065687d237909aea5e89a537ee6d80d2aa3b1d"
sys.path.insert(0, ENG_DIR); import f40_v39r3_engine as E
nr = E.normalize_rows

class Shadow:
    def __init__(s, eng):
        c = eng.config; s.cfg = c; s.bw = eng.bandwidth; s.nbr = eng.neighbors.copy(); s.thr = float(eng.assignment_threshold)
        s.labels = eng.labels.copy(); s.centres = eng.centres.copy(); s.tracks = eng.centre_track_ids.copy()
        s.ages = eng.basin_ages.copy(); s.next_track = eng.next_track_id; s.bn = eng.basin_neighbors.copy()
    def basin_graph(s):
        lab = s.labels.astype(np.int64); count = len(s.centres); width = s.cfg.basin_neighbor_k
        src = np.repeat(lab, s.nbr.shape[1]); tgt = lab[s.nbr].ravel(); keep = src != tgt
        pairs = np.unique(np.stack([src[keep], tgt[keep]], 1), axis=0) if keep.any() else np.empty((0, 2), np.int64)
        res = np.tile(np.arange(count, dtype=np.int32)[:, None], (1, width))
        if len(pairs):
            sim = np.einsum("id,id->i", s.centres[pairs[:, 0]], s.centres[pairs[:, 1]])
            pairs = pairs[np.lexsort((pairs[:, 1], -sim, pairs[:, 0]))]
            st = np.searchsorted(pairs[:, 0], np.arange(count), "left"); sp = np.searchsorted(pairs[:, 0], np.arange(count), "right")
            for b in range(count):
                r = pairs[st[b]:sp[b], 1]
                if len(r) == 0: r = np.array([b])
                if len(r) < width: r = np.concatenate([r, np.full(width - len(r), b)])
                res[b] = r[:width]
        s.bn = res
    def predict(s):
        C = s.centres
        if s.bn is None or len(C) < 2 or s.bn.shape[0] != len(C): return C.copy()
        out = np.empty_like(C); ch = s.cfg.well_chunk
        for a in range(0, len(C), ch):
            b = min(a + ch, len(C)); adj = C[s.bn[a:b]]; loc = C[a:b]
            sim = np.einsum("bd,bkd->bk", loc, adj); w = np.exp(sim / s.bw); w /= w.sum(1, keepdims=True) + 1e-9
            att = np.einsum("bk,bkd->bd", w, adj) - loc; rel = np.sign(sim - np.median(sim, axis=1, keepdims=True))
            blade = np.sum(rel[:, :, None] * (sim[:, :, None] * loc[:, None, :] - adj), axis=1)
            out[a:b] = nr(loc + s.cfg.dt * (s.cfg.basin_rho * blade + s.cfg.basin_couple * att))
        return out
    def predict_only(s):                  # thinned mode: no H_S between gates
        s.centres = s.predict(); s.ages = s.ages + 1; s.basin_graph()
    def centre_update(s, observed):        # observed = H_S
        if len(observed) != len(s.centres):
            s.centres = observed; s.ages = np.ones(len(observed), np.int32)
        else:
            g = s.cfg.observation_gain; s.centres = nr((1.0 - g) * s.predict() + g * observed); s.ages = s.ages + 1
        s.basin_graph()
    def reassign(s, best_label, best_ok, orphan_idx, orphan_z, observed_fn):
        prop = np.where(best_ok, best_label, s.labels).astype(np.int32)
        births = 0
        if len(orphan_idx):
            base = len(s.centres); prop[orphan_idx] = np.arange(base, base + len(orphan_idx), dtype=np.int32)
            s.centres = np.concatenate([s.centres, nr(orphan_z.astype(np.float32))])
            s.ages = np.concatenate([s.ages, np.ones(len(orphan_idx), np.int32)])
            s.tracks = np.concatenate([s.tracks, np.arange(s.next_track, s.next_track + len(orphan_idx), dtype=np.int64)])
            s.next_track += len(orphan_idx); births = len(orphan_idx)
        u = np.unique(prop); remap = np.full(int(u.max()) + 1, -1, np.int32); remap[u] = np.arange(len(u), dtype=np.int32)
        s.labels = remap[prop]; s.centres = s.centres[u]; s.ages = s.ages[u].astype(np.int32); s.tracks = s.tracks[u.astype(np.int64)]
        s.bn = None
        s.centre_update(observed_fn(s.labels, len(s.centres)))   # H_S after relabel (same zin)
        s.merge()
        return births
    def merge(s):
        C = s.centres; T = s.bn.astype(np.int64); count = len(C)
        if count < 2 or s.cfg.merge_threshold >= 1.0: return
        src = np.repeat(np.arange(count), T.shape[1]); tgt = T.ravel(); k = src < tgt; src, tgt = src[k], tgt[k]
        if not len(src): return
        sim = np.einsum("id,id->i", C[src], C[tgt]); close = sim >= s.cfg.merge_threshold
        if not close.any(): return
        parent = np.arange(count)
        def root(i):
            while parent[i] != i: parent[i] = parent[parent[i]]; i = int(parent[i])
            return i
        pairs = np.stack([src[close], tgt[close]], 1)[np.argsort(-sim[close])]
        for a, b in pairs.tolist():
            ra, rb = root(a), root(b)
            if ra == rb: continue
            kr, dr = (ra, rb) if s.tracks[ra] < s.tracks[rb] else (rb, ra); parent[dr] = kr
        groups = {}
        for i in range(count): groups.setdefault(root(i), []).append(i)
        if len(groups) == count: return
        ordered = sorted(groups.values(), key=lambda g: int(np.min(s.tracks[g])))
        o2n = np.empty(count, np.int32); nc, nt, na = [], [], []
        for nl, g in enumerate(ordered):
            o2n[g] = nl; nc.append(nr(C[g].mean(0, keepdims=True))[0]); nt.append(int(np.min(s.tracks[g]))); na.append(int(np.max(s.ages[g])))
        s.labels = o2n[s.labels.astype(np.int64)]; s.centres = np.asarray(nc, np.float32); s.tracks = np.asarray(nt, np.int64)
        s.ages = np.asarray(na, np.int32); s.basin_graph()

def observed(zin, labels, K):
    sums = np.zeros((K, zin.shape[1]), np.float32); cnt = np.zeros(K, np.float32)
    np.add.at(sums, labels, zin); np.add.at(cnt, labels, np.ones(len(labels), np.float32))
    return nr(sums / (cnt[:, None] + 1e-9))

def make_engine(block, perm_seed, cfg_seed, n=2000):
    V = np.load(os.environ["F50_VEC"]); names = open(os.environ["F50_WORDS"]).read().split("\n")
    idx = np.sort(np.random.default_rng(perm_seed).permutation(len(V))[block * n:(block + 1) * n])
    cfg = E.ScalableRunConfigV32(profile="F40_V3_9_NATIVE_L1_NONCANONICAL", glove_path="none", output_dir="/tmp/o",
                                 n_wells=n, dimension=V.shape[1], total_steps=6000, seed=cfg_seed)
    return E.ScalableF40EngineV32(cfg, [names[i] for i in idx], V[idx])

def run(block, perm_seed, cfg_seed, T=6000):
    eng = make_engine(block, perm_seed, cfg_seed); sh = Shadow(eng); sh2 = Shadow(eng); d = eng.config.dimension
    R2 = dict(gates=0, track_disagree_frac_sum=0.0, K_abs_diff_sum=0, centre_angle_err_max_deg=0.0, births_shadow2=0)
    R = dict(steps=0, label_mismatch_steps=0, track_mismatch_steps=0, age_mismatch_steps=0, K_mismatch_steps=0,
             basin_graph_mismatch_steps=0, centre_max_abs_err=0.0, centre_bitwise_equal_steps=0, first_failure=None,
             births=0, reassigns=0, payload={"H_S_bytes": 0, "H_T_bytes": 0}, gate_sign_mismatch_with_shadow_state=0)
    for _ in range(T):
        cs = eng.step_number + 2; z = eng.zin
        if cs % 50 == 0:
            cand = sh.labels[sh.nbr]; sim = np.einsum("nd,nkd->nk", z, sh.centres[cand])   # the base scan, but on SHADOW state
            ba = sim.argmax(1); bs = sim[np.arange(len(z)), ba]; bl = cand[np.arange(len(z)), ba]
            own = np.einsum("nd,nd->n", z, sh.centres[sh.labels])
            orph = np.where((bs < sh.thr) & (own < sh.thr))[0]
            if len(orph) > eng.config.birth_budget_per_reassign: orph = orph[np.argsort(own[orph])[:eng.config.birth_budget_per_reassign]]
            prev = sh.labels.copy()
            R["births"] += sh.reassign(bl, bs >= sh.thr, orph, z[orph], lambda L, K: observed(z, L, K)); R["reassigns"] += 1
            moved = int((sh.tracks[sh.labels] != 0).sum())  # placeholder not used
            R["payload"]["H_T_bytes"] += 8 * int((bl != prev).sum()) + (4 + 4 * d) * len(orph)
            R["payload"]["H_S_bytes"] += 4 * d * len(sh.centres)
            # thinned shadow sh2: same rule on its own state; H_S only here
            cand2 = sh2.labels[sh2.nbr]; sim2 = np.einsum("nd,nkd->nk", z, sh2.centres[cand2]); ba2 = sim2.argmax(1)
            bs2 = sim2[np.arange(len(z)), ba2]; bl2 = cand2[np.arange(len(z)), ba2]; own2 = np.einsum("nd,nd->n", z, sh2.centres[sh2.labels])
            o2 = np.where((bs2 < sh2.thr) & (own2 < sh2.thr))[0][:eng.config.birth_budget_per_reassign]
            R2["births_shadow2"] += sh2.reassign(bl2, bs2 >= sh2.thr, o2, z[o2], lambda L, K: observed(z, L, K))
        elif cs % 5 == 0:
            sh2.predict_only()
            sh.centre_update(observed(z, sh.labels, len(sh.centres))); R["payload"]["H_S_bytes"] += 4 * d * len(sh.centres)
        eng.step(); R["steps"] += 1
        bad = []
        if len(sh.centres) != len(eng.centres): R["K_mismatch_steps"] += 1; bad.append("K")
        else:
            if not np.array_equal(sh.labels, eng.labels): R["label_mismatch_steps"] += 1; bad.append("labels")
            if not np.array_equal(sh.tracks, eng.centre_track_ids): R["track_mismatch_steps"] += 1; bad.append("tracks")
            if not np.array_equal(sh.ages, eng.basin_ages): R["age_mismatch_steps"] += 1; bad.append("ages")
            if eng.basin_neighbors is not None and not np.array_equal(sh.bn, eng.basin_neighbors): R["basin_graph_mismatch_steps"] += 1; bad.append("graph")
            e = float(np.abs(sh.centres - eng.centres).max()); R["centre_max_abs_err"] = max(R["centre_max_abs_err"], e)
            R["centre_bitwise_equal_steps"] += int(e == 0.0)
        if bad and R["first_failure"] is None: R["first_failure"] = dict(step=cs, what=bad)
        if cs % 50 == 0:
            R2["gates"] += 1; R2["K_abs_diff_sum"] += abs(len(sh2.centres) - len(eng.centres))
            R2["track_disagree_frac_sum"] += float(np.mean(sh2.tracks[sh2.labels] != eng.centre_track_ids[eng.labels]))
            if len(sh2.centres) == len(eng.centres) and np.array_equal(sh2.tracks, eng.centre_track_ids):
                cosv = np.clip(np.sum(sh2.centres * eng.centres, 1), -1, 1); R2["centre_angle_err_max_deg"] = max(R2["centre_angle_err_max_deg"], float(np.degrees(np.arccos(cosv.min()))))
        if len(eng.centre_track_ids) < 3: break
    R["engine_lineage"] = eng.lineage_counts
    R2["track_disagree_frac_mean"] = R2["track_disagree_frac_sum"] / max(R2["gates"], 1); R2["K_abs_diff_mean"] = R2["K_abs_diff_sum"] / max(R2["gates"], 1)
    R["thinned_shadow_HS_only_at_gates"] = R2
    return dict(block=block, perm_seed=perm_seed, cfg_seed=cfg_seed, results=R)

if __name__ == "__main__":
    b, ps, c, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    json.dump(run(b, ps, c), open(out, "w"), indent=1, default=float); print("done", out)
