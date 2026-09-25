"""W13B: autonomous sub-bunch ("次生葡萄串") model for the core's OWN internal pull A_a.

Template: the validated W12C halo law (own single-well dynamics = salt + gain*nr(attraction to
point masses), integrated per ENGINE dt-step, not per coarse 5dt frame jump -- see
f50_canon_bridge/w12c_halo/w12c_law_test.py:law_owndyn).

At the start of each 10-gate block, split each core (n>=10) into k spherical-k-means sub-bunches
(centroid m_j, count n_j) from the true member positions at that block's first evaluated frame.
Each sub-bunch then evolves AUTONOMOUSLY (own macro law, no further ground truth) for the whole
rollout horizon:

    m_j <- nr( m_j + dt * [ salt_MF,j(m_j) + g * attract_gain * nr(pop_attract(m_j) - m_j) ] )

salt_MF,j(m_j): mean-field salt, each real member of sub-bunch j evaluated at m_j with ITS OWN teeth
 (same mean-field convention as w11/w12).
pop_attract(m_j): softmax (engine bandwidth) weighted mean of the population = {sibling sub-bunches
 of the SAME core} u {all OTHER groups' point masses (M_b, n_b)}, self excluded.
g: fitted scalar gain (default 1.0 = use the engine's own attract_gain unscaled, matching the halo
 template exactly).

Non-tested groups (all other river groups, core or not) advect meanwhile by plain mean-field salt
only, coarse-stepped once per frame (5 engine steps) -- same simplification used as the environment
baseline in w11_rollout.py / w12a_eval.py.

Core-centre prediction = normalize(count-weighted mean of a core's sub-bunch centroids).

Usage:
  python3 w13b_subbunch.py "name1:k1,g1;name2:k2,g2;..." tape:block:perm:seed [...] \
      [--stride N] [--maxcores K] [--t0offsets 3,13] [--horizons 1,10,50]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w12a_internal_pull"))
from w12a_common import make_engine, river_home, gmean, nr, ang

args = sys.argv[1:]
laws_arg = args[0]
specs = []
stride = 1; maxcores = None; t0offsets = [3]; Hs = [1, 10, 50]
i = 1
while i < len(args):
    if args[i] == "--stride": stride = int(args[i + 1]); i += 2
    elif args[i] == "--maxcores": maxcores = int(args[i + 1]); i += 2
    elif args[i] == "--t0offsets": t0offsets = [int(x) for x in args[i + 1].split(",")]; i += 2
    elif args[i] == "--horizons": Hs = [int(x) for x in args[i + 1].split(",")]; i += 2
    else: specs.append(args[i]); i += 1

LAWS = {}
for chunk in laws_arg.split(";"):
    name, coefs = chunk.split(":")
    k, g = coefs.split(",")
    LAWS[name] = (int(k), float(g))

AR3 = [1.1639, 0.5491, -0.7167]
MAXH = max(Hs)
rng = np.random.default_rng(0)


def skmeans(x, k, it=15, seed=0):
    n = len(x)
    if n <= k:
        C = x.copy()
        while len(C) < k:
            C = np.concatenate([C, x[:1]])
        return C[:k], np.ones(k, dtype=int), np.minimum(np.arange(n), k - 1)
    r = np.random.default_rng(seed)
    C = x[r.choice(n, k, replace=False)].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(it):
        lab = np.argmax(x @ C.T, 1)
        newC = []
        for j in range(k):
            m = lab == j
            newC.append(nr(x[m].mean(0, keepdims=True))[0] if m.any() else C[j])
        C = np.stack(newC)
    lab = np.argmax(x @ C.T, 1)
    return C, np.bincount(lab, minlength=k), lab


def salt_real_fn(eng):
    def salt_real(zm):
        inner = eng._fsalt(zm.astype(np.float32))
        return 0.5 * (inner - zm * np.sum(zm * inner, 1, keepdims=True))
    return salt_real


def pop_attract_mean(active_idx, M2, cnt2, eng, gain_scale):
    """Softmax-weighted (engine bandwidth) mean-field attraction target for EVERY active row, point
    mass weighted by cnt2, self excluded (siblings of the same core + all other groups' point
    masses). Returns the tangent pull, aligned to active_idx order."""
    bw = eng.bandwidth
    Mpop = M2[active_idx]; wpop = np.log(np.maximum(cnt2[active_idx], 1).astype(np.float64))
    n = len(active_idx)
    sim = Mpop @ Mpop.T
    sim[np.arange(n), np.arange(n)] = -1e30
    L = sim / bw + wpop[None, :]
    L -= L.max(1, keepdims=True)
    w = np.exp(L); w /= w.sum(1, keepdims=True) + 1e-12
    target = w @ Mpop
    raw = eng.config.attract_gain * gain_scale * nr(target - Mpop)
    tan = raw - np.sum(raw * Mpop, 1, keepdims=True) * Mpop
    return tan


def run_tape(f, block, perm, seed):
    eng = make_engine(block, perm, seed); dt = eng.config.dt
    salt_real = salt_real_fn(eng)
    D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]
    G, HOME = river_home(trk, h=12)
    res = {m: {k: [[], []] for k in Hs} for m in (["L0_AR3"] + list(LAWS.keys()))}
    g0, B, minsize = 60, 10, 10
    bidx = 0
    for gA in range(g0, len(G) - B + 1, B):
        lab = HOME[gA]; u, inv = np.unique(lab, return_inverse=True); Kfull = len(u)
        cnt = np.bincount(inv); keep = np.where(cnt >= minsize)[0]
        if len(keep) < 2:
            continue
        bidx += 1
        if stride > 1 and (bidx - 1) % stride != 0:
            continue
        cores = keep.copy()
        if maxcores is not None and len(cores) > maxcores:
            cores = np.sort(rng.choice(cores, size=maxcores, replace=False))
        fr0 = gA * 10 + 9
        frs = np.arange(fr0, min(fr0 + 10 * B, len(Z)))
        Ms = np.stack([nr(gmean(Z[x].astype(np.float32), inv, Kfull)) for x in frs])  # [F,Kfull,d]

        for t0 in t0offsets:
            if t0 + MAXH >= len(frs):
                continue
            fr_abs = frs[t0]
            z0 = nr(Z[fr_abs].astype(np.float32))
            M0full = Ms[t0]
            truth = {k: Ms[t0 + k][cores] for k in Hs}
            disp = {k: ang(M0full[cores], truth[k]) for k in Hs}

            # --- L0: universal AR3 baseline, core-only state ---
            Mck = M0full[cores].copy()
            ds = [Ms[t0 - i2][cores] - Ms[t0 - i2 - 1][cores] for i2 in range(3)]
            for k in range(1, MAXH + 1):
                dn = sum(AR3[i2] * ds[i2] for i2 in range(3)); Mck = nr(Mck + dn); ds = [dn] + ds[:-1]
                if k in Hs:
                    res["L0_AR3"][k][0].append(ang(Mck, truth[k])); res["L0_AR3"][k][1].append(disp[k])

            # --- sub-bunch laws ---
            for name, (kk, gain_scale) in LAWS.items():
                inv2 = inv.copy()
                M2 = M0full.copy()
                cnt2 = cnt.astype(float).copy()
                extra_M = []; extra_cnt = []
                subinfo = {}
                next_id = Kfull
                for a in cores:
                    idx_a = np.where(inv == a)[0]
                    C, cj, lab_local = skmeans(z0[idx_a], kk, seed=int(a))
                    meta_ids = np.arange(next_id, next_id + kk); next_id += kk
                    inv2[idx_a] = meta_ids[lab_local]
                    extra_M.append(C); extra_cnt.append(cj.astype(float))
                    subinfo[int(a)] = meta_ids
                if extra_M:
                    M2 = np.concatenate([M2, np.concatenate(extra_M, 0)], 0)
                    cnt2 = np.concatenate([cnt2, np.concatenate(extra_cnt, 0)], 0)
                active = np.ones(len(M2), dtype=bool); active[cores] = False  # drop emptied original core rows
                active_idx = np.where(active)[0]              # [non-tested groups] + [all sub-bunches], in order
                sub_rows = np.arange(Kfull, len(M2))           # rows that get micro-stepped
                n_sub = len(sub_rows)
                sub_pos_in_active = np.arange(len(active_idx) - n_sub, len(active_idx))  # tail slice = sub rows

                for k in range(1, MAXH + 1):
                    # 5 engine micro-steps per outer frame-step k (matches the w12c halo template)
                    for micro in range(5):
                        zm_full = M2[inv2]
                        s_real = salt_real(zm_full)
                        s_sub = np.zeros((n_sub, M2.shape[1]))
                        mask_sub_members = inv2 >= Kfull
                        if mask_sub_members.any():
                            np.add.at(s_sub, inv2[mask_sub_members] - Kfull, s_real[mask_sub_members])
                            cnt_local = np.maximum(np.bincount(inv2[mask_sub_members] - Kfull, minlength=n_sub), 1)
                            s_sub /= cnt_local[:, None]
                        attr_all = pop_attract_mean(active_idx, M2, cnt2, eng, gain_scale)
                        attr_sub = attr_all[sub_pos_in_active]
                        M2[sub_rows] = nr(M2[sub_rows] + dt * (s_sub + attr_sub))
                    # non-tested groups stay fixed at their block-start (t0) point-mass position for the
                    # whole rollout (cheap approximation; the tested cores' own dynamics is what is scored)
                    if k in Hs:
                        pred = np.zeros((len(cores), z0.shape[1]))
                        for ci, a in enumerate(cores):
                            meta_ids = subinfo[int(a)]
                            w = cnt2[meta_ids]
                            pred[ci] = nr((w[:, None] * M2[meta_ids]).sum(0, keepdims=True))[0]
                        res[name][k][0].append(ang(pred, truth[k])); res[name][k][1].append(disp[k])
    R = {"tape": os.path.basename(f)}
    for m, v in res.items():
        R[m] = {f"{5 * k}steps": dict(err=float(np.mean(np.concatenate(e))), disp=float(np.mean(np.concatenate(d))),
                                       skill=float(1 - np.mean(np.concatenate(e)) / np.mean(np.concatenate(d))))
                for k, (e, d) in v.items() if e}
    return R


out = {}
for spec in specs:
    f, block, perm, seed = spec.split(":")
    R = run_tape(f, int(block), int(perm), int(seed))
    out[os.path.basename(f)] = R
    print(json.dumps(R, indent=1)); sys.stdout.flush()

print(json.dumps(out, indent=1))
