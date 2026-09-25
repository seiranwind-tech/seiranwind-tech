"""W13A: can a core's own internal pull (attraction) be computed from a LOW-DIMENSIONAL
sub-bunch (次生葡萄串) description of the core's OWN members, instead of individual member
positions?

Environment (unchanged, closed per W11/W12): other cores as point masses (M_b, n_b) + halo
wells kept individually.

Candidates for the core's OWN members, replacing the n exact member positions:
  A  spherical k-means sub-bunches, k in {2,4,8,16,32}: each sub-bunch j is a point mass at
     centroid C_j with weight w_j = member count. Attraction is evaluated AT each centroid
     (as a stand-in for every member assigned to it), with the OWN population being the
     k centroids themselves, self-weight corrected to (w_j - 1) at the diagonal (excluding
     the querying member itself). The core force estimate is the count-weighted average
     over the k sub-bunch centroid forces: A_est = sum_j (w_j/n) * attr(C_j; pop).
  B  the ENGINE's own raw instantaneous track/group id (trk at the sampled gate) as the
     sub-bunch label, instead of k-means. (Checked separately: this rarely subdivides a
     river core -- see report. Included for completeness / negative result.)
  C  A + a per-sub-bunch spread scalar (mean 1-cos to centroid). At evaluation time each
     sub-bunch is expanded into S synthetic members by an isotropic tangent perturbation of
     that RMS magnitude, renormalized; the per-synthetic-member attraction is computed
     against the same population and averaged, to approximate E[nr(attraction)] over the
     cluster instead of nr(E[attraction]) at a single point.

Salt uses the already-closed S_MF mean-field form (every member placed at the group centre
M_a, own teeth kept) in the estimate; the FULL baseline uses the exact per-member salt.
This isolates what the OWN-member reduction costs on the attraction term, on top of the
already-accepted S_MF closure.

Usage: w13a_test.py <tapefile> <block> <perm> <seed> <candidate:A|B|C> <ks csv, e.g. 2,4,8,16,32> [S_synth for C]
"""
import sys, os, json, collections, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E

nr = E.normalize_rows

f, block, perm, seed, cand = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ks = [int(x) for x in sys.argv[6].split(",")]
S_synth = int(sys.argv[7]) if len(sys.argv) > 7 else 8

eng = make_engine(block, perm, seed)
c = eng.config
bw = eng.bandwidth
D = np.load(f, allow_pickle=True, mmap_mode=None)
trk = D["trk"]
Z = D["Z"]
T, N = trk.shape
G = np.arange(49, T, 50)
H = trk[G]
h = 12
g0 = 60
home = H[0].copy()
away = np.zeros(N, np.int32)
HOME = [home.copy()]
for g in range(1, len(G)):
    prev, cur = H[g - 1], H[g]
    alive = set(np.unique(cur).tolist())
    s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
    if s:
        home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
    back = cur == home
    away[back] = 0
    away[~back] += 1
    sw = away >= h
    home[sw] = cur[sw]
    away[sw] = 0
    HOME.append(home.copy())

teeth, mask, skew = eng.teeth, eng.mask, eng.skew


def salt_sub(z, idx):
    zn = nr(z)
    aff = np.einsum("ntd,nd->nt", teeth[idx], zn)
    aff = np.where(mask[idx] > 0, aff, -1e9)
    p = np.exp(3.0 * aff)
    p /= p.sum(1, keepdims=True) + 1e-9
    up = np.triu(skew[idx], 1)
    sym = up + up.transpose(0, 2, 1)
    inner = np.einsum("nt,ntd->nd", p * (1.0 + c.salt * np.einsum("nab,nb->na", sym, p)), teeth[idx])
    return 0.5 * (inner - z * np.sum(z * inner, 1, keepdims=True))


def attr(zg, pop, logw):
    sim = zg @ pop.T
    sim[np.arange(len(zg)), np.arange(len(zg))] = -1e30
    L = sim / bw + logw[None]
    L -= L.max(1, keepdims=True)
    w = np.exp(L)
    w /= w.sum(1, keepdims=True) + 1e-9
    return c.attract_gain * nr(w @ pop - zg)


def attr_query(zq, pop, logw):
    """attraction for a set of queries zq against a SEPARATE population pop (no self-exclusion needed
    other than what is already encoded in logw); zq is not assumed to be a row of pop."""
    sim = zq @ pop.T
    L = sim / bw + logw[None]
    L -= L.max(1, keepdims=True)
    w = np.exp(L)
    w /= w.sum(1, keepdims=True) + 1e-9
    return c.attract_gain * nr(w @ pop - zq)


def kmeans(x, k, it=15):
    k = min(k, len(x))
    rng = np.random.default_rng(0)
    C = x[rng.choice(len(x), k, replace=False)]
    for _ in range(it):
        a = np.argmax(x @ C.T, 1)
        C = np.stack([nr(x[a == j].mean(0, keepdims=True))[0] if (a == j).any() else C[j] for j in range(k)])
    a = np.argmax(x @ C.T, 1)
    W = np.bincount(a, minlength=k).astype(float)
    keep = W > 0
    return C[keep], W[keep], a


def tan(v, M):
    return v - np.dot(v, M) * M


def cosv(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def subbunch_attr_estimate(C, W, halo, P, Wl_other, rng, spread=None):
    """C: (k,d) centroids, W: (k,) counts (sum=n) of OWN members.
    halo: (h,d) exact halo positions. P: (m,d) other-core centroids. Wl_other: (m,) their log-weights.
    Returns count-weighted average attraction over the k sub-bunch queries (or over S synthetic
    members per sub-bunch if spread is not None), i.e. the A_a attraction estimate (d,)."""
    k = len(C)
    n = W.sum()
    env_pop = np.concatenate([P, halo]) if len(P) or len(halo) else np.zeros((0, C.shape[1]), np.float32)
    env_lw = np.concatenate([Wl_other, np.zeros(len(halo))]) if len(P) or len(halo) else np.zeros(0)

    if spread is None:
        # queries = the k centroids themselves; self-cluster weight corrected to (w_j - 1)
        pop = C
        logW = np.log(np.maximum(W, 1e-9))
        L_self_diag = np.where(W > 1, np.log(np.maximum(W - 1, 1e-9)), -1e30)
        sim = C @ pop.T
        L = sim / bw + logW[None]
        L[np.arange(k), np.arange(k)] = L_self_diag
        if len(env_pop):
            sim_env = C @ env_pop.T
            L = np.concatenate([L, sim_env / bw + env_lw[None]], axis=1)
            full_pop = np.concatenate([pop, env_pop])
        else:
            full_pop = pop
        L -= L.max(1, keepdims=True)
        w = np.exp(L)
        w /= w.sum(1, keepdims=True) + 1e-9
        est_per_j = c.attract_gain * nr(w @ full_pop - C)
        return (W[:, None] * est_per_j).sum(0) / n
    else:
        # C: k centroids, spread[j] = RMS tangent offset magnitude (1-cos based). Expand each
        # sub-bunch into S_synth synthetic members via an isotropic tangent perturbation, matched
        # in expected squared magnitude, renormalize, evaluate attraction, average.
        acc = np.zeros(C.shape[1])
        for j in range(k):
            m = int(W[j])
            if m <= 0:
                continue
            sig = spread[j]
            if sig <= 1e-9 or m == 1:
                # degenerate: no useful spread info (or only ever 1 member observed) -> just use centroid,
                # self-weight (w_j - 1) among own synth samples not meaningful; fall back to point estimate
                pop = np.concatenate([C[np.arange(k) != j], env_pop]) if k > 1 else env_pop
                lw = np.concatenate([np.log(np.maximum(W[np.arange(k) != j], 1e-9)), env_lw]) if k > 1 else env_lw
                if len(pop) == 0:
                    continue
                est = attr_query(C[j:j + 1], pop, lw)[0]
                acc += W[j] * est
                continue
            noise = rng.standard_normal((S_synth, C.shape[1]))
            noise = noise - (noise @ C[j])[:, None] * C[j]
            noise *= sig / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-12)
            synth = nr(C[j][None] + noise)
            # own-cluster population for these synthetic members: the OTHER (m-1) same-cluster members are
            # unobserved individually -> approximate them at the centroid with weight (m-1), plus other
            # clusters at their centroids, plus environment.
            other_own = np.concatenate([C[np.arange(k) != j], C[j:j + 1]]) if k > 1 else C[j:j + 1]
            other_w = np.concatenate([W[np.arange(k) != j], [max(m - 1, 1e-9)]]) if k > 1 else np.array([max(m - 1, 1e-9)])
            pop = np.concatenate([other_own, env_pop]) if len(env_pop) else other_own
            lw = np.concatenate([np.log(other_w), env_lw]) if len(env_pop) else np.log(other_w)
            est = attr_query(synth, pop, lw).mean(0)
            acc += W[j] * est
        return acc / n


res = collections.defaultdict(list)
rng = np.random.default_rng(1234)
for g in range(g0, len(G) - 1, 4):
    fr = g * 10 + 12
    z = nr(Z[fr].astype(np.float32))
    lab = HOME[g]
    cur = H[g]
    u, cnt = np.unique(lab, return_counts=True)
    cores = u[cnt >= 10]
    allc = u[cnt >= 5]
    is_core = np.isin(lab, allc)
    halo = z[~is_core]
    Mc = {b: nr(z[lab == b].mean(0, keepdims=True))[0] for b in allc}
    nc = {b: int((lab == b).sum()) for b in allc}
    for a in cores:
        idx = np.where(lab == a)[0]
        n = len(idx)
        M = Mc[a]
        zg = z[idx]
        sa_full = salt_sub(zg, idx)
        order = np.concatenate([idx, np.setdiff1d(np.arange(N), idx)])
        full = tan((sa_full + attr(zg, z[order], np.zeros(N))).mean(0), M)

        # S_MF salt (own members replaced by M for the salt calc; teeth kept per member)
        zm = np.tile(M, (n, 1)).astype(np.float32)
        sa_mf = salt_sub(zm, idx)
        sa_mf_mean = tan(sa_mf.mean(0), M)

        oth = [b for b in allc if b != a]
        P = np.stack([Mc[b] for b in oth]) if oth else np.zeros((0, z.shape[1]), np.float32)
        Wl_other = np.log([nc[b] for b in oth]) if oth else np.zeros(0)

        key = "n10-29" if n < 30 else "n30-99" if n < 100 else "n100+"

        if cand == "B":
            rawids, inv = np.unique(cur[idx], return_inverse=True)
            C = np.stack([nr(zg[inv == j].mean(0, keepdims=True))[0] for j in range(len(rawids))])
            W = np.bincount(inv).astype(float)
            attr_est = subbunch_attr_estimate(C, W, halo, P, Wl_other, rng)
            est = tan(sa_mf_mean + attr_est, M)
            res[("B", len(rawids), key)].append((cosv(est, full), float(np.linalg.norm(est - full) / np.linalg.norm(full))))
            continue

        for k in ks:
            C, W, a_lab = kmeans(zg, k)
            if cand == "A":
                attr_est = subbunch_attr_estimate(C, W, halo, P, Wl_other, rng)
            else:  # C: centroid + spread scalar
                # recompute cluster assignment directly against the (possibly filtered) centroids C
                assign = np.argmax(zg @ C.T, 1)
                spread = np.zeros(len(C))
                for j in range(len(C)):
                    mem = zg[assign == j]
                    if len(mem):
                        spread[j] = float(np.sqrt(max(0.0, 1 - np.mean(mem @ C[j]))))
                attr_est = subbunch_attr_estimate(C, W, halo, P, Wl_other, rng, spread=spread)
            est = tan(sa_mf_mean + attr_est, M)
            res[(cand, k, key)].append((cosv(est, full), float(np.linalg.norm(est - full) / np.linalg.norm(full))))

out = {"tape": f.split("/")[-1], "candidate": cand}
for key3, v in sorted(res.items(), key=lambda kv: str(kv[0])):
    a = np.array(v)
    tagk = key3[1]
    bucket = key3[2]
    out[f"{cand}_k{tagk}_{bucket}"] = dict(count=len(v), cos_median=round(float(np.median(a[:, 0])), 5),
                                            cos_p10=round(float(np.quantile(a[:, 0], 0.1)), 4),
                                            relerr_median=round(float(np.median(a[:, 1])), 4))
print(json.dumps(out, indent=1))
