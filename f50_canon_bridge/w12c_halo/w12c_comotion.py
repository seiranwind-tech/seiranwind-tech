"""W12C step 1: co-motion measurement. For each halo well (river home-group size < 5, h=12),
find nearest core by cosine to core centre M_c at each gate. Track, over consecutive gates
(50-step spacing, same anchor convention as w11_decouple.py: fr = g*10+12):
  - angle(halo well, M_c) stability (ring radius)
  - offset stability under parallel transport of M_c's own motion (rigid co-motion test)
  - velocity correlation: cos(tan(dz_halo), tan(dM_c))
  - dwell time: run length of "same nearest core" across gates
Output JSON summary."""
import sys, os, json, collections, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E
nr = E.normalize_rows

f, block, perm, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
gstep = int(sys.argv[5]) if len(sys.argv) > 5 else 1
eng = make_engine(block, perm, seed)  # required only to establish provenance / sha check; unused otherwise
D = np.load(f, allow_pickle=True); trk = D["trk"]; Z = D["Z"]; T, N = trk.shape
G = np.arange(49, T, 50); H = trk[G]; h = 12; g0 = 60

home = H[0].copy(); away = np.zeros(N, np.int32); HOME = [home.copy()]
for g in range(1, len(G)):
    prev, cur = H[g - 1], H[g]; alive = set(np.unique(cur).tolist())
    s = {t: int(np.bincount(cur[prev == t]).argmax()) for t in np.unique(prev).tolist() if t not in alive}
    if s: home = np.array([s.get(x, x) for x in home.tolist()], np.int32)
    back = cur == home; away[back] = 0; away[~back] += 1; sw = away >= h; home[sw] = cur[sw]; away[sw] = 0; HOME.append(home.copy())

def tan_rows(V, M): return V - (np.sum(V * M, 1, keepdims=True)) * M
def rotate_pt(v, Mfrom, Mto):
    """parallel-transport v (tangent at Mfrom, high-dim) along the great circle Mfrom->Mto -> tangent at Mto.
    Standard spherical parallel transport (geodesic plane spanned by Mfrom, Mto); works in any dimension."""
    dot = np.clip(np.sum(Mfrom * Mto, 1), -1, 1); theta = np.arccos(dot)
    sinT = np.sin(theta); small = sinT < 1e-9
    sinT_safe = np.where(small, 1.0, sinT)
    e = (Mto - Mfrom * dot[:, None]) / sinT_safe[:, None]
    a = np.sum(v * e, 1)
    v_perp = v - a[:, None] * e
    transported = a[:, None] * (-Mfrom * sinT[:, None] + e * np.cos(theta)[:, None]) + v_perp
    transported = np.where(small[:, None], v, transported)
    return transported

max_g = min(len(G) - 1, 113)
gates = list(range(g0, max_g, gstep))

# per-gate: nearest core id + M_c + offset (tangent, unnormalized = full tangential residual) for every halo well
prev_state = None  # dict well_idx -> (core_id, M_c, offset_vec, z_w)
angle_series = collections.defaultdict(list)      # well -> list of angle-to-Mc (deg)
same_core_run = collections.defaultdict(int)
dwell_lengths = []
offset_cos_pt = []       # rigid-comotion test: cos(offset(g+1), PT(offset(g)))
vel_cos = []              # velocity correlation
vel_mag_ratio = []
n_halo_frac = []

for gi, g in enumerate(gates):
    lab = HOME[g]; fr = g * 10 + 12
    z = nr(Z[fr].astype(np.float32))
    u, cnt = np.unique(lab, return_counts=True); cores = u[cnt >= 5]; is_core = np.isin(lab, cores)
    halo_idx = np.where(~is_core)[0]
    n_halo_frac.append(float(len(halo_idx)) / N)
    if len(cores) == 0 or len(halo_idx) == 0:
        prev_state = None; continue
    coreM = np.stack([nr(z[lab == a].mean(0, keepdims=True))[0] for a in cores])
    sim = z[halo_idx] @ coreM.T
    nn = np.argmax(sim, 1); nearest_core = cores[nn]
    Mw = coreM[nn]
    ang_deg = np.degrees(np.arccos(np.clip(np.sum(z[halo_idx] * Mw, 1), -1, 1)))
    offset = tan_rows(z[halo_idx], Mw)
    cur_state = {"idx": halo_idx, "core": nearest_core, "M": Mw, "z": z[halo_idx], "offset": offset, "ang": ang_deg, "coreMbyid": dict(zip(cores.tolist(), coreM))}
    for w, cid, a in zip(halo_idx.tolist(), nearest_core.tolist(), ang_deg.tolist()):
        angle_series[w].append(a)
    if prev_state is not None:
        pidx = prev_state["idx"]; pcore = dict(zip(pidx.tolist(), prev_state["core"].tolist()))
        pM = dict(zip(pidx.tolist(), list(prev_state["M"]))); poff = dict(zip(pidx.tolist(), list(prev_state["offset"])))
        pz = dict(zip(pidx.tolist(), list(prev_state["z"])))
        common = [w for w in cur_state["idx"].tolist() if w in pcore]
        cur_core_by_w = dict(zip(cur_state["idx"].tolist(), cur_state["core"].tolist()))
        cur_M_by_w = dict(zip(cur_state["idx"].tolist(), list(cur_state["M"])))
        cur_off_by_w = dict(zip(cur_state["idx"].tolist(), list(cur_state["offset"])))
        cur_z_by_w = dict(zip(cur_state["idx"].tolist(), list(cur_state["z"])))
        for w in common:
            same_core = (pcore[w] == cur_core_by_w[w])
            if same_core:
                same_core_run[w] = same_core_run.get(w, 0) + 1
                Mfrom = pM[w][None]; Mto = cur_M_by_w[w][None]
                off_pt = rotate_pt(poff[w][None], Mfrom, Mto)[0]
                off_now = cur_off_by_w[w]
                nrm = np.linalg.norm(off_pt) * np.linalg.norm(off_now)
                if nrm > 1e-9:
                    offset_cos_pt.append(float(np.dot(off_pt, off_now) / nrm))
                dz = cur_z_by_w[w] - pz[w]; dM = cur_M_by_w[w] - Mfrom[0]
                vz = tan_rows(dz[None], Mfrom)[0]; vm = tan_rows(dM[None], Mfrom)[0]
                nrm2 = np.linalg.norm(vz) * np.linalg.norm(vm)
                if nrm2 > 1e-9:
                    vel_cos.append(float(np.dot(vz, vm) / nrm2))
                    if np.linalg.norm(vm) > 1e-9:
                        vel_mag_ratio.append(float(np.linalg.norm(vz) / np.linalg.norm(vm)))
            else:
                if same_core_run.get(w, 0) > 0:
                    dwell_lengths.append(same_core_run[w])
                same_core_run[w] = 0
    prev_state = cur_state

for w, r in same_core_run.items():
    if r > 0: dwell_lengths.append(r)

def stats(x):
    x = np.array(x, float)
    if len(x) == 0: return None
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), p10=float(np.quantile(x, 0.1)), p90=float(np.quantile(x, 0.9)))

ang_flat = np.concatenate([np.array(v) for v in angle_series.values() if len(v) > 0]) if angle_series else np.array([])
ang_std_per_well = [float(np.std(v)) for v in angle_series.values() if len(v) >= 3]

out = {
    "tape": f.split("/")[-1],
    "n_gates_sampled": len(gates),
    "halo_well_fraction": stats(n_halo_frac),
    "angle_to_nearest_core_deg": stats(ang_flat.tolist()),
    "angle_std_per_well_deg": stats(ang_std_per_well),
    "offset_cos_after_parallel_transport_1gate": stats(offset_cos_pt),
    "velocity_cos_halo_vs_core_1gate": stats(vel_cos),
    "velocity_mag_ratio_halo_over_core": stats(vel_mag_ratio),
    "dwell_time_gates_same_nearest_core": stats(dwell_lengths),
}
print(json.dumps(out, indent=1))
