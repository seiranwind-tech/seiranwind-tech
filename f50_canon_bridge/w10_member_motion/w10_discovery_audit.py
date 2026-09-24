"""W10 discovery audit (W9 discovery partition only): dependency graph, force decomposition, sufficiency counterexamples, compression."""
import sys, json, zlib, numpy as np
sys.path.insert(0, "."); from w10_common import *
eng = make_engine(0, 12345, 1); c = eng.config
dep = {"z_w (zin)": "member-level L1", "zout_w, zop_w": "member-level L1 hidden field (own dynamics: blade + salt, wall clip)",
       "teeth_w, mask_w, skew_w": "static but member-level (per-word, N x 4 x 300)", "labels, well kNN": "L1.5-native / static known",
       "salt_w = 0.5(f_salt(z_w) - z_w(z_w.f_salt))": "member-level (needs z_w and teeth_w); not a function of basin moments",
       "wave_w = 0.6(zout_w' - zop_w)/wall": "member-level hidden field; sum over basin is exactly aggregatable ONLY if zout sums are tracked, but zout itself needs member-level blade/salt/clip",
       "A_w = normalize(sum_v softmax_v(z_w.z_v/h) z_v - z_w)": "global-member-kernel (all N wells); normalize per well",
       "z_w+ = normalize(z_w + dt d_w)": "memberwise normalization -> sum does not commute with normalize",
       "m_i+ = sum_{w in i} z_w+": "not exactly aggregatable from (n_i, m_i, any zin moments) — see counterexample P2"}
FD = []; CE = []; comp = []
sh = Shadow(eng); last_obs = None
for _ in range(6000):
    cs = eng.step_number + 2
    sample = cs >= 1500 and cs % 25 == 12      # never a centre-update / gate step
    if sample:
        z0 = eng.zin.copy(); lab = eng.labels.copy(); K = len(eng.centres)
        salt, wave, attr, zplus = decompose(eng)
    # exact W9 shadow driven with H_S (for compression audit)
    if cs % 50 == 0:
        z = eng.zin; cand = sh.labels[sh.nbr]; sim = np.einsum("nd,nkd->nk", z, sh.centres[cand]); ba = sim.argmax(1)
        bs = sim[np.arange(len(z)), ba]; bl = cand[np.arange(len(z)), ba]; own = np.einsum("nd,nd->n", z, sh.centres[sh.labels])
        orph = np.where((bs < sh.thr) & (own < sh.thr))[0][:c.birth_budget_per_reassign]
        box = {}
        def ob(L, K, z=z): o = observed(z, L, K); box["o"] = o; return o
        sh.reassign(bl, bs >= sh.thr, orph, z[orph], ob); last_obs = box["o"]
    elif cs % 5 == 0:
        o = observed(eng.zin, sh.labels, len(sh.centres))
        if cs >= 1500 and cs % 25 == 0 and last_obs is not None and last_obs.shape == o.shape:
            raw = o.astype(np.float32).tobytes(); sums = np.zeros_like(o); np.add.at(sums, sh.labels, eng.zin)
            xor = (np.frombuffer(raw, np.uint32) ^ np.frombuffer(last_obs.astype(np.float32).tobytes(), np.uint32)).tobytes()
            comp.append(dict(step=cs, K=len(o), raw=len(raw), zlib_raw=len(zlib.compress(raw, 6)), zlib_xor_vs_prev=len(zlib.compress(xor, 6)),
                             zlib_unnormalized_m=len(zlib.compress(sums.astype(np.float32).tobytes(), 6))))
        sh.centre_update(o); last_obs = o
    if sample:
        e2 = E.clone_engine_state(eng)
    eng.step()
    if sample:
        z1 = eng.zin; recon_err = float(np.abs(zplus - z1).max())
        n = np.bincount(lab, minlength=K); dm = np.zeros((K, z0.shape[1])); np.add.at(dm, lab, z1 - z0)
        lin = np.zeros_like(dm); np.add.at(lin, lab, c.dt * (salt + wave + c.attract_gain * attr))
        Fs = np.zeros_like(dm); np.add.at(Fs, lab, c.dt * salt); Fw = np.zeros_like(dm); np.add.at(Fw, lab, c.dt * wave)
        Fa = np.zeros_like(dm); np.add.at(Fa, lab, c.dt * c.attract_gain * attr); Rn = dm - lin
        big = n >= 5; nm = lambda a: np.linalg.norm(a[big], axis=1)
        FD.append(dict(step=cs, recon_max_abs_err=recon_err, basins=int(big.sum()),
                       median_norm={"dm": float(np.median(nm(dm))), "salt": float(np.median(nm(Fs))), "wave": float(np.median(nm(Fw))),
                                    "attr": float(np.median(nm(Fa))), "R_norm": float(np.median(nm(Rn)))},
                       median_rel_R_norm=float(np.median(nm(Rn) / np.maximum(nm(dm), 1e-12)))))
        # counterexamples on the largest basin: P2 swap zin of two members (all basin zin moments identical), P1 rotate perp to m
        i = int(np.argmax(n)); mem = np.where(lab == i)[0]
        if len(CE) < 12 and len(mem) >= 4:
            for kind in ("P2_swap_two_members_zin", "P1_rotate_perp_to_m"):
                e3 = E.clone_engine_state(e2); zz = e3.zin.copy()
                if kind.startswith("P2"):
                    a, b = mem[0], mem[-1]; zz[[a, b]] = zz[[b, a]]
                else:
                    m = zz[mem].sum(0); u = np.random.default_rng(cs).normal(size=zz.shape[1]); u -= u @ m / (m @ m) * m; u /= np.linalg.norm(u)
                    v = np.random.default_rng(cs + 1).normal(size=zz.shape[1]); v -= v @ m / (m @ m) * m; v -= (v @ u) * u; v /= np.linalg.norm(v)
                    th = 0.3; pu, pv = zz[mem] @ u, zz[mem] @ v
                    zz[mem] += np.outer(pu * (np.cos(th) - 1) - pv * np.sin(th), u) + np.outer(pu * np.sin(th) + pv * (np.cos(th) - 1), v)
                e3.zin = zz.astype(np.float32); m0a = z0[mem].sum(0); m0b = e3.zin[mem].sum(0)
                M2a = z0[mem].T @ z0[mem]; M2b = e3.zin[mem].T @ e3.zin[mem]
                e3.step(); dmb = (e3.zin[mem] - zz[mem]).sum(0); dma = (z1[mem] - z0[mem]).sum(0)
                CE.append(dict(step=cs, kind=kind, basin_n=int(len(mem)), m_before_diff=float(np.abs(m0a - m0b).max()),
                               M2_before_diff=float(np.abs(M2a - M2b).max()), dm_diff_norm=float(np.linalg.norm(dma - dmb)),
                               dm_norm=float(np.linalg.norm(dma)), rel=float(np.linalg.norm(dma - dmb) / max(np.linalg.norm(dma), 1e-12)),
                               obs_angle_deg=float(np.degrees(np.arccos(np.clip(nr((m0a + dma)[None])[0] @ nr((m0b + dmb)[None])[0], -1, 1))))))
                del e3
        del e2
agg = lambda k: float(np.median([f["median_norm"][k] for f in FD]))
json.dump(dict(nodes=dep, verdict="Delta m_i depends on member-level zin, member-level hidden zout/zop, per-well static teeth, a global N-body kernel, and per-well normalization"),
          open("DEPENDENCY_GRAPH_W10.json", "w"), indent=1)
json.dump(dict(samples=FD, summary={"median_norm": {k: agg(k) for k in ["dm", "salt", "wave", "attr", "R_norm"]},
               "median_rel_R_norm": float(np.median([f["median_rel_R_norm"] for f in FD])), "max_recon_err": float(max(f["recon_max_abs_err"] for f in FD))}),
          open("FORCE_DECOMPOSITION_AUDIT.json", "w"), indent=1)
json.dump(dict(counterexamples=CE), open("MOMENT_SUFFICIENCY_AUDIT.json", "w"), indent=1)
tot = {k: sum(x[k] for x in comp) for k in ["raw", "zlib_raw", "zlib_xor_vs_prev", "zlib_unnormalized_m"]}
json.dump(dict(samples=len(comp), totals=tot, ratios={k: tot["raw"] / max(v, 1) for k, v in tot.items()}, per_sample=comp[:20]),
          open("PAYLOAD_COMPRESSION_AUDIT_DISCOVERY.json", "w"), indent=1)
print(json.dumps(dict(force=json.load(open("FORCE_DECOMPOSITION_AUDIT.json"))["summary"], ce=CE[:6], comp_ratios=tot), indent=1))
