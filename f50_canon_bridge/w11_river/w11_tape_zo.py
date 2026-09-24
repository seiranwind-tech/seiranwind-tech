"""W11 tape recorder ("錄影帶"): runs the frozen engine unmodified and records
  trk[t, w]  = track id of well w after step t (every step)
  Z[f]       = zin (float16) every 5 steps, f = t//5
  C[t]       = engine centres + track ids every 5 steps (for comparison)
Output: npz in $W11_TAPES."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w10_member_motion"))
from w10_common import make_engine, E
block, perm, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); T = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
eng = make_engine(block, perm, seed); N = eng.config.n_wells
trk = np.zeros((T, N), np.int32); Z = []; ZO = []; Cs = {}; t0 = time.time()
for t in range(T):
    eng.step()
    trk[t] = eng.centre_track_ids[eng.labels]
    if (t + 1) % 5 == 0:
        Z.append(eng.zin.astype(np.float16)); ZO.append(eng.zout.astype(np.float16)); Cs[t] = (eng.centre_track_ids.copy(), eng.centres.astype(np.float16).copy())
    if t % 500 == 0: print(t, len(eng.centres), eng.lineage_counts, round(time.time() - t0), flush=True)
out = os.path.join(os.environ["W11_TAPES"], f"tape_{block}_{perm}_{seed}_zo.npz")
ct = np.array(sorted(Cs)); np.savez(out, trk=trk, Z=np.stack(Z), ZO=np.stack(ZO), ctimes=ct,
     ctracks=np.array([Cs[k][0] for k in ct], dtype=object), ccent=np.array([Cs[k][1] for k in ct], dtype=object),
     lineage=str(eng.lineage_counts))
print("done", out, round(time.time() - t0))
