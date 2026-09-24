import sys, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "w1"); import pu_engine as e
def one(s): return e.run(s)
if __name__ == "__main__":
    split, out = sys.argv[1], sys.argv[2]
    seeds = {"dev": [1, 2, 3, 4, 5, 6], "fresh": [101, 102, 103, 104, 105, 106]}[split]
    with Pool(4) as p: res = p.map(one, seeds)
    rows = [x for r, _, _ in res for x in r]
    arr = {c: np.array([r[c] for r in rows]) for c in rows[0]}
    np.savez_compressed(out + "_table.npz", **arr)
    json.dump(dict(events=[x for _, ev, _ in res for x in ev], moves=[x for _, _, m in res for x in m]), open(out + "_events.json", "w"))
    print(split, len(rows), sum(len(ev) for _, ev, _ in res))
