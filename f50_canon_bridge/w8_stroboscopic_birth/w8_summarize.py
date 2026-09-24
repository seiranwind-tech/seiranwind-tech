import json, glob, hashlib
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
disc = json.load(open("runs/DISC.json")); pros = [json.load(open(p)) for p in sorted(glob.glob("prospective/P_*.json"))]
KEYS = ["gates", "basin_readouts", "sign_mismatch", "count_mismatch_engine_counter", "near_miss_mismatch", "births_count_mismatch",
        "identity_violation", "budget_hit", "budget_selection_mismatch", "birth_basins", "multi_birth_basins", "extra_high_dim_ops_for_handoff"]
def agg(runs):
    a = {k: sum(r["results"][k] for r in runs) for k in KEYS}
    a["max_NB"] = max(r["results"]["max_NB"] for r in runs); a["max_global_orphans"] = max(r["results"]["max_global_orphans"] for r in runs)
    a["overflow_gates"] = {k: sum(r["results"]["overflow"][k] for r in runs) for k in runs[0]["results"]["overflow"]}
    a["overflow_basins"] = {k: sum(r["results"]["overflow_basins"][k] for r in runs) for k in runs[0]["results"]["overflow_basins"]}
    h = {}
    for r in runs:
        for m, c in r["results"]["multiplicity_hist"].items(): h[m] = h.get(m, 0) + c
    a["multiplicity_hist"] = dict(sorted(h.items(), key=lambda x: int(x[0])))
    pb = {k: sum(r["results"]["payload_bytes"][k] for r in runs) for k in runs[0]["results"]["payload_bytes"]}
    a["payload_bytes_total"] = pb; a["payload_bytes_per_gate"] = {k: v / max(a["gates"], 1) for k, v in pb.items()}
    rs = {k: sum(r["results"]["reset_seed"][k] for r in runs) for k in runs[0]["results"]["reset_seed"]}
    a["reset_seed"] = rs; a["reset_seed_hit_rate"] = rs["argmin_in_stale_bottom16_or_entrants"] / max(rs["total"], 1)
    return a
P = agg(pros); D = agg([disc])
json.dump(dict(engine_sha="4a71aeb4a66a8a071a6b6f9a9b065687d237909aea5e89a537ee6d80d2aa3b1d", glove_sha="a0997bb3a3866b4b48fc44503969457c57379ff03cca03ab289f2f07c4948bdc",
               protocol_sha=sha("PROTOCOL_W8.md"), harness_sha=sha("w8_harness.py"), freeze_lock_sha=sha("W8_FREEZE_LOCK.txt"),
               runs=[dict(block=r["block"], perm_seed=r["perm_seed"], cfg_seed=r["cfg_seed"], lineage=r["lineage"]) for r in [disc] + pros]),
          open("PRECHECK_W8.json", "w"), indent=1)
json.dump(dict(base_arrays_used=["own (N×d dot, computed in _reassign_sparse)", "best_score (N×k×d einsum, computed in _reassign_sparse)", "labels"],
               handoff_ops=["q = max(own, best)  O(N)", "segmented min / count over labels  O(N)", "orphan list = where(q<thr)  O(N)"],
               NO_ADDITIONAL_HIGH_DIMENSIONAL_SCAN=True, orphan_condition_identity="orphan == (best<thr)&(own<thr) == max(own,best)<thr (asserted every gate)",
               engine_counter_checks=dict(orphan_candidates_mismatch=P["count_mismatch_engine_counter"], near_miss_mismatch=P["near_miss_mismatch"]),
               note="harness recomputes own/best for audit only; engine counters _birth_orphan_candidates/_birth_near_misses match -> same arrays"),
          open("BASE_SCAN_DEPENDENCY_AUDIT.json", "w"), indent=1)
json.dump(dict(prospective={k: P[k] for k in ["birth_basins", "multi_birth_basins", "max_NB", "max_global_orphans", "multiplicity_hist", "budget_hit"]},
               discovery={k: D[k] for k in ["birth_basins", "multi_birth_basins", "max_NB", "max_global_orphans", "multiplicity_hist", "budget_hit"]}),
          open("BIRTH_MULTIPLICITY_AUDIT.json", "w"), indent=1)
json.dump(dict(H1="q_min per basin (float32)", H2="(q_min float32, N_B uint16)", H3_k="bottom-k (q,id,own) + q_(k+1), k in {8,16,32,64}",
               H4="q_min per basin + (id,own) per orphan", threshold=0.9, budget=256, frozen_in="W8_FREEZE_LOCK.txt"),
          open("HANDOFF_FREEZE.json", "w"), indent=1)
json.dump(dict(prospective=P, discovery=D, per_run=[dict(perm_seed=r["perm_seed"], block=r["block"], cfg_seed=r["cfg_seed"], **{k: r["results"][k] for k in KEYS}) for r in pros]),
          open("PROSPECTIVE_W8_RESULTS.json", "w"), indent=1)
json.dump(dict(bytes_per_gate=P["payload_bytes_per_gate"], total_bytes=P["payload_bytes_total"], gates=P["gates"],
               overflow_gates=P["overflow_gates"], overflow_basins=P["overflow_basins"]), open("PAYLOAD_COST_AUDIT.json", "w"), indent=1)
json.dump(dict(reset_seed_prospective=P["reset_seed"], hit_rate=P["reset_seed_hit_rate"],
               definition="argmin q at t_r+1 contained in (pre-reassign bottom-16 from base scan) ∪ (wells whose track changed at the reassign)",
               exact_birth_path_needs_reset=False, note="exact birth gate uses only the base scan at each t_r; the bottom-k reset is only for the early-warning tracker"),
          open("POST_REASSIGN_RESET_AUDIT.json", "w"), indent=1)
print(json.dumps({k: P[k] for k in KEYS + ["max_NB", "max_global_orphans", "multiplicity_hist", "overflow_gates", "payload_bytes_per_gate", "reset_seed_hit_rate"]}, indent=1))
