import json, glob
disc = [json.load(open(p)) for p in sorted(glob.glob("runs/DISC_*.json"))]
pros = [json.load(open(p)) for p in sorted(glob.glob("sealed/PROS_*.json"))]
def agg(runs):
    modes = runs[0]["trackers"].keys(); out = {}
    for m in modes:
        a = {k: sum(r["trackers"][m][k] for r in runs) for k in ["readouts", "exact_value_mismatch", "sign_mismatch", "refresh", "pass_but_wrong", "fn", "fp"]}
        a["refresh_rate"] = a["refresh"] / max(a["readouts"], 1); a["exact_rate"] = 1 - a["exact_value_mismatch"] / max(a["readouts"], 1)
        nb = {}
        for r in runs:
            for b, c in r["trackers"][m]["sign_mismatch_by_nbin"].items(): nb[b] = nb.get(b, 0) + c
        a["sign_mismatch_by_nbin"] = nb; out[m] = a
    return out
def witness(runs):
    rp = {}
    for r in runs:
        for st, d in r["witness"]["rank_prev_witness"].items():
            rp.setdefault(st, {})
            for k, v in d.items(): rp[st][k] = rp[st].get(k, 0) + v
    return dict(rank_prev_witness=rp, lifetime_quantiles_10_50_90_99=[r["witness"]["lifetime_quantiles"] for r in runs],
                entrants_max_per_step={k: max(r["witness"]["entrants_max"][k] for r in runs) for k in runs[0]["witness"]["entrants_max"]},
                entrants_mean_per_step={k: sum(r["witness"]["entrants_mean"][k] for r in runs) / len(runs) for k in runs[0]["witness"]["entrants_mean"]},
                gap_median_q_k1_minus_q1={k: [r["witness"]["gap_median"][k] for r in runs] for k in runs[0]["witness"]["gap_median"]},
                event=[r["event"] for r in runs])
json.dump(dict(discovery=[dict(block=r["block"], gate=r["gate_audit"], births_max=r["births_per_reassign_max"], lineage=r["lineage"]) for r in disc],
               prospective=[dict(block=r["block"], gate=r["gate_audit"], births_max=r["births_per_reassign_max"], lineage=r["lineage"]) for r in pros]),
          open("EXACT_GATE_AUDIT.json", "w"), indent=1)
json.dump(dict(discovery=witness(disc), prospective=witness(pros)), open("WITNESS_RANK_DYNAMICS.json", "w"), indent=1)
json.dump(dict(discovery=agg(disc)), open("CANDIDATE_STATES_FREEZE.json", "w"), indent=1)
P = agg(pros)
json.dump(dict(prospective=P, runs=[dict(block=r["block"], perm_seed=r["perm_seed"], cfg_seed=r["cfg_seed"]) for r in pros]), open("PROSPECTIVE_RESULTS.json", "w"), indent=1)
json.dump(dict(prospective_sign_mismatch_by_nbin={m: v["sign_mismatch_by_nbin"] for m, v in P.items()},
               event_aware=[r["event"] for r in pros]), open("EVENT_TYPE_FAILURE_AUDIT.json", "w"), indent=1)
for name, A in [("DISCOVERY", agg(disc)), ("PROSPECTIVE", P)]:
    print(name)
    for m, a in A.items(): print(f"  {m:18s} readouts {a['readouts']:6d} exact {a['exact_rate']:.4f} signMis {a['sign_mismatch']:3d} refresh {a['refresh_rate']:.3f} passWrong {a['pass_but_wrong']}")
print(json.dumps(witness(pros), indent=0)[:2500])
