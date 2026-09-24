import json, glob, os, numpy as np
R = {}; J = lambda p: json.load(open(p))
tapes = sorted({os.path.basename(p)[len("river_"):-5] for p in glob.glob("runs/river_*.json")})
R["tapes"] = tapes
R["A_birth_and_excursion"] = {t: {k: J(f"runs/replay_{t}.json")[k] for k in ["K_first_last", "births_total", "birth_fate_counts", "birth_size_quantiles_p50_p90_max", "birth_life_gates_quantiles_p50_p90_p99", "excursions_subsampled"]} for t in tapes}
R["B_river_events_per_gate"] = {t: {"raw": J(f"runs/river_{t}.json")["raw"]["events_per_gate"], **{k: v["events_per_gate"] for k, v in J(f"runs/river_{t}.json").items() if k.startswith("river_")}} for t in tapes}
R["C_macro_event_taxonomy_h12"] = {t: {k: v for k, v in J(f"runs/macro_events_{t}.json").items() if k not in ("tape", "macro_K_first_last_late")} for t in tapes}
ar = J("runs/ar_law.json"); R["D_universal_AR_law"] = {k: {"coef": v["coef"], **{n: {h: round(x["skill"], 3) for h, x in s.items()} for n, s in v.items() if n != "coef"}} for k, v in ar.items() if k in ("AR1", "AR3", "AR8")}
zo = [t for t in tapes if t.endswith("_zo")]
R["E_wave_clock"] = {t: {k: J(f"runs/wave_drive_{t}.json")[k] for k in ["R2_wave_only_betafit", "cos_D_W_median"]} for t in zo}
R["F_meanfield"] = {t: {k: J(f"runs/meanfield_{t}.json")[k] for k in ["R2_exact_vs_actual", "R2_salt_mf", "cos_mfsalt_salt", "R2_attr_mf", "cos_mfattr_attr"]} for t in zo}
R["G_balance"] = {t: {k: J(f"runs/balance_{t}.json")[k] for k in ["S", "A", "Dm", "cosSA", "spread", "fit_D_eq_c_SMF"]} for t in zo}
R["H_rollouts"] = {t: {m: {h: round(x["skill"], 3) for h, x in v.items()} for m, v in J(f"runs/rollout_{t}.json").items() if m.startswith("L")} for t in zo}
R["I_slaved_internal_shape"] = {t: json.loads(open(f"runs/slaved_{t}.txt").read().split("\n{\n", 1)[1].join(["{\n", ""]) if False else "{" + open(f"runs/slaved_{t}.txt").read().rsplit("\n{", 1)[1])["median"] for t in zo}
R["J_core_halo_decoupling"] = {t: J(f"runs/decouple_{t}.json") for t in zo}
R["K_environment_representation"] = {t: {k: v for k, v in J(f"runs/env_representation_{t}.json").items() if k != "tape"} for t in zo}
R["L_internal_shape_dimension"] = {t: {k: v for k, v in J(f"runs/shape_pc_{t}.json").items() if k != "tape"} for t in zo}
json.dump(R, open("W11_RESULTS.json", "w"), indent=1)
for k in ["B_river_events_per_gate", "C_macro_event_taxonomy_h12", "E_wave_clock", "F_meanfield", "H_rollouts", "I_slaved_internal_shape"]: print(k, json.dumps(R[k])[:1500])
for t in zo:
    print(t, {k: (v["cos_median"], v["relerr_median"]) for k, v in R["K_environment_representation"][t].items() if k.startswith("m1_") or k.startswith("m20_")})
    print(t, {k: (v["cos_median"], v["relerr_median"]) for k, v in R["L_internal_shape_dimension"][t].items() if k.startswith("k0_") or k.startswith("k32_") or k.startswith("k8_")})
