# W5 — Birth precursor curves and the tail power law (canonical engine, spaCy-md vectors)

These are the precursor curves the user proposed. They were measured on DEV (blocks 0–3); hidden quantities were used only as axes and targets.

**1. Tail collapse.**
(ȳ_own − y_min) ≈ 1.075·σ_own·sqrt(2 ln n), with corr 0.907.
The ratio U = (ȳ − y_min)/(σ·sqrt(2 ln n)) has median 0.94, but it drifts with n: 0.90 at n = 3–5, 1.28 at n = 20–50, and 1.69 for n > 200. So sqrt(2 ln n) is the wrong size scaling.
A log fit gives **ȳ − y_min = e^0.140 · σ^1.058 · n^0.345**. The tail is heavier than Gaussian and scales roughly like n^(1/3).

**2. Approach to the wall.**
Aligned on the first birth crossing (955 events), the margin G = q_min − 0.9 falls near-linearly from 0.022 (at t = −50) to 0 (at t = 0).
The speed is v ≈ 4–6×10⁻⁴ per step, with no acceleration. U stays at about 0.93 throughout; R falls from 0.974 to 0.947, and σ rises from 0.0144 to 0.0205.

**3. Time to birth (hidden G).**
T = G/v gives a median ratio to the actual time of 0.94, with 66% of cases within a factor of 2.

## F3 (preregistered in fit_f3.py, fit on DEV, frozen as F3_FREEZE sha 40eefb88…)
- **F3_POWER_1P:** Γ̂ = 0.9 − ȳ_own + 0.7065·σ_own·n^0.3447. It has one fitted amplitude plus an exponent taken from the DEV log-law.
- **F3_POWER_REG:** Γ̂ = 0.8919 − 0.9894·ȳ_own + 0.7318·σ_own·n^0.3447 − 0.0349·σ_best·n^0.3447. The free regression recovers intercept ≈ 0.9 and a coefficient on ȳ of ≈ −1, i.e. the physical law.

## FRESH2 (untouched vector blocks 8–9: 340,580 rows, 777 births)

| family | #coef | redraw precision | redraw recall | redraw corr | window recall | window precision | median abs Δt | P90 abs Δt | abs Δt ≤ 10 |
|---|---|---|---|---|---|---|---|---|---|
| F1_LEGAL | 7 | 0.716 | 0.609 | 0.933 | 0.535 | 0.763 | 15 | 34 | 0.39 |
| F2_AUG | 10 | 0.694 | 0.806 | 0.946 | 0.728 | 0.741 | 7 | 26 | 0.63 |
| **F3_POWER_1P** | **1 (+γ)** | 0.646 | **0.826** | **0.950** | **0.838** | **0.795** | **6** | 25 | **0.68** |
| F3_POWER_REG | 4 | 0.651 | 0.785 | 0.951 | 0.807 | 0.803 | 7 | 24 | 0.66 |

The legal-state time-to-birth estimate, T = Γ̂/v̂ from F3_REG, is biased. Its median ratio to the actual time is 2.15 (it predicts late), and only 36% of cases fall within a factor of 2. So timing needs a better velocity estimate.

## Result
**Γ̂_birth = 0.9 − ȳ_own + β·σ_own·n^γ, with β ≈ 0.71 and γ ≈ 0.345 (≈ 1/3).**
This is a physical, almost parameter-free tail law. It beats the 10-coefficient F2 on untouched data.
The observable L1.5 still lacks is the scalar projected second moment σ_own. It does **not** need the full 300×300 M2.

Caveats: the vectors are spaCy-md, not GloVe. The GloVe rerun is pending the file transfer. This is not a contract result.
