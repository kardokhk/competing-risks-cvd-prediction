#!/usr/bin/env python
"""Known-answer and internal-consistency tests for src/v2/eval_lib_v2.py.

These are the cheap checks that do not need R. `src/v2/07_validate_metrics.R`
is the external check against riskRegression, pec, geepack and prodlim.

Run: python src/v2/test_eval_lib.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_lib_v2 import (aj_cif_at, pseudo_cif, censor_surv, censor_surv_stratified,
                         calib_intercept_slope, flexible_calibration, ipcw_auc,
                         cr_cindex, net_benefit, net_benefit_all, cloglog,
                         at_risk_fraction)

rng = np.random.default_rng(20260903)
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def approx(a, b, tol=1e-8):
    return abs(a - b) < tol


print("--- Aalen-Johansen, hand-computed example ---")
# 5 subjects. times 1,2,3,4,5; events 1,2,0,1,0
t = np.array([1., 2., 3., 4., 5.]); e = np.array([1, 2, 0, 1, 0])
# k=1: n=5 d=1 e1=1 -> S0=1, inc=1/5=0.2, S1=0.8
# k=2: n=4 d=1 e1=0 -> inc=0,          S2=0.8*0.75=0.6
# k=3: n=3 d=0                          S3=0.6
# k=4: n=2 d=1 e1=1 -> inc=0.6*0.5=0.3, S4=0.3
check("AJ at t=1", approx(aj_cif_at(t, e, 1.0, 1), 0.2))
check("AJ at t=3", approx(aj_cif_at(t, e, 3.0, 1), 0.2))
check("AJ at t=4", approx(aj_cif_at(t, e, 4.0, 1), 0.5))
check("AJ cause 2 at t=2", approx(aj_cif_at(t, e, 2.0, 2), 0.2))
check("AJ before first time is 0", approx(aj_cif_at(t, e, 0.5, 1), 0.0))
check("AJ cause1 + cause2 <= 1", aj_cif_at(t, e, 5.0, 1) + aj_cif_at(t, e, 5.0, 2) <= 1 + 1e-12)

print("--- AJ with no censoring equals the empirical proportion ---")
n = 2000
tt = rng.uniform(0.1, 5, n)
ee = rng.choice([1, 2], n, p=[0.3, 0.7])
check("AJ = empirical proportion when nothing is censored",
      approx(aj_cif_at(tt, ee, 10.0, 1), (ee == 1).mean(), 1e-10),
      f"{aj_cif_at(tt, ee, 10.0, 1):.6f} vs {(ee == 1).mean():.6f}")

print("--- pseudo-values: mean equals the AJ estimate ---")
for n in (200, 3000):
    tt = np.round(rng.uniform(0.1, 12, n) * 12) / 12   # whole months, as in the data
    ee = rng.choice([0, 1, 2], n, p=[0.6, 0.1, 0.3])
    for h in (5.0, 10.0):
        ps = pseudo_cif(tt, ee, h, 1)
        aj = aj_cif_at(tt, ee, h, 1)
        check(f"mean(pseudo) == AJ  (n={n}, t={h:.0f})", approx(float(ps.mean()), aj, 1e-9),
              f"{ps.mean():.10f} vs {aj:.10f}")

print("--- pseudo-values: brute-force leave-one-out on a small sample ---")
n = 60
tt = np.round(rng.uniform(0.1, 8, n) * 12) / 12
ee = rng.choice([0, 1, 2], n, p=[0.5, 0.2, 0.3])
h = 4.0
fast = pseudo_cif(tt, ee, h, 1)
full = aj_cif_at(tt, ee, h, 1)
brute = np.array([n * full - (n - 1) * aj_cif_at(np.delete(tt, i), np.delete(ee, i), h, 1)
                  for i in range(n)])
check("pseudo-values match brute-force leave-one-out",
      np.max(np.abs(fast - brute)) < 1e-9, f"max abs diff {np.max(np.abs(fast - brute)):.2e}")

print("--- censoring KM ---")
t = np.array([1., 2., 3., 4.]); e = np.array([0, 1, 0, 1])
# censor events at t=1 and t=3. G(1-)=1, G(2-)=0.75, G(3.5-)=0.75*(2/3)... check monotone
g = censor_surv(t, e, np.array([0.5, 1.0, 2.0, 3.0, 4.0]))
check("G(s-) is 1 before the first censoring", approx(g[0], 1.0))
check("G(s-) at the first censoring time is still 1", approx(g[1], 1.0))
check("G(s-) is non-increasing", np.all(np.diff(g) <= 1e-12), str(g))
check("G(2-) = 1 - 1/4", approx(g[2], 0.75))

print("--- stratified censoring reduces to marginal when there is one stratum ---")
n = 500
tt = np.round(rng.uniform(0.1, 12, n) * 12) / 12
ee = rng.choice([0, 1, 2], n, p=[0.5, 0.2, 0.3])
s1 = censor_surv(tt, ee, tt)
s2 = censor_surv_stratified(tt, ee, np.zeros(n), tt)
check("stratified == marginal with a single stratum", np.allclose(s1, s2))

print("--- weak calibration: a perfectly calibrated model gives slope 1, intercept 0 ---")
n = 40000
x = rng.normal(size=n)
# true CIF from a cloglog model, no censoring so pseudo-values are the indicator
eta = -3.2 + 0.7 * x
p_true = 1 - np.exp(-np.exp(eta))
y = (rng.uniform(size=n) < p_true).astype(float)
icp, slp = calib_intercept_slope(y, p_true)
check("perfect model: intercept near 0", abs(icp) < 0.05, f"{icp:.4f}")
check("perfect model: slope near 1", abs(slp - 1) < 0.06, f"{slp:.4f}")

print("--- weak calibration detects a known miscalibration ---")
# halve the linear predictor: the fitted slope should be about 2
p_flat = 1 - np.exp(-np.exp(-3.2 + 0.35 * x))
icp2, slp2 = calib_intercept_slope(y, p_flat)
check("under-dispersed predictions give slope near 2", abs(slp2 - 2) < 0.15, f"{slp2:.4f}")
# inflate every risk: intercept should be clearly negative
p_hi = 1 - np.exp(-np.exp(eta + 0.5))
icp3, _ = calib_intercept_slope(y, p_hi)
check("over-predicted risks give a negative intercept", icp3 < -0.3, f"{icp3:.4f}")

print("--- moderate calibration ---")
fc = flexible_calibration(y, p_true, nknots=3, return_curve=True)
check("ICI of a perfectly calibrated model is small", fc["ici"] < 0.004, f"{fc['ici']:.5f}")
check("E50 <= E90", fc["e50"] <= fc["e90"] + 1e-12)
fc2 = flexible_calibration(y, p_hi, nknots=3)
check("ICI increases under over-prediction", fc2["ici"] > fc["ici"], f"{fc2['ici']:.5f}")

print("--- AUC ---")
n = 8000
tt = np.round(rng.uniform(0.1, 15, n) * 12) / 12
ee = rng.choice([0, 1, 2], n, p=[0.5, 0.2, 0.3])
G_T = censor_surv(tt, ee, tt)
G_t = censor_surv(tt, ee, np.full(n, 10.0))
noise = rng.uniform(size=n)
check("AUC of pure noise is near 0.5",
      abs(ipcw_auc(tt, ee, noise, 10.0, G_T, G_t) - 0.5) < 0.03,
      f"{ipcw_auc(tt, ee, noise, 10.0, G_T, G_t):.4f}")
perfect = np.where((ee == 1) & (tt <= 10), 0.9, 0.1) + 1e-6 * rng.uniform(size=n)
check("AUC of a perfect marker is 1",
      ipcw_auc(tt, ee, perfect, 10.0, G_T, G_t) > 0.999,
      f"{ipcw_auc(tt, ee, perfect, 10.0, G_T, G_t):.5f}")
check("reversing the marker gives 1 - AUC",
      approx(ipcw_auc(tt, ee, -noise, 10.0, G_T, G_t),
             1 - ipcw_auc(tt, ee, noise, 10.0, G_T, G_t), 1e-9))

print("--- C-index ---")
c_noise = cr_cindex(tt, ee, noise, 10.0, G_T)
check("C of pure noise is near 0.5", abs(c_noise - 0.5) < 0.03, f"{c_noise:.4f}")
c_perf = cr_cindex(tt, ee, perfect, 10.0, G_T)
check("C of a perfect marker is high", c_perf > 0.95, f"{c_perf:.4f}")
c_1024 = cr_cindex(tt, ee, noise, 10.0, G_T, nbins=1024)
c_8192 = cr_cindex(tt, ee, noise, 10.0, G_T, nbins=8192)
check("C is stable in the number of risk bins", abs(c_1024 - c_8192) < 1e-3,
      f"1024 bins {c_1024:.6f}, 8192 bins {c_8192:.6f}")

print("--- net benefit ---")
tt = np.round(rng.uniform(0.1, 15, 5000) * 12) / 12
ee = rng.choice([0, 1, 2], 5000, p=[0.5, 0.1, 0.4])
pred = rng.uniform(0.01, 0.2, 5000)          # every prediction exceeds 0.001
f_all = aj_cif_at(tt, ee, 10.0, 1)
nb_lo = net_benefit(tt, ee, pred, 0.001, 10.0)
nb_all = net_benefit_all(tt, ee, 0.001, 10.0)
check("net benefit at a threshold below every prediction equals treat-all",
      approx(nb_lo, nb_all, 1e-9), f"{nb_lo:.8f} vs {nb_all:.8f}")
check("net benefit is 0 when nobody is above threshold",
      approx(net_benefit(tt, ee, pred, 0.99, 10.0), 0.0))
check("treat-all formula CIF - (1-CIF)*pt/(1-pt)",
      approx(net_benefit_all(tt, ee, 0.075, 10.0),
             f_all - (1 - f_all) * 0.075 / 0.925, 1e-12))
# treat-all is positive exactly when the observed CIF exceeds the threshold; the
# sign of the treat-all reference is what version 1's docs/table1.md got wrong.
check("sign of treat-all follows CIF vs threshold, above",
      (net_benefit_all(tt, ee, 0.5 * f_all, 10.0) > 0),
      f"CIF {f_all:.4f}, NB_all at pt={0.5*f_all:.4f} is "
      f"{net_benefit_all(tt, ee, 0.5 * f_all, 10.0):.5f}")
check("sign of treat-all follows CIF vs threshold, below",
      (net_benefit_all(tt, ee, min(0.9, 2 * f_all), 10.0) < 0),
      f"CIF {f_all:.4f}, NB_all at pt={min(0.9, 2*f_all):.4f} is "
      f"{net_benefit_all(tt, ee, min(0.9, 2 * f_all), 10.0):.5f}")
check("treat-all is exactly 0 at pt equal to the observed CIF",
      approx(net_benefit_all(tt, ee, f_all, 10.0), 0.0, 1e-12))

print("--- at-risk fraction ---")
t = np.array([1., 12., 3., 20.]); e = np.array([0, 0, 1, 2])
check("at-risk fraction at 10 y", approx(at_risk_fraction(t, e, 10.0), 0.75),
      str(at_risk_fraction(t, e, 10.0)))

print("--- cloglog round trip ---")
p = np.array([0.001, 0.05, 0.5, 0.9])
check("cloglog inverse", np.allclose(1 - np.exp(-np.exp(cloglog(p))), p))

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("all eval_lib_v2 self-tests passed")
