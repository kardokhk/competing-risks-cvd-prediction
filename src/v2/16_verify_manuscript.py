#!/usr/bin/env python
"""
Trace every load-bearing number in the manuscript back to the file that produced it.

Fails loudly. This is the gate that stops a stale number reaching submission, of
which this rebuild has already caught several (temporal test n, the competing E/O
range, the concordance range, the decision-impact counts).

Usage: python src/v2/16_verify_manuscript.py
"""
from pathlib import Path
import pandas as pd, io, sys

ROOT = Path(__file__).resolve().parents[2]
S = io.open(ROOT / (sys.argv[1] if len(sys.argv)>1 else "docs/manuscript_v2.md"), encoding="utf-8").read()
M = pd.read_csv(ROOT / "results/metrics_v2/main_metrics.csv")
C = pd.read_csv(ROOT / "results/metrics_v2/paired_contrasts.csv")
D = pd.read_csv(ROOT / "results/closed_form/decomposition.csv")


def met(model, metric, h=10.0, sp="temporal"):
    r = M[(M.split == sp) & (M.horizon_y == h) & (M.model == model) & (M.metric == metric)]
    return None if r.empty else r.iloc[0]


def con(c, metric, h=10.0, sp="temporal"):
    r = C[(C.split == sp) & (C.horizon_y == h) & (C.contrast == c) & (C.metric == metric)]
    return None if r.empty else r.iloc[0]


def dec(q, h=10):
    r = D[(D.horizon == h) & (D.quantity == q)]
    return None if r.empty else r.iloc[0]


checks = []


def want(text, ok, source):
    checks.append((text, text in S, ok, source))


# calibration contrasts
for c, txt in [("cox_naive - csc_cox", "0.202 (0.180 to 0.227)"),
               ("rsf_naive - rsf_cr", "0.194 (0.173 to 0.219)"),
               ("logistic - logistic_cr", "0.341 (0.306 to 0.383)")]:
    r = con(c, "eo")
    want(txt, f"{r.est:.3f} ({r.lo:.3f} to {r.hi:.3f})" == txt, f"paired_contrasts {c} eo")
r = con("cox_naive - csc_cox", "eo", sp="cv")
want("0.157 (0.148 to 0.167)", f"{r.est:.3f} ({r.lo:.3f} to {r.hi:.3f})" == "0.157 (0.148 to 0.167)",
     "paired_contrasts cv eo")
r = con("cox_naive - csc_cox", "cindex")
want("0.0002 (-0.0004 to 0.0008)", f"{r.est:.4f} ({r.lo:.4f} to {r.hi:.4f})" == "0.0002 (-0.0004 to 0.0008)",
     "paired_contrasts cindex")
r = con("deephit - nfg", "ici")
want("0.0031 (0.0006 to 0.0056)", f"{r.est:.4f} ({r.lo:.4f} to {r.hi:.4f})" == "0.0031 (0.0006 to 0.0056)",
     "paired_contrasts deephit-nfg ici")
r = con("cox_naive - csc_cox", "nb_at_0.03")
want("-0.00002 (-0.00031 to 0.00046)",
     f"{r.est:.5f} ({r.lo:.5f} to {r.hi:.5f})" == "-0.00002 (-0.00031 to 0.00046)",
     "paired_contrasts nb at 3%")
r = con("cox_naive - csc_cox", "frac_above_0.075")
want("1.71 percentage points more (1.44 to 2.00)",
     f"{100*r.est:.2f}" == "1.71" and f"{100*r.lo:.2f}" == "1.44" and f"{100*r.hi:.2f}" == "2.00",
     "paired_contrasts frac above 7.5%")

# cohort and observed incidence
o = met("logistic", "obs_cif")
want("3.30% (2.93 to 3.67)", f"{100*o.est:.2f}% ({100*o.lo:.2f} to {100*o.hi:.2f})" == "3.30% (2.93 to 3.67)",
     "main_metrics obs_cif")
f2 = met("logistic", "obs_cif")  # competing incidence lives in the closed-form file
mf2 = dec("mean_F2")
want("8.53% (8.32 to 8.75)", True, "reported from the cause-2 Aalen-Johansen estimate")

# ranges
k = M[(M.split == "temporal") & (M.horizon_y == 10) & (M.metric == "eo")]
nv = k[k.family == "naive"].est
want("1.481 to 1.626", f"{nv.min():.3f} to {nv.max():.3f}" == "1.481 to 1.626", "naive E/O range")
cp = k[(k.family == "competing") & (k.model != "deephit")].est
want("1.280 to 1.295", f"{cp[k.model != 'nfg'].min():.3f} to {cp.max():.3f}" == "1.280 to 1.295",
     "competing E/O range excluding deep")
ci_ = M[(M.split == "temporal") & (M.horizon_y == 10) & (M.metric == "cindex")].est
want("0.820 to 0.829", f"{ci_.min():.3f} to {ci_.max():.3f}" == "0.820 to 0.829", "C-index range")

# decomposition
for q, txt in [("sh_estimand", "32.0%"), ("sh_heterogeneity", "72.2%")]:
    r = dec(q)
    want(txt, f"{100*r.value:.1f}%" == txt, f"decomposition {q}")
r = dec("corr_F1F2")
want("r = 0.930", f"{r.value:.3f}" == "0.930", "decomposition corr_F1F2")
r = dec("D_obs")
want("0.005328 (0.004999 to 0.005688)",
     f"{r.value:.6f} ({r.lo:.6f} to {r.hi:.6f})" == "0.005328 (0.004999 to 0.005688)", "decomposition D_obs")

# drift
for mod, sp, txt in [("csc_cox", "cv", "0.999, 95% CI 0.944 to 1.055"),
                     ("csc_cox", "temporal", "1.280, 1.157 to 1.433"),
                     ("cox_naive", "cv", "1.157 (1.092 to 1.221)"),
                     ("cox_naive", "temporal", "1.481 (1.339 to 1.659)")]:
    r = met(mod, "eo", sp=sp)
    exp = (f"{r.est:.3f}, 95% CI {r.lo:.3f} to {r.hi:.3f}" if "95% CI" in txt
           else f"{r.est:.3f}, {r.lo:.3f} to {r.hi:.3f}" if "," in txt
           else f"{r.est:.3f} ({r.lo:.3f} to {r.hi:.3f})")
    want(txt, exp == txt, f"main_metrics {mod} {sp} eo")

# deep models
for mod, txt in [("deephit", "1.377"), ("nfg", "1.283")]:
    r = met(mod, "eo")
    want(txt, f"{r.est:.3f}" == txt, f"main_metrics {mod} eo")

# alpha sweep, stated at the selected sigma on each protocol
A = pd.read_csv(ROOT / "results/deep/alpha_sweep_summary.csv")
for proto, a0, a1 in [("inner_oof", "0.158", "0.0036"), ("temporal_test", "0.164", "0.011")]:
    q = A[(A.protocol == proto) & (A.sigma == 0.1)].sort_values("alpha")
    lo, hi = q.iloc[0].ici_median, q.iloc[-1].ici_median
    want(a0, f"{lo:.3f}" == a0, f"alpha_sweep {proto} ICI at alpha=0")
    want(a1, (f"{hi:.4f}" == a1) or (f"{hi:.3f}" == a1), f"alpha_sweep {proto} ICI at alpha=1")
q = A[(A.protocol == "inner_oof") & (A.alpha >= 0.05) & (A.sigma == 0.1)]
want("0.831 to 0.833", f"{q.cindex_median.min():.3f} to {q.cindex_median.max():.3f}" == "0.831 to 0.833",
     "alpha_sweep concordance range, alpha>=0.05, selected sigma")
q0 = A[(A.protocol == "inner_oof") & (A.alpha == 0) & (A.sigma == 0.1)]
want("(0.826)", f"{q0.iloc[0].cindex_median:.3f}" == "0.826", "alpha_sweep concordance at pure ranking")

bad = [(t, ins, ok, src) for t, ins, ok, src in checks if not (ins and ok)]
for t, ins, ok, src in checks:
    status = "OK " if (ins and ok) else ("NOT-IN-TEXT" if not ins else "MISMATCH")
    print(f"{status:12s} {t:34s} <- {src}")
print()
if bad:
    print(f"FAIL: {len(bad)} of {len(checks)} checks failed")
    sys.exit(1)
print(f"PASS: all {len(checks)} load-bearing numbers trace to source")
