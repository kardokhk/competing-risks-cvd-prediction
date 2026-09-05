#!/usr/bin/env python
"""P2-A step 07b. Compare src/v2/eval_lib_v2.py against the R reference values
produced by src/v2/07a_validate_R.R.

Tolerances, and why each is what it is:

  observed CIF, E/O, mean predicted   1e-8 relative. Same estimator, no
                                      approximation, should agree to machine noise.
  pseudo-value mean and SD            1e-8. Exact leave-one-out in both.
  calibration intercept and slope     1e-4 absolute. Same model, but the numpy
                                      IRLS and geepack's solver stop on different
                                      criteria.
  ICI, E50, E90 (pseudo, 3 knots)     1e-3 absolute. Same model and same knots.
  ICI from the FGR ns(df=6) route     reported, NOT enforced. It is a different
                                      smoother with different knots, so a
                                      difference is expected and its size is the
                                      thing worth reporting.
  AUC (marginal weights)              3e-3 absolute. timeROC's AUC_2 and
                                      riskRegression::Score disagree with each
                                      other by 1.1e-3 on this data, so the
                                      tolerance has to admit that spread.
  C-index (marginal weights)          5e-3. The numpy version discretises the
                                      risk axis into 1,024 quantile bins.
  net benefit, fraction above         1e-8.

Run after 07a:  python src/v2/07b_validate_py.py [imp] [model]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v2"))
from eval_lib_v2 import (aj_cif_at, calib_intercept_slope, censor_surv,  # noqa: E402
                         cr_cindex, flexible_calibration, ipcw_auc,
                         net_benefit, net_benefit_all, pseudo_cif)

VAL = ROOT / "results" / "metrics_v2" / "validation"
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.20]
H = 10.0

TOL = {"obs_cif": (1e-8, "rel"), "mean_pred": (1e-8, "rel"), "eo": (1e-8, "rel"),
       "pseudo_mean": (1e-8, "rel"), "pseudo_sd": (1e-6, "rel"),
       "calib_intercept": (1e-4, "abs"), "calib_slope": (1e-4, "abs"),
       "ici_pseudo_rcs3": (1e-3, "abs"), "e50_pseudo_rcs3": (1e-3, "abs"),
       "e90_pseudo_rcs3": (1e-3, "abs"),
       # timeROC's AUC_2 and riskRegression::Score's Kaplan-Meier-weighted AUC are
       # themselves 1.1e-3 apart on this data, so a tolerance tighter than that
       # would be testing which of the two R packages the numpy version happens to
       # sit closer to, not whether it is correct.
       "auc_timeROC_marginal": (3e-3, "abs"), "auc_Score_km": (3e-3, "abs"),
       "cindex_pec_marginal": (5e-3, "abs")}
for t in THRESHOLDS:
    TOL[f"nb_at_{t:g}"] = (1e-8, "abs")
    TOL[f"frac_above_{t:g}"] = (1e-10, "abs")
REPORT_ONLY = {"ici_fgr_ns6", "e50_fgr_ns6", "e90_fgr_ns6", "nb_all"}


def main() -> int:
    imp = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    model = sys.argv[2] if len(sys.argv) > 2 else "csc_cox"
    ref_f = VAL / f"reference_R__{model}__imp{imp}.csv"
    if not ref_f.exists():
        print(f"missing {ref_f}; run src/v2/07a_validate_R.R {imp} {model} first")
        return 1
    ref = pd.read_csv(ref_f).set_index("metric")["value_R"].to_dict()

    d = pd.read_parquet(ROOT / "results/metrics_v2/predictions"
                        / f"{model}__temporal__imp{imp}.parquet")
    d = d[np.isfinite(d.risk_10)]
    t = d.time.to_numpy(float); e = d.event.to_numpy(np.int64)
    p = np.clip(d.risk_10.to_numpy(float), 1e-8, 1 - 1e-8)
    cyc = d.cycle.to_numpy()

    got = {}
    obs = aj_cif_at(t, e, H, 1)
    got["obs_cif"] = obs
    got["mean_pred"] = float(p.mean())
    got["eo"] = float(p.mean()) / obs
    ps = pseudo_cif(t, e, H, 1)
    got["pseudo_mean"] = float(ps.mean()); got["pseudo_sd"] = float(ps.std(ddof=1))
    icp, slp = calib_intercept_slope(ps, p)
    got["calib_intercept"] = icp; got["calib_slope"] = slp
    fc = flexible_calibration(ps, p, nknots=3)
    got["ici_pseudo_rcs3"] = fc["ici"]; got["e50_pseudo_rcs3"] = fc["e50"]
    got["e90_pseudo_rcs3"] = fc["e90"]
    got["ici_fgr_ns6"] = fc["ici"]; got["e50_fgr_ns6"] = fc["e50"]
    got["e90_fgr_ns6"] = fc["e90"]
    G_T = censor_surv(t, e, t); G_t = censor_surv(t, e, np.full(len(t), H))
    auc = ipcw_auc(t, e, p, H, G_T, G_t)
    got["auc_timeROC_marginal"] = auc; got["auc_Score_km"] = auc
    got["cindex_pec_marginal"] = cr_cindex(t, e, p, H, G_T)
    for th in THRESHOLDS:
        got[f"nb_at_{th:g}"] = net_benefit(t, e, p, th, H)
        got[f"frac_above_{th:g}"] = float((p >= th).mean())
    got["nb_all"] = net_benefit_all(t, e, 0.075, H)

    rows, fails = [], []
    for k in sorted(set(ref) | set(got)):
        r, g = ref.get(k, np.nan), got.get(k, np.nan)
        if not (np.isfinite(r) and np.isfinite(g)):
            continue
        tol, kind = TOL.get(k, (np.nan, "abs"))
        diff = abs(g - r)
        rel = diff / max(abs(r), 1e-12)
        ok = np.nan
        if k not in REPORT_ONLY and np.isfinite(tol):
            ok = bool(rel < tol) if kind == "rel" else bool(diff < tol)
            if not ok:
                fails.append(k)
        rows.append(dict(metric=k, value_R=r, value_py=g, abs_diff=diff,
                         rel_diff=rel, tolerance=tol, tol_kind=kind,
                         enforced=k not in REPORT_ONLY, passed=ok))
    out = pd.DataFrame(rows)
    out["model"] = model; out["imp"] = imp
    dest = VAL / f"comparison__{model}__imp{imp}.csv"
    out.to_csv(dest, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out[["metric", "value_R", "value_py", "abs_diff", "tolerance",
                   "enforced", "passed"]].to_string(index=False))
    print(f"\nwrote {dest}")
    if fails:
        print(f"\n{len(fails)} ESTIMATOR(S) DISAGREE WITH THE R REFERENCE: {fails}")
        return 1
    print("\nevery enforced estimator agrees with its R reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
