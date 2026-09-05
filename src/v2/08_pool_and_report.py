#!/usr/bin/env python
"""P2-A step 08. Pool across imputations by Rubin's rules and write the
deliverables.

**How the bootstrap and the imputations are combined.** This follows Schomaker
and Heumann (Stat Med 2018;37:2252-66, doi:10.1002/sim.7654), who compare four
ways of putting a bootstrap together with multiple imputation and find three of
them valid. The one used here as the interval is their **MI Boot (PS)**: pool the
m x B bootstrap estimates into one empirical distribution and take its 2.5th and
97.5th percentiles. It makes no normality assumption, which matters because
several of the quantities here are not close to normal: E/O is a ratio, and a net
benefit near zero is bounded below by the treat-none line. They report near
nominal coverage for MI Boot (PS) at m = 20, which is the regime here. Their
**Boot MI (PS)**, pooling in the other order, is the one they rule out, and it is
not used.

Alongside it, the classical **MI Boot** interval is reported for every quantity:

    Qbar  = mean over imputations of the estimate on the unbootstrapped data
    Ubar  = mean over imputations of the within-imputation bootstrap variance
    B     = between-imputation variance, sum (Qm - Qbar)^2 / (m - 1)
    T     = Ubar + (1 + 1/m) B
    lambda = (1 + 1/m) B / T
    nu_old = (m - 1) / lambda^2
    nu_obs = ((nu_com + 1)/(nu_com + 3)) nu_com (1 - lambda)     Barnard-Rubin
    nu     = 1 / (1/nu_old + 1/nu_obs)
    CI     = Qbar +/- t(nu, 0.975) sqrt(T)

with nu_com the complete-data degrees of freedom, taken as the number of
subjects in the split minus one. Both intervals are in the output. Where they
disagree materially, the percentile interval is the one quoted and the
disagreement is reported: that is the signal that the statistic is not normal.

The point estimate is always Qbar, the mean over imputations of the estimate on
the unbootstrapped imputed data, never the mean of the bootstrap distribution.

The fraction of missing information for each reported quantity is computed from
the same decomposition and written out, so the sufficiency of m is judged on the
estimands the paper quotes and not only on the imputation-model coefficients.

Outputs:
    results/metrics_v2/main_metrics.csv
    results/metrics_v2/paired_contrasts.csv
    results/metrics_v2/pooling_diagnostics.csv

Run: python src/v2/08_pool_and_report.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
BOOT = ROOT / "results" / "metrics_v2" / "boot"
OUT = ROOT / "results" / "metrics_v2"

N_SUBJECTS = {"temporal": 9329, "cv": 41151}

# Every contrast the manuscript will make. `matched` records whether the two
# models share a learner; an unmatched contrast is reported and labelled, never
# presented as evidence about competing risks.
CONTRASTS = [
    ("cox_naive", "csc_cox", "primary", True,
     "same Cox proportional hazards learner, competing death censored versus modelled"),
    ("rsf_naive", "rsf_cr", "secondary", True,
     "same random survival forest learner, logrank versus logrankCR splitting"),
    ("logistic", "logistic_cr", "secondary", True,
     "same IPCW binomial learner, competing death as censoring versus as an event"),
    ("cox_naive", "fine_gray", "secondary", False,
     "differs in the target of estimation as well as in the handling of competing risk"),
    ("gbs_naive", "deephit", "not_matched", False,
     "NOT a learner-matched pair: gradient-boosted Cox against a deep ranking model. "
     "They share no learner. Reported to show that the pairing structure carries the result."),
    ("deephit", "nfg", "secondary", False,
     "Both are competing-risk deep models fitted as 10-member ensembles, so this contrast "
     "isolates the training objective (ranking-weighted likelihood versus a monotone "
     "cumulative-incidence parameterisation) rather than the competing-risk assumption. "
     "Not learner-matched in the sense used for the naive-versus-competing pairs."),
]

PCT_METRICS = {"eo"}          # reported on the log scale as well


def load_all():
    """Return point[(split, imp)] and boot[(split, imp)] plus the schema."""
    schema = json.loads((BOOT / "schema.json").read_text())
    pat = re.compile(r"^(?P<split>\w+)__imp(?P<k>\d+)__chunk(?P<c>\d+)\.npz$")
    point: dict = {}
    chunks: dict = {}
    for f in sorted(BOOT.glob("*.npz")):
        mt = pat.match(f.name)
        if not mt:
            continue
        key = (mt["split"], int(mt["k"]))
        z = np.load(f)
        point.setdefault(key, z["point"])
        chunks.setdefault(key, []).append((int(mt["c"]), z["boot"]))
    boot = {k: np.concatenate([b for _, b in sorted(v)], axis=0)
            for k, v in chunks.items()}
    return point, boot, schema


def pool(qm, var_m, boot_pooled, n_com):
    """Rubin pooling plus the MI Boot (PS) percentile interval.

    qm         (m,)      per-imputation estimate on the unbootstrapped data
    var_m      (m,)      per-imputation bootstrap variance
    boot_pooled (m*B,)   every bootstrap estimate from every imputation
    """
    qm = np.asarray(qm, float); var_m = np.asarray(var_m, float)
    ok = np.isfinite(qm)
    m = int(ok.sum())
    if m == 0:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, lo_rubin=np.nan,
                    hi_rubin=np.nan, se=np.nan, fmi=np.nan, df=np.nan, m=0)
    qm = qm[ok]
    vm = var_m[ok]
    vm = np.where(np.isfinite(vm), vm, np.nan)
    qbar = float(np.mean(qm))
    ubar = float(np.nanmean(vm)) if np.isfinite(vm).any() else np.nan
    bvar = float(np.var(qm, ddof=1)) if m > 1 else 0.0
    if not np.isfinite(ubar):
        return dict(est=qbar, lo=np.nan, hi=np.nan, lo_rubin=np.nan,
                    hi_rubin=np.nan, se=np.nan, fmi=np.nan, df=np.nan, m=m)
    tvar = ubar + (1 + 1 / m) * bvar
    lam = ((1 + 1 / m) * bvar / tvar) if tvar > 0 else 0.0
    lam = min(max(lam, 0.0), 1 - 1e-12)
    nu_obs = ((n_com + 1) / (n_com + 3)) * n_com * (1 - lam)
    if m > 1 and lam > 1e-10:
        nu_old = (m - 1) / lam**2
        nu = 1.0 / (1.0 / nu_old + 1.0 / nu_obs)
    else:
        # a single imputation, or no detectable between-imputation variance:
        # there is nothing for Rubin's correction to do and the degrees of
        # freedom are the complete-data ones
        nu = nu_obs if nu_obs > 0 else float(n_com)
    q = stats.t.ppf(0.975, max(nu, 1.0))
    se = float(np.sqrt(tvar))
    fmi = float(lam + 2.0 / (nu + 3.0))
    bp = np.asarray(boot_pooled, float)
    bp = bp[np.isfinite(bp)]
    lo = float(np.percentile(bp, 2.5)) if bp.size else np.nan
    hi = float(np.percentile(bp, 97.5)) if bp.size else np.nan
    return dict(est=qbar, lo=lo, hi=hi, lo_rubin=qbar - q * se,
                hi_rubin=qbar + q * se, se=se, fmi=fmi, df=float(nu), m=m)


def main() -> int:
    point, boot, schema = load_all()
    if not point:
        print(f"no bootstrap output in {BOOT}; run src/v2/06_evaluate.py first")
        return 1
    models = schema["models"]; horizons = schema["horizons"]
    metrics = schema["metrics"]
    midx = {m: i for i, m in enumerate(metrics)}
    splits = sorted({k[0] for k in point})

    # ------------------------------------------------------------- main table
    rows = []
    for split in splits:
        imps = sorted(k[1] for k in point if k[0] == split)
        n_com = N_SUBJECTS[split] - 1
        for mi, model in enumerate(models):
            for hi, h in enumerate(horizons):
                for met in metrics:
                    j = midx[met]
                    qm = np.array([point[(split, k)][mi, hi, j] for k in imps], float)
                    if not np.isfinite(qm).any():
                        continue
                    vm = np.array([np.nanvar(boot[(split, k)][:, mi, hi, j], ddof=1)
                                   for k in imps], float)
                    bp = np.concatenate([boot[(split, k)][:, mi, hi, j] for k in imps])
                    r = pool(qm, vm, bp, n_com)
                    rows.append(dict(split=split, model=model, family=(
                        "naive" if model in ("logistic", "cox_naive", "rsf_naive",
                                             "gbs_naive") else "competing"),
                        horizon_y=h, metric=met, **r))
    main = pd.DataFrame(rows)
    main.to_csv(OUT / "main_metrics.csv", index=False)
    print(f"wrote {OUT/'main_metrics.csv'} ({len(main)} rows)")

    # -------------------------------------------------------- paired contrasts
    crows = []
    for split in splits:
        imps = sorted(k[1] for k in point if k[0] == split)
        n_com = N_SUBJECTS[split] - 1
        for a, b, tier, matched, note in CONTRASTS:
            if a not in models or b not in models:
                continue
            ia, ib = models.index(a), models.index(b)
            for hi, h in enumerate(horizons):
                for met in metrics:
                    j = midx[met]
                    qm = np.array([point[(split, k)][ia, hi, j]
                                   - point[(split, k)][ib, hi, j] for k in imps], float)
                    if not np.isfinite(qm).any():
                        continue
                    dboot = [boot[(split, k)][:, ia, hi, j] - boot[(split, k)][:, ib, hi, j]
                             for k in imps]
                    vm = np.array([np.nanvar(d, ddof=1) for d in dboot], float)
                    r = pool(qm, vm, np.concatenate(dboot), n_com)
                    excl = (np.isfinite(r["lo"]) and np.isfinite(r["hi"])
                            and (r["lo"] > 0 or r["hi"] < 0))
                    crows.append(dict(split=split, contrast=f"{a} - {b}",
                                      model_a=a, model_b=b, tier=tier,
                                      learner_matched=matched, horizon_y=h,
                                      metric=met, **r,
                                      ci_excludes_zero=bool(excl), note=note))
                # E/O also on the log scale, where a ratio is closer to normal
                j = midx["eo"]
                qa = np.array([point[(split, k)][ia, hi, j] for k in imps], float)
                qb = np.array([point[(split, k)][ib, hi, j] for k in imps], float)
                if np.isfinite(qa).any() and np.isfinite(qb).any():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        qm = np.log(qa) - np.log(qb)
                        dboot = [np.log(boot[(split, k)][:, ia, hi, j])
                                 - np.log(boot[(split, k)][:, ib, hi, j]) for k in imps]
                    vm = np.array([np.nanvar(d, ddof=1) for d in dboot], float)
                    r = pool(qm, vm, np.concatenate(dboot), n_com)
                    excl = (np.isfinite(r["lo"]) and np.isfinite(r["hi"])
                            and (r["lo"] > 0 or r["hi"] < 0))
                    crows.append(dict(split=split, contrast=f"{a} - {b}",
                                      model_a=a, model_b=b, tier=tier,
                                      learner_matched=matched, horizon_y=h,
                                      metric="log_eo_ratio", **r,
                                      ci_excludes_zero=bool(excl), note=note))
    con = pd.DataFrame(crows)
    con.to_csv(OUT / "paired_contrasts.csv", index=False)
    print(f"wrote {OUT/'paired_contrasts.csv'} ({len(con)} rows)")

    # ------------------------------------------------------------ diagnostics
    diag = main[["split", "model", "horizon_y", "metric", "fmi", "df", "m", "se"]].copy()
    diag["m_required_von_hippel"] = np.nan
    fin = diag.fmi.between(1e-6, 0.999) & diag.m.gt(1)
    fu = 1 / (1 + np.exp(-(np.log(diag.loc[fin, "fmi"] / (1 - diag.loc[fin, "fmi"]))
                           + 1.96 * np.sqrt(2 / diag.loc[fin, "m"]))))
    diag.loc[fin, "m_required_von_hippel"] = np.ceil(1 + 0.5 * (fu / 0.05) ** 2)
    diag.to_csv(OUT / "pooling_diagnostics.csv", index=False)
    key = diag[diag.metric.isin(["eo", "calib_slope", "calib_intercept", "ici",
                                 "auc", "cindex", "nb_at_0.03"])]
    print(f"wrote {OUT/'pooling_diagnostics.csv'}")
    print(f"largest FMI over the reported estimands: {key.fmi.max():.3f}; "
          f"largest m required: {key.m_required_von_hippel.max():.0f}; m used: {key.m.max():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
