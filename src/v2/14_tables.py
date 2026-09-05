#!/usr/bin/env python
"""
Tables 1 to 3 for the rebuilt manuscript.

Table 1  Baseline characteristics of the analysis cohort, on OBSERVED data with
         missingness reported. Table 1 describes what was measured; imputed
         values are not data and are not shown here. The primary analysis is
         multiply imputed on this same cohort (N = 41,151).
Table 2  Model performance, temporal test set, 10 y, Rubin-pooled over m = 30.
Table 3  Paired contrasts, the comparisons the manuscript actually makes.

Every value traces to data/processed/analytic.parquet (Table 1) or
results/metrics_v2/{main_metrics,paired_contrasts}.csv (Tables 2 and 3).

Usage:  python src/v2/14_tables.py
Seed:   not stochastic.
"""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_MD = ROOT / "docs"
OUT_CSV = ROOT / "results" / "tables"
OUT_CSV.mkdir(parents=True, exist_ok=True)

CONT = [("age", "Age, years", 1), ("sbp", "Systolic blood pressure, mm Hg", 1),
        ("totchol", "Total cholesterol, mg/dL", 1), ("hdl", "HDL cholesterol, mg/dL", 1),
        ("bmi", "Body-mass index, kg/m2", 1), ("egfr", "eGFR, mL/min/1.73 m2", 1)]
BIN = [("male", "Male sex"), ("antihtn", "Antihypertensive use"), ("statin", "Statin use"),
       ("diabetes", "Diabetes"), ("smoke_current", "Current smoking")]


def fmt_cont(s, dp):
    s = s.dropna()
    return f"{s.mean():.{dp}f} ({s.std():.{dp}f})" if len(s) else "-"


def fmt_bin(s):
    s = s.dropna()
    return f"{int(s.sum()):,} ({100*s.mean():.1f})" if len(s) else "-"


def table1():
    d = pd.read_parquet(ROOT / "data/processed/analytic.parquet").copy()
    d["male"] = (d.sex == 1).astype(float)
    era = pd.cut(d.cycle_startyr, [1998, 2006, 2010, 2018],
                 labels=["Train, 1999-2006", "Temporal test, 2007-2010", "Later cycles, 2011-2018"])
    groups = [("All eligible", d)]
    for lab in era.cat.categories:
        groups.append((lab, d[era == lab]))
    for lab, code in [("CVD death", 1), ("Competing death", 2), ("Censored", 0)]:
        groups.append((lab, d[d.event == code]))

    rows = []
    rows.append(["N", *[f"{len(g):,}" for _, g in groups]])
    for col, lab, dp in CONT:
        rows.append([lab, *[fmt_cont(g[col], dp) for _, g in groups]])
    for col, lab in BIN:
        rows.append([lab, *[fmt_bin(g[col]) for _, g in groups]])
    rows.append(["Median follow-up, years",
                 *[f"{g.followup_years_exm.median():.1f}" if g.followup_years_exm.notna().any() else "-"
                   for _, g in groups]])
    rows.append(["CVD deaths", *[f"{int((g.event==1).sum()):,}" for _, g in groups]])
    rows.append(["Competing deaths", *[f"{int((g.event==2).sum()):,}" for _, g in groups]])

    hdr = ["Characteristic"] + [lab for lab, _ in groups]
    df = pd.DataFrame(rows, columns=hdr)
    df.to_csv(OUT_CSV / "table1_baseline.csv", index=False)

    miss = pd.DataFrame({
        "Predictor": ["Systolic blood pressure", "Total cholesterol", "HDL cholesterol",
                      "Body-mass index", "eGFR"],
        "n missing": [int(d[c].isna().sum()) for c in ["sbp", "totchol", "hdl", "bmi", "egfr"]],
        "% missing": [f"{100*d[c].isna().mean():.2f}" for c in ["sbp", "totchol", "hdl", "bmi", "egfr"]]})
    return df, miss, len(d)


def _get(m, model, metric, h=10.0, split="temporal"):
    r = m[(m.split == split) & (m.horizon_y == h) & (m.model == model) & (m.metric == metric)]
    if r.empty:
        return None
    return r.iloc[0]


def ci(r, dp=3, scale=1.0):
    if r is None:
        return "-"
    return f"{r.est*scale:.{dp}f} ({r.lo*scale:.{dp}f} to {r.hi*scale:.{dp}f})"


LABEL = {"logistic": "IPCW binomial, naive", "cox_naive": "Cox, naive",
         "rsf_naive": "Random survival forest, naive", "gbs_naive": "Gradient-boosted survival, naive",
         "logistic_cr": "IPCW binomial, competing risk", "csc_cox": "Cause-specific Cox",
         "fine_gray": "Fine-Gray", "rsf_cr": "Random survival forest, competing risk",
         "deephit": "DeepHit (10-member ensemble)", "nfg": "Neural Fine-Gray (10-member ensemble)"}
ORDER = ["logistic", "cox_naive", "rsf_naive", "gbs_naive",
         "logistic_cr", "csc_cox", "fine_gray", "rsf_cr", "deephit", "nfg"]


def table2():
    m = pd.read_csv(ROOT / "results/metrics_v2/main_metrics.csv")
    obs = _get(m, "logistic", "obs_cif")
    rows = []
    for mod in ORDER:
        rows.append({
            "Model": LABEL[mod],
            "Family": "naive" if mod in ("logistic", "cox_naive", "rsf_naive", "gbs_naive") else "competing risk",
            "E/O (95% CI)": ci(_get(m, mod, "eo")),
            "Calibration slope": ci(_get(m, mod, "calib_slope")),
            "Calibration intercept": ci(_get(m, mod, "calib_intercept")),
            "ICI per 1,000": ci(_get(m, mod, "ici"), dp=1, scale=1000),
            "C-index": ci(_get(m, mod, "cindex")),
            "Above 5% threshold, %": ci(_get(m, mod, "frac_above_0.05"), dp=1, scale=100)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV / "table2_model_performance.csv", index=False)
    return df, obs


def table3():
    c = pd.read_csv(ROOT / "results/metrics_v2/paired_contrasts.csv")
    keep = ["cox_naive - csc_cox", "rsf_naive - rsf_cr", "logistic - logistic_cr",
            "cox_naive - fine_gray", "gbs_naive - deephit", "deephit - nfg"]
    rows = []
    for con in keep:
        s = c[(c.split == "temporal") & (c.horizon_y == 10) & (c.contrast == con)]
        if s.empty:
            continue
        g = lambda met, dp=3, sc=1.0: (
            f"{s[s.metric==met].iloc[0].est*sc:.{dp}f} ({s[s.metric==met].iloc[0].lo*sc:.{dp}f} to "
            f"{s[s.metric==met].iloc[0].hi*sc:.{dp}f})" if not s[s.metric == met].empty else "-")
        matched = bool(s.iloc[0].learner_matched)
        rows.append({
            "Contrast": con.replace("_", " "),
            "Learner-matched": "yes" if matched else "no",
            "delta E/O": g("eo"),
            "delta calibration slope": g("calib_slope"),
            "delta ICI per 1,000": g("ici", 2, 1000),
            "delta C-index": g("cindex", 4),
            "delta net benefit at 3%, per 1,000": g("nb_at_0.03", 2, 1000),
            "delta above 5% threshold, pp": g("frac_above_0.05", 2, 100)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV / "table3_paired_contrasts.csv", index=False)
    return df


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    t1, miss, n = table1()
    t2, obs = table2()
    t3 = table3()
    txt = [f"""# Tables 1 to 3

Generated by `src/v2/14_tables.py`. Table 1 from `data/processed/analytic.parquet`;
Tables 2 and 3 from `results/metrics_v2/main_metrics.csv` and `paired_contrasts.csv`.

## Table 1. Baseline characteristics of the analysis cohort

Observed data, N = {n:,} adults aged 30 to 79 eligible for mortality linkage. Continuous variables
are mean (SD); binary variables are n (%). Denominators for variables with missing data are the
observed values; missingness is given below. The primary analysis multiply imputes these
predictors on this same cohort (m = 30).

{md(t1)}

### Missing values, by predictor

The six questionnaire and demographic predictors are complete by construction.

{md(miss)}

## Table 2. Model performance, temporal test set, 10 years

Rubin-pooled over m = 30 imputations, with 95% percentile intervals from a shared-index bootstrap
(B = 2,000). Observed 10-year Aalen-Johansen cumulative incidence {obs.est*100:.2f}%
({obs.lo*100:.2f} to {obs.hi*100:.2f}). E/O above 1 indicates over-prediction; a calibration slope
below 1 indicates predictions that are too extreme. The two deep models are 10-member ensembles
and so are a different estimator from the eight single fits.

{md(t2)}

## Table 3. Paired contrasts, temporal test set, 10 years

Each contrast is computed on shared bootstrap resamples and pooled by Rubin's rules, so it is an
estimate of the difference rather than a difference of two separately estimated numbers. Only
learner-matched contrasts isolate the competing-risk assumption. `deephit - nfg` compares two
competing-risk deep models and so isolates the training objective, not the competing-risk
assumption.

{md(t3)}
"""]
    (OUT_MD / "tables.md").write_text("\n".join(txt))
    print("wrote docs/tables.md and three CSVs in results/tables/")
