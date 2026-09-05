"""Generate codebook.md and cohort_flow.md from the analytic dataset."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from config import PROCESSED, CYCLES

df = pd.read_parquet(PROCESSED / "analytic.parquet")
flow = json.loads((PROCESSED / "_flow.json").read_text())

# ---------------- codebook ----------------
CODEBOOK = [
    # var, source file(s)/field, units, derivation/coding
    ("SEQN", "DEMO / LMF", "id", "Respondent sequence number (merge key)"),
    ("cycle", "derived", "-", "NHANES 2-year cycle label"),
    ("cycle_startyr", "derived", "year", "Cycle start year (1999..2017)"),
    ("age", "DEMO RIDAGEYR", "years", "Age at exam; cohort restricted to 30-79"),
    ("sex", "DEMO RIAGENDR", "code", "1=male, 2=female"),
    ("race", "DEMO RIDRETH1", "code", "1=MexAm 2=OthHisp 3=NHWhite 4=NHBlack 5=Other"),
    ("race_eth3", "DEMO RIDRETH3", "code", "Extended race/eth (2011+ only; NaN earlier)"),
    ("sbp", "BPX BPXSY1-4 (+BPXO BPXOSY1-3 in 2017-18)", "mmHg",
     "Mean systolic: drop 1st reading if >=2 available else use single; "
     "manual (BPX) preferred, oscillometric (BPXO) fills gaps in 2017-18"),
    ("antihtn", "BPQ BPQ050A", "0/1",
     "1 if currently taking antihypertensive (BPQ050A==1); BPQ skip-outs "
     "(never told high BP) coded 0 per NHANES convention"),
    ("totchol", "TCHOL/L13_x/LAB13 LBXTC", "mg/dL", "Total cholesterol"),
    ("hdl", "HDL/TCHOL LBDHDD|LBXHDD|LBDHDL", "mg/dL",
     "HDL cholesterol; variable name varies by cycle"),
    ("statin", "RXQ_RX RXDDRUG|RXD240B", "0/1",
     "1 if any prescription drug name contains a statin ingredient "
     "(atorva/simva/rosuva/prava/lova/fluva/pitava/cerivastatin); "
     "0 if in RXQ_RX file without statin; NaN if not in file"),
    ("diabetes", "DIQ DIQ010 + GHB LBXGH", "0/1",
     "1 if DIQ010==1 OR HbA1c>=6.5; NaN only if both unknown"),
    ("hba1c", "GHB/L10_x/LAB10 LBXGH", "%", "Glycohemoglobin (continuous)"),
    ("smoke_current", "SMQ SMQ040 (+SMQ020)", "0/1",
     "1 if SMQ040 in {1,2} (every/some days); never-smokers (SMQ020==2) set 0"),
    ("bmi", "BMX BMXBMI", "kg/m^2", "Body mass index"),
    ("scr", "BIOPRO/L40_x/LAB18 LBXSCR|LBDSCR", "mg/dL", "Serum creatinine"),
    ("egfr", "derived from scr,age,sex", "mL/min/1.73m^2",
     "CKD-EPI 2021 creatinine (race-free), Inker NEJM 2021"),
    ("uacr", "ALB_CR URDACT or URXUMA/URXUCR", "mg/g",
     "Urine albumin-creatinine ratio; URDACT used when present (2009+), "
     "else URXUMA(mg/L)/URXUCR(mg/dL)*100"),
    ("followup_years_exm", "LMF PERMTH_EXM/12", "years",
     "PRIMARY follow-up (exam to death/censor)"),
    ("followup_years_int", "LMF PERMTH_INT/12", "years",
     "Sensitivity follow-up (interview to death/censor)"),
    ("event", "LMF MORTSTAT,UCOD_LEADING", "0/1/2",
     "0=censored, 1=CVD death (UCOD_LEADING in {1,5}), 2=competing death"),
    ("MORTSTAT", "LMF", "code", "0=alive, 1=deceased"),
    ("UCOD_LEADING", "LMF", "code", "Recoded leading cause (1=heart..5=cerebro..10=other)"),
    ("ELIGSTAT", "LMF", "code", "1=eligible for linkage (cohort==1)"),
    ("SDMVPSU", "DEMO", "id", "Masked survey PSU (variance estimation)"),
    ("SDMVSTRA", "DEMO", "id", "Masked survey stratum (variance estimation)"),
    ("WTMEC2YR", "DEMO", "weight", "2-year MEC exam weight"),
    ("WTINT2YR", "DEMO", "weight", "2-year interview weight"),
]


def write_codebook():
    lines = ["# Analytic dataset codebook\n",
             f"Rows: {len(df):,}  |  one row per eligible respondent "
             "(ages 30-79, ELIGSTAT==1).\n",
             "\n| variable | source | units | % missing | derivation |",
             "|---|---|---|---|---|"]
    for var, src, unit, desc in CODEBOOK:
        miss = f"{df[var].isna().mean()*100:.1f}%" if var in df.columns else "n/a"
        lines.append(f"| {var} | {src} | {unit} | {miss} | {desc} |")
    lines += [
        "\n## Survey-weight guidance",
        "- Use **WTMEC2YR** for analyses using MEC/lab-based predictors (the",
        "  primary predictor set here). Use WTINT2YR only for interview-only vars.",
        "- When pooling K cycles, divide the 2-year weight by K "
        "(pooled weight = WTMEC2YR / n_cycles).",
        "- Always Taylor-linearize with SDMVSTRA (strata) and SDMVPSU (PSU).",
        "- Weights are provided but NOT applied in this file; downstream code",
        "  decides weighted vs unweighted ML training.",
        "\n## Notes",
        "- Missingness is preserved (no imputation here).",
        "- ~1,680 rows have PERMTH_EXM missing (interviewed, not MEC-examined; "
        "MEC weight ~0). followup_years_int is available for them.",
    ]
    (PROCESSED / "codebook.md").write_text("\n".join(lines) + "\n")


# ---------------- cohort flow ----------------
def write_flow():
    d = df
    # per-cycle event + follow-up table
    rows = []
    for cyc in CYCLES:
        s = d[d.cycle == cyc]
        ev = s.event.value_counts()
        fu = s.followup_years_exm.dropna()
        rows.append(dict(
            cycle=cyc, n=len(s),
            cvd=int(ev.get(1, 0)), competing=int(ev.get(2, 0)),
            censored=int(ev.get(0, 0)),
            median_fu=round(fu.median(), 1) if len(fu) else float("nan"),
            max_fu=round(fu.max(), 1) if len(fu) else float("nan"),
        ))
    tab = pd.DataFrame(rows)

    lines = ["# Cohort flow & event summary\n",
             "## Sample-size flow",
             f"1. Downloaded/merged NHANES respondents (all ages): "
             f"**{flow['n_downloaded']:,}**",
             f"2. Linked to LMF (any ELIGSTAT): {flow['n_linked_any']:,}",
             f"3. Age 30-79 at exam: **{flow['n_age_30_79']:,}**",
             f"4. Mortality-eligible (ELIGSTAT==1) -> FINAL N: "
             f"**{flow['n_eligible_final']:,}**",
             "",
             "## Event counts & follow-up availability BY CYCLE",
             "(follow-up = PERMTH_EXM/12, years)",
             "",
             "| cycle | N | CVD death | competing | censored | median FU (y) | max FU (y) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| {r.cycle} | {r.n:,} | {r.cvd} | {r.competing} | "
                     f"{r.censored:,} | {r.median_fu} | {r.max_fu} |")
    tot = dict(n=len(d), cvd=int((d.event == 1).sum()),
               comp=int((d.event == 2).sum()), cens=int((d.event == 0).sum()))
    lines.append(f"| **TOTAL** | {tot['n']:,} | {tot['cvd']} | {tot['comp']} | "
                 f"{tot['cens']:,} | | |")
    lines += [
        "",
        "## Temporal-split guidance",
        "- Cycles with **>=10y median follow-up** (good for long-horizon "
        "training/eval): "
        + ", ".join(tab.loc[tab.median_fu >= 10, "cycle"].tolist()),
        "- Later cycles have shorter follow-up (mortality linkage through "
        "Dec 31 2019), so they carry fewer observed CVD deaths -> favour using "
        "earlier cycles for training and later cycles for temporal validation.",
    ]
    (PROCESSED / "cohort_flow.md").write_text("\n".join(lines) + "\n")
    return tab


if __name__ == "__main__":
    write_codebook()
    tab = write_flow()
    print(tab.to_string(index=False))
