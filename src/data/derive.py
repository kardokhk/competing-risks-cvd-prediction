"""Derivation helpers: eGFR (CKD-EPI 2021 race-free), SBP, statin matching, UACR."""
from __future__ import annotations
import numpy as np
import pandas as pd

# Statin ingredients (generic names). Matched case-insensitively as substrings
# against the NHANES drug-name field (RXD240B in 1999-2002, RXDDRUG in 2003+).
STATINS = [
    "ATORVASTATIN", "SIMVASTATIN", "ROSUVASTATIN", "PRAVASTATIN",
    "LOVASTATIN", "FLUVASTATIN", "PITAVASTATIN", "CERIVASTATIN",
]


def ckdepi_2021(scr: pd.Series, age: pd.Series, sex: pd.Series) -> pd.Series:
    """CKD-EPI 2021 creatinine equation (race-free), Inker et al. NEJM 2021.

        eGFR = 142 * min(Scr/kappa, 1)**alpha
                   * max(Scr/kappa, 1)**(-1.200)
                   * 0.9938**age
                   * (1.012 if female)
      Scr in mg/dL. Female: kappa=0.7, alpha=-0.241. Male: kappa=0.9, alpha=-0.302.
      sex coding follows NHANES RIAGENDR: 1=male, 2=female.
    """
    scr = pd.to_numeric(scr, errors="coerce")
    age = pd.to_numeric(age, errors="coerce")
    female = (sex == 2)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / kappa
    egfr = (142.0
            * np.minimum(ratio, 1.0) ** alpha
            * np.maximum(ratio, 1.0) ** (-1.200)
            * 0.9938 ** age
            * np.where(female, 1.012, 1.0))
    egfr = pd.Series(egfr, index=scr.index)
    egfr[scr.isna() | age.isna()] = np.nan
    return egfr


def mean_sbp(df: pd.DataFrame, sy_cols: list[str]) -> pd.Series:
    """Mean systolic BP.

    Rule: among the available (non-missing) readings, if >=2 exist drop the
    first reading and average the rest (first measurement runs high); if only
    one reading exists use it. Applied per-respondent across the given columns.
    """
    present = [c for c in sy_cols if c in df.columns]
    vals = df[present].apply(pd.to_numeric, errors="coerce")

    def _row(row):
        r = row.dropna().values
        if len(r) == 0:
            return np.nan
        if len(r) >= 2:
            return float(np.mean(r[1:]))
        return float(r[0])

    return vals.apply(_row, axis=1)


def compute_uacr(urxuma: pd.Series, urxucr: pd.Series,
                 urdact: pd.Series | None) -> pd.Series:
    """Urine albumin-to-creatinine ratio in mg/g.

    Prefer the NCHS-provided URDACT (mg/g) when present; otherwise compute from
    URXUMA (mg/L) / URXUCR (mg/dL) * 100  ->  mg/g.
    """
    uma = pd.to_numeric(urxuma, errors="coerce")
    ucr = pd.to_numeric(urxucr, errors="coerce")
    computed = uma / ucr * 100.0
    if urdact is not None:
        provided = pd.to_numeric(urdact, errors="coerce")
        return provided.where(provided.notna(), computed)
    return computed


def statin_users(rxq: pd.DataFrame, drug_col: str) -> set:
    """Return set of SEQN with >=1 statin prescription."""
    if drug_col not in rxq.columns:
        return set()
    names = rxq[drug_col].astype(str).str.upper()
    mask = names.apply(lambda s: any(st in s for st in STATINS))
    return set(rxq.loc[mask, "SEQN"].unique())
