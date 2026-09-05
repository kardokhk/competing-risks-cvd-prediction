"""Build the analytic dataset from downloaded NHANES XPT + parsed LMF.

One row per eligible respondent (ages 30-79 at exam, ELIGSTAT==1). Raw
missingness is preserved (NO imputation here).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import CYCLES, RAW_NHANES, PROCESSED
from mortality import load_all_mortality, derive_event
from derive import ckdepi_2021, mean_sbp, compute_uacr, statin_users, STATINS

# ---- per-cycle field-name resolution (verified empirically) ----
# HDL variable name differs by cycle:
HDL_VARS = ["LBDHDD", "LBXHDD", "LBDHDL"]      # try in order
SCR_VARS = ["LBXSCR", "LBDSCR"]               # serum creatinine
# drug-name field for statin matching:
RX_DRUG_VARS = ["RXDDRUG", "RXD240B"]


def _read(component: str, cycle: str) -> pd.DataFrame | None:
    path = RAW_NHANES / f"{component}_{cycle}.XPT"
    if not path.exists():
        return None
    df = pd.read_sas(path, format="xport")
    if "SEQN" in df.columns:
        df["SEQN"] = df["SEQN"].astype("int64")
    return df


def _first_col(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_cycle(cycle: str, startyr: int) -> pd.DataFrame:
    demo = _read("DEMO", cycle)
    keep = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1",
            "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR"]
    keep = [c for c in keep if c in demo.columns]
    base = demo[keep].copy()
    base["cycle"] = cycle
    base["cycle_startyr"] = startyr
    base["race"] = demo["RIDRETH1"] if "RIDRETH1" in demo.columns else np.nan
    base["race_eth3"] = demo["RIDRETH3"] if "RIDRETH3" in demo.columns else np.nan

    # --- blood pressure: manual BPX + oscillometric BPXO (2017-18) ---
    bpx = _read("BPX", cycle)
    bpxo = _read("BPXO", cycle)
    sbp = pd.Series(np.nan, index=base.index)
    seqn_index = base.set_index("SEQN")
    sbp_map = {}
    if bpx is not None:
        s = mean_sbp(bpx, ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"])
        sbp_map.update(dict(zip(bpx["SEQN"], s)))
    if bpxo is not None:
        so = mean_sbp(bpxo, ["BPXOSY1", "BPXOSY2", "BPXOSY3"])
        # fill only where manual missing (manual preferred for comparability)
        for seqn, v in zip(bpxo["SEQN"], so):
            if pd.isna(sbp_map.get(seqn, np.nan)) and not pd.isna(v):
                sbp_map[seqn] = v
    base["sbp"] = base["SEQN"].map(sbp_map)

    # --- antihypertensive use: BPQ050A == 1 ---
    bpq = _read("BPQ", cycle)
    if bpq is not None and "BPQ050A" in bpq.columns:
        m = dict(zip(bpq["SEQN"], (pd.to_numeric(bpq["BPQ050A"], errors="coerce") == 1).astype(float)))
        base["antihtn"] = base["SEQN"].map(m)
    else:
        base["antihtn"] = np.nan

    # --- BMI ---
    bmx = _read("BMX", cycle)
    if bmx is not None and "BMXBMI" in bmx.columns:
        base["bmi"] = base["SEQN"].map(dict(zip(bmx["SEQN"], bmx["BMXBMI"])))
    else:
        base["bmi"] = np.nan

    # --- total & HDL cholesterol ---
    tchol = _read("TCHOL", cycle)
    hdl = _read("HDL", cycle)
    if tchol is not None and "LBXTC" in tchol.columns:
        base["totchol"] = base["SEQN"].map(dict(zip(tchol["SEQN"], tchol["LBXTC"])))
    else:
        base["totchol"] = np.nan
    hdl_src = hdl if hdl is not None else tchol   # early combined file holds both
    hdl_col = _first_col(hdl_src, HDL_VARS)
    if hdl_src is not None and hdl_col:
        base["hdl"] = base["SEQN"].map(dict(zip(hdl_src["SEQN"], hdl_src[hdl_col])))
    else:
        base["hdl"] = np.nan

    # --- HbA1c ---
    ghb = _read("GHB", cycle)
    if ghb is not None and "LBXGH" in ghb.columns:
        base["hba1c"] = base["SEQN"].map(dict(zip(ghb["SEQN"], ghb["LBXGH"])))
    else:
        base["hba1c"] = np.nan

    # --- diabetes: DIQ010==1 OR HbA1c>=6.5 ---
    diq = _read("DIQ", cycle)
    diq_dx = pd.Series(np.nan, index=base.index)
    if diq is not None and "DIQ010" in diq.columns:
        m = dict(zip(diq["SEQN"], (pd.to_numeric(diq["DIQ010"], errors="coerce") == 1).astype(float)))
        diq_dx = base["SEQN"].map(m)
    hba_db = (base["hba1c"] >= 6.5).astype(float)
    # diabetes=1 if either flag positive; NaN only if both unknown
    both_na = diq_dx.isna() & base["hba1c"].isna()
    diab = ((diq_dx == 1) | (hba_db == 1)).astype(float)
    diab[both_na] = np.nan
    base["diabetes"] = diab

    # --- current smoking: SMQ040 in {1,2} ---
    smq = _read("SMQ", cycle)
    if smq is not None and "SMQ040" in smq.columns:
        v = pd.to_numeric(smq["SMQ040"], errors="coerce")
        m = dict(zip(smq["SEQN"], v.isin([1, 2]).astype(float)))
        base["smoke_current"] = base["SEQN"].map(m)
        # SMQ040 only asked of ever-smokers; never-smokers (SMQ020==2) -> 0
        if "SMQ020" in smq.columns:
            never = dict(zip(smq["SEQN"], (pd.to_numeric(smq["SMQ020"], errors="coerce") == 2)))
            nm = base["SEQN"].map(never)
            base.loc[(nm == True) & base["smoke_current"].isna(), "smoke_current"] = 0.0
    else:
        base["smoke_current"] = np.nan

    # --- serum creatinine -> eGFR (CKD-EPI 2021) ---
    biopro = _read("BIOPRO", cycle)
    scr_col = _first_col(biopro, SCR_VARS)
    if biopro is not None and scr_col:
        base["scr"] = base["SEQN"].map(dict(zip(biopro["SEQN"], biopro[scr_col])))
    else:
        base["scr"] = np.nan
    base["egfr"] = ckdepi_2021(base["scr"], base["RIDAGEYR"], base["RIAGENDR"])

    # --- UACR ---
    albcr = _read("ALB_CR", cycle)
    if albcr is not None and {"URXUMA", "URXUCR"}.issubset(albcr.columns):
        urdact = albcr["URDACT"] if "URDACT" in albcr.columns else None
        uacr = compute_uacr(albcr["URXUMA"], albcr["URXUCR"], urdact)
        base["uacr"] = base["SEQN"].map(dict(zip(albcr["SEQN"], uacr)))
    else:
        base["uacr"] = np.nan

    # --- statin use (RXQ_RX) ---
    rxq = _read("RXQ_RX", cycle)
    drug_col = _first_col(rxq, RX_DRUG_VARS)
    if rxq is not None and drug_col:
        users = statin_users(rxq, drug_col)
        # RXQ covers respondents seen; SEQN present in RXQ but not statin -> 0
        seen = set(rxq["SEQN"].unique())
        base["statin"] = base["SEQN"].apply(
            lambda s: 1.0 if s in users else (0.0 if s in seen else np.nan))
    else:
        base["statin"] = np.nan

    base["sex"] = base["RIAGENDR"]
    base["age"] = base["RIDAGEYR"]
    return base


def main() -> pd.DataFrame:
    parts = [build_cycle(cy, sy) for cy, (seg, suf, sy) in CYCLES.items()]
    nh = pd.concat(parts, ignore_index=True)
    n_downloaded = len(nh)

    # mortality
    mort = derive_event(load_all_mortality())
    mcols = ["SEQN", "cycle", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING",
             "event", "followup_years_exm", "followup_years_int",
             "PERMTH_EXM", "PERMTH_INT"]
    merged = nh.merge(mort[mcols], on=["SEQN", "cycle"], how="left")
    n_merged = len(merged)

    # age restriction 30-79 at exam
    age_ok = merged["age"].between(30, 79)
    aged = merged[age_ok].copy()

    # eligible mortality linkage
    final = aged[aged["ELIGSTAT"] == 1].copy()

    out_cols = [
        "SEQN", "cycle", "cycle_startyr", "age", "sex", "race", "race_eth3",
        "sbp", "antihtn", "totchol", "hdl", "statin", "diabetes", "hba1c",
        "smoke_current", "bmi", "scr", "egfr", "uacr",
        "followup_years_exm", "followup_years_int", "PERMTH_EXM", "PERMTH_INT",
        "event", "MORTSTAT", "UCOD_LEADING", "ELIGSTAT",
        "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTINT2YR",
    ]
    out_cols = [c for c in out_cols if c in final.columns]
    final = final[out_cols].reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    final.to_parquet(PROCESSED / "analytic.parquet", index=False)

    # stash flow counts for reporting
    flow = dict(
        n_downloaded=n_downloaded, n_merged=n_merged,
        n_age_30_79=int(age_ok.sum()),
        n_eligible_final=len(final),
        n_linked_any=int(merged["ELIGSTAT"].notna().sum()),
    )
    import json
    (PROCESSED / "_flow.json").write_text(json.dumps(flow, indent=2))
    print("FLOW:", flow)
    return final


if __name__ == "__main__":
    main()
