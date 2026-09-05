"""Parse NCHS public-use Linked Mortality Files (2019 release, fixed-width)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import LMF_COLSPECS, RAW_MORT, MORT_FILE, CYCLES, CVD_CODES


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.strip().replace("", np.nan), errors="coerce")


def read_mort_file(path) -> pd.DataFrame:
    names = list(LMF_COLSPECS.keys())
    colspecs = [LMF_COLSPECS[n] for n in names]
    df = pd.read_fwf(path, colspecs=colspecs, names=names,
                     dtype=str, header=None)
    for c in names:
        df[c] = _to_num(df[c])
    df["SEQN"] = df["SEQN"].astype("Int64")
    return df


def load_all_mortality() -> pd.DataFrame:
    frames = []
    for cycle, (seg, suffix, startyr) in CYCLES.items():
        fname = MORT_FILE.format(s=startyr, e=startyr + 1)
        path = RAW_MORT / fname
        if not path.exists():
            continue
        m = read_mort_file(path)
        m["cycle"] = cycle
        frames.append(m)
    mort = pd.concat(frames, ignore_index=True)
    return mort


def derive_event(mort: pd.DataFrame) -> pd.DataFrame:
    """event: 0=censored, 1=CVD death, 2=competing death.

    CVD death   = MORTSTAT==1 & UCOD_LEADING in {1,5} (heart, cerebrovascular)
    competing   = MORTSTAT==1 & UCOD_LEADING not in {1,5}
    censored    = MORTSTAT==0
    """
    ms = mort["MORTSTAT"]
    ucod = mort["UCOD_LEADING"]
    is_cvd = (ms == 1) & ucod.isin(CVD_CODES)
    is_comp = (ms == 1) & ~ucod.isin(CVD_CODES)
    event = np.where(is_cvd, 1, np.where(is_comp, 2, 0))
    mort = mort.copy()
    mort["event"] = event.astype(int)
    mort["followup_years_exm"] = mort["PERMTH_EXM"] / 12.0
    mort["followup_years_int"] = mort["PERMTH_INT"] / 12.0
    return mort
