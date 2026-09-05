#!/usr/bin/env python
"""P2-B step 3, residual characterisation: is the 22% over-prediction shared by
BOTH model families a competing-risk artefact or temporal drift?

Contemporaneous out-of-fold cross-validation on the whole 1999-2018 cohort is
compared with the temporal 1999-2006 -> 2007-2010 split.  If the shared
over-prediction is drift, the cross-validated E/O should sit near 1 while the
temporal E/O sits near 1.22.  Observed risk is the Aalen-Johansen cause-1 CIF.

Note the 15 y temporal rows are not estimable (maximum follow-up on the test set
is 13.25 y); they are emitted with at_risk_frac = 0 and must not be quoted.

Seed 20260903.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cf_lib import SEED, cif_and_naive  # noqa: E402

OUT = ROOT / "results" / "closed_form"
HORIZONS = [5, 10, 15]
N_BOOT = 500


def eo(d: pd.DataFrame, h: int, rng) -> dict:
    t, e = d["time"].to_numpy(float), d["event"].to_numpy(int)
    r = d[f"risk_{h}"].to_numpy(float)
    o = cif_and_naive(t, e, h)
    n = len(d)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ix = rng.integers(0, n, n)
        ob = cif_and_naive(t[ix], e[ix], h)
        draws[b] = r[ix].mean() / ob["F1"] if ob["F1"] > 0 else np.nan
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return dict(horizon=h, n=n, n_ev1=o["n_ev1"], at_risk_frac=o["at_risk_frac"],
                mean_pred=r.mean(), aj_obs=o["F1"],
                EO=r.mean() / o["F1"], EO_lo=lo, EO_hi=hi)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    full = pd.read_parquet(ROOT / "data/processed/model_matrices/full.parquet",
                           columns=["SEQN", "time", "event"])
    test = pd.read_parquet(ROOT / "data/processed/model_matrices/test.parquet",
                           columns=["SEQN", "time", "event"])
    cols = [f"risk_{h}" for h in HORIZONS]
    for m in ("csc_cox", "cox_naive"):
        p = pd.read_parquet(ROOT / f"results/predictions/{m}__cv.parquet",
                            columns=["SEQN"] + cols)
        p = p.groupby("SEQN", as_index=False)[cols].mean()
        d = full.merge(p, on="SEQN")
        for h in HORIZONS:
            rows.append(dict(model=m, design="cv_oof_1999_2018", **eo(d, h, rng)))
        q = pd.read_parquet(ROOT / f"results/predictions/{m}__temporal.parquet",
                            columns=["SEQN"] + cols)
        d = test.merge(q, on="SEQN")
        for h in HORIZONS:
            rows.append(dict(model=m, design="temporal_2007_2010", **eo(d, h, rng)))
        print(f"{m} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "drift_check.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
