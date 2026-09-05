#!/usr/bin/env python
"""P2-A step 13. Stratified results, each carrying the at-risk fraction.

Version 1's headline stratified finding came from a 15-year, age 75-79 stratum in
which 7.9% of the stratum was still at risk at the horizon. A stratum can be
nominally large and carry almost no information about a long horizon, so every
stratified number produced here is reported with:

  n                    subjects in the stratum
  n_cvd_by_horizon     observed CVD deaths before the horizon
  n_competing_by_horizon
  at_risk_fraction     fraction followed to the horizon or with an event of
                       either cause before it
  flag_sparse          fewer than 15 observed CVD deaths before the horizon
  flag_low_at_risk     at-risk fraction below 0.5

Strata flagged on either count are not to be interpreted. They are written out
rather than deleted so that a reader can see what was excluded and why.

Strata: age group (30-44, 45-54, 55-64, 65-74, 75-79), NHANES cycle, sex, and
the competing-to-primary event ratio tertile, at each horizon.

Output: results/metrics_v2/strata/<split>__h<t>__<stratifier>.csv

Run: python src/v2/13_strata.py [--splits temporal,cv]
"""
from __future__ import annotations

import argparse
import importlib.util as il
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v2"))
from eval_lib_v2 import aj_cif_at, at_risk_fraction  # noqa: E402

_sp = il.spec_from_file_location("ev6", ROOT / "src/v2/06_evaluate.py")
_ev = il.module_from_spec(_sp); _sp.loader.exec_module(_ev)

OUT = ROOT / "results" / "metrics_v2" / "strata"
HORIZONS = [5.0, 10.0, 15.0]
MIN_EVENTS = 15
AGE_BINS = [30, 45, 55, 65, 75, 80]
AGE_LAB = ["30-44", "45-54", "55-64", "65-74", "75-79"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="temporal,cv")
    ap.add_argument("--imps", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pdir = ROOT / "results/metrics_v2/predictions"
    idir = ROOT / "data/processed/imputed"

    for split in a.splits.split(","):
        imps = (sorted({int(f.name.split("imp")[-1].split(".")[0])
                        for f in pdir.glob(f"*__{split}__imp*.parquet")})
                if a.imps is None else [int(x) for x in a.imps.split(",")])
        if not imps:
            print(f"no predictions for {split}")
            continue
        for hi, h in enumerate(HORIZONS):
            acc = {}
            for k in imps:
                base, risks = _ev.load_split(split, k)
                if not any(np.isfinite(v[:, hi]).any() for v in risks.values()):
                    continue
                cov = pd.read_parquet(idir / f"imp_{k}.parquet",
                                      columns=["SEQN", "age", "sex"])
                b = base.merge(cov, on="SEQN", how="left")
                b["age_group"] = pd.cut(b.age, AGE_BINS, right=False, labels=AGE_LAB)
                b["sex_label"] = np.where(b.sex == 1, "male", "female")
                for strat in ["age_group", "cycle", "sex_label"]:
                    for lev, g in b.groupby(strat, observed=True):
                        idx = g.index.to_numpy()
                        t = g.time.to_numpy(float); e = g.event.to_numpy(np.int64)
                        n_rep = max(1, b.cv_rep.nunique()) if split == "cv" else 1
                        rec = dict(
                            stratifier=strat, stratum=str(lev),
                            n=int(len(g) / n_rep),
                            n_cvd_by_horizon=int(((e == 1) & (t <= h)).sum() / n_rep),
                            n_competing_by_horizon=int(((e == 2) & (t <= h)).sum() / n_rep),
                            at_risk_fraction=at_risk_fraction(t, e, h),
                            obs_cif=aj_cif_at(t, e, h, 1),
                            obs_cif_competing=aj_cif_at(t, e, h, 2))
                        for m, v in risks.items():
                            p = v[idx, hi]
                            rec[f"meanpred_{m}"] = float(np.nanmean(p)) \
                                if np.isfinite(p).any() else np.nan
                        acc.setdefault((strat, str(lev)), []).append(rec)
            if not acc:
                continue
            rows = []
            for key, recs in acc.items():
                df = pd.DataFrame(recs)
                r = df.mean(numeric_only=True).to_dict()
                r["stratifier"], r["stratum"] = key
                r["n_imputations"] = len(recs)
                rows.append(r)
            t = pd.DataFrame(rows)
            t["flag_sparse"] = t.n_cvd_by_horizon < MIN_EVENTS
            t["flag_low_at_risk"] = t.at_risk_fraction < 0.5
            t["interpretable"] = ~(t.flag_sparse | t.flag_low_at_risk)
            for m in [c[9:] for c in t.columns if c.startswith("meanpred_")]:
                t[f"eo_{m}"] = t[f"meanpred_{m}"] / t.obs_cif.replace(0, np.nan)
            t["split"] = split; t["horizon_y"] = h
            front = ["split", "horizon_y", "stratifier", "stratum", "n",
                     "n_cvd_by_horizon", "n_competing_by_horizon", "at_risk_fraction",
                     "obs_cif", "obs_cif_competing", "flag_sparse", "flag_low_at_risk",
                     "interpretable", "n_imputations"]
            t = t[front + [c for c in t.columns if c not in front]]
            for strat, g in t.groupby("stratifier"):
                f = OUT / f"{split}__h{h:.0f}__{strat}.csv"
                g.to_csv(f, index=False)
                bad = int((~g.interpretable).sum())
                print(f"wrote {f.name}: {len(g)} strata, {bad} not interpretable "
                      f"(fewer than {MIN_EVENTS} events or at-risk fraction below 0.5)")
    print("DONE step 13")
    return 0


if __name__ == "__main__":
    sys.exit(main())
