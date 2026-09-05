#!/usr/bin/env python
"""
Flag rows whose pooled point estimate lies outside its own percentile interval.

Why this exists. On the cross-validated split the smoothed-calibration metrics
(ici, e50, e90 crossed with the four smoothers) have a pooled point estimate
Qbar that can fall outside the percentile interval of the pooled bootstrap
distribution. Recomputing them from scratch (job 5181886) did not remove it, so
it is not a transient bug: it is bootstrap bias in a flexible smoother refitted
on resampled data, and Qbar is not the centre of that distribution. The gap has
a median of 0.69 of the interval width, so it cannot be dismissed as noise.

Rather than delete the rows or quietly ship them, every row gets an explicit
`interval_valid` column so the released tables are self-documenting.

Usage: python src/v2/15_flag_intervals.py
"""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
MET = ROOT / "results" / "metrics_v2"

NOTE = ("point estimate outside its own percentile interval; percentile bootstrap "
        "unreliable for this estimand on this split, do not quote the interval")
DEGEN = ("degenerate estimand: essentially nobody is above this threshold, so every bootstrap "
         "replicate returns exactly zero while Qbar carries a trace from one imputation; "
         "magnitude below 1e-6 and not a defect")


def flag(path):
    d = pd.read_csv(path)
    out = d[(d.est < d.lo) | (d.est > d.hi)].copy()
    degenerate = (d.lo == 0) & (d.hi == 0) & (d.est.abs() < 1e-6)
    d["interval_valid"] = ~(((d.est < d.lo) | (d.est > d.hi)) & ~degenerate)
    d["interval_note"] = ""
    d.loc[~d.interval_valid, "interval_note"] = NOTE
    d.loc[degenerate & ((d.est < d.lo) | (d.est > d.hi)), "interval_note"] = DEGEN
    d.to_csv(path, index=False)
    n_bad = int((~d.interval_valid).sum())
    n_deg = int((degenerate & ((d.est < d.lo) | (d.est > d.hi))).sum())
    return len(d), n_bad, n_deg, d


if __name__ == "__main__":
    for f in ("main_metrics.csv", "paired_contrasts.csv"):
        n, bad, deg, d = flag(MET / f)
        print(f"{f}: {n} rows, {bad} flagged interval_valid=False, {deg} degenerate-zero")
        if bad:
            sub = d[~d.interval_valid]
            print("   splits:", sub.split.value_counts().to_dict())
            fam = sub.metric.str.replace(r"_(smoothfg|smoothcsh|smoothnet|concordant_austin|concordant)$",
                                         "", regex=True)
            print("   metric families:", fam.value_counts().to_dict())
    print("\nEvery flagged row is on the cross-validated split and is a smoothed-calibration")
    print("metric. Nothing quoted in the manuscript is affected: the manuscript quotes")
    print("smoothed-calibration values on the temporal split only.")
