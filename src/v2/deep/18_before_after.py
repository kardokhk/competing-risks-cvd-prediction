#!/usr/bin/env python
"""P2-C2: before/after comparison of single-seed vs 10-member ensembled deep models.

Before  = results/deep/predictions_singleseed/main_metrics_singleseed.csv
After   = results/metrics_v2/main_metrics.csv

Writes results/deep/before_after_metrics.csv (all deep estimands, both splits)
and prints the FMI summary plus the substantive headline table.
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BEFORE = ROOT / "results/deep/predictions_singleseed/main_metrics_singleseed.csv"
AFTER = ROOT / "results/metrics_v2/main_metrics.csv"
OUT = ROOT / "results/deep/before_after_metrics.csv"

DEEP = ["deephit", "nfg"]
KEY = ["split", "model", "horizon_y", "metric"]
HEADLINE = ["eo", "ici", "calib_slope", "cindex", "mean_pred", "e50", "e90", "auc"]


def load(p, tag):
    d = pd.read_csv(p)
    d = d[d.model.isin(DEEP)].copy()
    cols = KEY + ["est", "lo", "hi", "se", "fmi", "df", "m"]
    d = d[[c for c in cols if c in d.columns]]
    return d.rename(columns={c: f"{c}_{tag}" for c in d.columns if c not in KEY})


def fmt(r, tag):
    e, lo, hi = r.get(f"est_{tag}"), r.get(f"lo_{tag}"), r.get(f"hi_{tag}")
    if pd.isna(e):
        return "NA"
    return f"{e:.4f} ({lo:.4f} to {hi:.4f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", default=str(AFTER))
    a = ap.parse_args()
    b = load(BEFORE, "before")
    f = load(Path(a.after), "after")
    m = b.merge(f, on=KEY, how="outer", indicator=True)
    m["fmi_delta"] = m["fmi_after"] - m["fmi_before"]
    m = m.sort_values(KEY)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(m)} rows)")

    tgt = m[m.metric.str.startswith("frac_above") | (m.metric == "mean_pred")]
    print("\n=== success criterion: FMI for mean_pred and frac_above_* (deep models) ===")
    for split in ["temporal", "cv"]:
        s = tgt[tgt.split == split].dropna(subset=["fmi_before", "fmi_after"])
        if s.empty:
            print(f"  {split}: no overlapping rows")
            continue
        print(f"  {split}: n={len(s)}  max FMI {s.fmi_before.max():.3f} -> {s.fmi_after.max():.3f}"
              f"   median {s.fmi_before.median():.3f} -> {s.fmi_after.median():.3f}"
              f"   n(FMI>1) {int((s.fmi_before > 1).sum())} -> {int((s.fmi_after > 1).sum())}")

    print("\n=== all deep estimands ===")
    for split in ["temporal", "cv"]:
        s = m[m.split == split].dropna(subset=["fmi_before", "fmi_after"])
        if s.empty:
            continue
        print(f"  {split}: n={len(s)}  max {s.fmi_before.max():.3f} -> {s.fmi_after.max():.3f}"
              f"   median {s.fmi_before.median():.3f} -> {s.fmi_after.median():.3f}"
              f"   n(>1) {int((s.fmi_before > 1).sum())} -> {int((s.fmi_after > 1).sum())}"
              f"   n(>0.9) {int((s.fmi_before > 0.9).sum())} -> {int((s.fmi_after > 0.9).sum())}")

    print("\n=== headline metrics, est (95% CI) ===")
    h = m[m.metric.isin(HEADLINE)].sort_values(["split", "horizon_y", "metric", "model"])
    for split in ["temporal", "cv"]:
        for hz in [5.0, 10.0, 15.0]:
            s = h[(h.split == split) & (h.horizon_y == hz)]
            if s.empty:
                continue
            print(f"\n-- {split}, {hz:.0f} y --")
            print(f"{'metric':<14}{'model':<9}{'before':<30}{'after':<30}{'FMI b->a'}")
            for _, r in s.iterrows():
                fb = r.fmi_before
                fa = r.fmi_after
                fs = ("NA" if pd.isna(fb) else f"{fb:.3f}") + " -> " + ("NA" if pd.isna(fa) else f"{fa:.3f}")
                print(f"{r.metric:<14}{r.model:<9}{fmt(r,'before'):<30}{fmt(r,'after'):<30}{fs}")


if __name__ == "__main__":
    main()
