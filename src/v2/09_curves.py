#!/usr/bin/env python
"""P2-A step 09. Curve data for the figures agent (P2-D).

Writes, for every model, split and horizon, averaged over imputations:

results/metrics_v2/calibration/
    smooth__<split>__h<t>.csv    flexible calibration curve: predicted risk on a
                                 common grid against the smoothed observed risk
                                 from the pseudo-value model, one row per model
                                 per grid point, averaged over imputations
    deciles__<split>__h<t>.csv   the ten predicted-risk decile groups with the
                                 Aalen-Johansen observed CIF and a percentile CI
                                 in each, which is what a calibration plot shows
                                 as points

results/metrics_v2/dca/
    curves__<split>__h<t>.csv    net benefit over a threshold grid, per model,
                                 plus treat-all and treat-none
    thresholds__<split>__h<t>.csv  the pre-specified thresholds only, with the
                                 fraction above threshold and the number above
                                 threshold per 1,000

Intervals on the decile points come from a percentile bootstrap within this
script. The smooth curve and the decision curves are point estimates only: the
intervals that carry the argument are the ones at the pre-specified thresholds in
`main_metrics.csv` and `paired_contrasts.csv`, which come from the full
shared-index bootstrap. Putting a band on every point of a decision curve invites
reading it as a family of simultaneous intervals, which it is not.

Thresholds are re-anchored to this endpoint. The 7.5% and 20% thresholds of the
pooled-cohort ASCVD guidelines are defined on incident atherosclerotic
cardiovascular disease, not on cardiovascular mortality, whose observed 10-year
cumulative incidence here is about 3.7%. The primary thresholds are 1, 2, 3 and
5%; 7.5% is secondary; 20% is supplementary and is flagged with the fraction of
the sample it actually interrogates.

Run: python src/v2/09_curves.py [--splits temporal,cv] [--boot 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v2"))
from eval_lib_v2 import (aj_cif_at, at_risk_fraction, flexible_calibration,  # noqa: E402
                         net_benefit, net_benefit_all, pseudo_cif)

sys.path.insert(0, str(ROOT / "src" / "v2"))
import importlib.util as _il
_spec = _il.spec_from_file_location("ev6", ROOT / "src/v2/06_evaluate.py")
_ev = _il.module_from_spec(_spec); _spec.loader.exec_module(_ev)  # reuse load_split

SPEC = json.loads((ROOT / "src/v2/feature_spec.json").read_text())
SEED = SPEC["seed"]
OUT_CAL = ROOT / "results/metrics_v2/calibration"
OUT_DCA = ROOT / "results/metrics_v2/dca"
HORIZONS = [5.0, 10.0, 15.0]
GRID_TH = np.round(np.concatenate([np.arange(0.001, 0.02, 0.001),
                                   np.arange(0.02, 0.101, 0.005),
                                   np.array([0.125, 0.15, 0.20])]), 4)
PRIMARY_TH = SPEC["thresholds_primary"] + SPEC["thresholds_secondary"] + SPEC["thresholds_supplement"]
NDEC = 10


def decile_table(t, e, p, h, nboot, rng):
    q = np.quantile(p, np.linspace(0, 1, NDEC + 1))
    q[0], q[-1] = -np.inf, np.inf
    grp = np.clip(np.searchsorted(np.unique(q)[1:-1], p, side="right"), 0, NDEC - 1)
    rows = []
    for g in range(NDEC):
        m = grp == g
        if m.sum() < 5:
            continue
        obs = aj_cif_at(t[m], e[m], h, 1)
        bs = np.empty(nboot)
        idx_pool = np.where(m)[0]
        for b in range(nboot):
            s = rng.integers(0, len(idx_pool), len(idx_pool))
            bs[b] = aj_cif_at(t[idx_pool][s], e[idx_pool][s], h, 1)
        rows.append(dict(decile=g + 1, n=int(m.sum()), mean_pred=float(p[m].mean()),
                         obs_cif=obs, obs_lo=float(np.percentile(bs, 2.5)),
                         obs_hi=float(np.percentile(bs, 97.5)),
                         n_events=int(((e[m] == 1) & (t[m] <= h)).sum()),
                         at_risk_fraction=at_risk_fraction(t[m], e[m], h)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="temporal,cv")
    ap.add_argument("--imps", default=None,
                    help="default: every imputation with prediction files present")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    OUT_CAL.mkdir(parents=True, exist_ok=True); OUT_DCA.mkdir(parents=True, exist_ok=True)

    pdir = ROOT / "results/metrics_v2/predictions"
    for split in a.splits.split(","):
        imps = (sorted({int(f.name.split("imp")[-1].split(".")[0])
                        for f in pdir.glob(f"*__{split}__imp*.parquet")})
                if a.imps is None else
                [int(x) for x in a.imps.replace("-", ",").split(",")])
        if not imps:
            print(f"no predictions for split {split}, skipping")
            continue
        print(f"{split}: {len(imps)} imputations", flush=True)
        for hi, h in enumerate(HORIZONS):
            smooth, dec, dca, thr = [], [], [], []
            for k in imps:
                base, risks = _ev.load_split(split, k)
                subs, rows_by_rep = _ev.subject_index(base, split)
                for ri, rows in enumerate(rows_by_rep):
                    t = base.time.to_numpy(float)[rows]
                    e = base.event.to_numpy(np.int64)[rows]
                    obs_all = aj_cif_at(t, e, h, 1)
                    if not np.any((e == 1) & (t <= h)):
                        continue
                    # skip a horizon no model can predict at: 15 y on the
                    # temporal split, where maximum follow-up is 13.2 y
                    if not any(np.isfinite(v[rows][:, hi]).any() for v in risks.values()):
                        continue
                    ps = pseudo_cif(t, e, h, 1)
                    for model, r in risks.items():
                        p = r[rows][:, hi]
                        if not np.isfinite(p).any():
                            continue
                        ok = np.isfinite(p)
                        pp = np.clip(p[ok], 1e-8, 1 - 1e-8)
                        tt, ee = t[ok], e[ok]
                        grid = np.quantile(pp, np.linspace(0.005, 0.995, 100))
                        fc = flexible_calibration(ps[ok], pp, nknots=3, grid=grid,
                                                  return_curve=True)
                        if fc["curve"] is not None:
                            gx, gy = fc["curve"]
                            smooth.append(pd.DataFrame(
                                dict(model=model, imp=k, cv_rep=ri, pred=gx, obs=gy)))
                        rng = np.random.default_rng([SEED, k, ri, hi])
                        dt = decile_table(tt, ee, pp, h, a.boot, rng)
                        dt["model"] = model; dt["imp"] = k; dt["cv_rep"] = ri
                        dec.append(dt)
                        dca.append(pd.DataFrame(dict(
                            model=model, imp=k, cv_rep=ri, threshold=GRID_TH,
                            net_benefit=[net_benefit(tt, ee, pp, x, h) for x in GRID_TH],
                            frac_above=[float((pp >= x).mean()) for x in GRID_TH])))
                        thr.append(pd.DataFrame(dict(
                            model=model, imp=k, cv_rep=ri, threshold=PRIMARY_TH,
                            net_benefit=[net_benefit(tt, ee, pp, x, h) for x in PRIMARY_TH],
                            frac_above=[float((pp >= x).mean()) for x in PRIMARY_TH],
                            n_above=[int((pp >= x).sum()) for x in PRIMARY_TH],
                            n_total=len(pp))))
                    dca.append(pd.DataFrame(dict(
                        model="treat_all", imp=k, cv_rep=ri, threshold=GRID_TH,
                        net_benefit=[net_benefit_all(t, e, x, h) for x in GRID_TH],
                        frac_above=1.0)))
                    dca.append(pd.DataFrame(dict(
                        model="treat_none", imp=k, cv_rep=ri, threshold=GRID_TH,
                        net_benefit=0.0, frac_above=0.0)))
                    thr.append(pd.DataFrame(dict(
                        model="treat_all", imp=k, cv_rep=ri, threshold=PRIMARY_TH,
                        net_benefit=[net_benefit_all(t, e, x, h) for x in PRIMARY_TH],
                        frac_above=1.0, n_above=len(t), n_total=len(t))))
                    thr.append(pd.DataFrame(dict(
                        model="treat_none", imp=k, cv_rep=ri, threshold=PRIMARY_TH,
                        net_benefit=0.0, frac_above=0.0, n_above=0, n_total=len(t))))
            if not smooth and not dca:
                continue
            tag = f"{split}__h{h:.0f}"
            if smooth:
                s = pd.concat(smooth)
                s["pred_bin"] = s.groupby("model").pred.transform(
                    lambda x: pd.qcut(x.rank(method="first"), 100, labels=False))
                agg = (s.groupby(["model", "pred_bin"])
                        .agg(pred=("pred", "mean"), obs=("obs", "mean"),
                             n_imp=("imp", "nunique")).reset_index())
                agg["split"] = split; agg["horizon_y"] = h
                agg.to_csv(OUT_CAL / f"smooth__{tag}.csv", index=False)
            if dec:
                dd = pd.concat(dec)
                aggd = (dd.groupby(["model", "decile"])
                          .agg(n=("n", "mean"), mean_pred=("mean_pred", "mean"),
                               obs_cif=("obs_cif", "mean"), obs_lo=("obs_lo", "mean"),
                               obs_hi=("obs_hi", "mean"), n_events=("n_events", "mean"),
                               at_risk_fraction=("at_risk_fraction", "mean"),
                               n_imp=("imp", "nunique")).reset_index())
                aggd["split"] = split; aggd["horizon_y"] = h
                aggd["flag_sparse"] = aggd.n_events < 15
                aggd.to_csv(OUT_CAL / f"deciles__{tag}.csv", index=False)
            if dca:
                cc = pd.concat(dca)
                aggc = (cc.groupby(["model", "threshold"])
                          .agg(net_benefit=("net_benefit", "mean"),
                               frac_above=("frac_above", "mean"),
                               n_imp=("imp", "nunique")).reset_index())
                aggc["split"] = split; aggc["horizon_y"] = h
                aggc.to_csv(OUT_DCA / f"curves__{tag}.csv", index=False)
            if thr:
                tt2 = pd.concat(thr)
                aggt = (tt2.groupby(["model", "threshold"])
                          .agg(net_benefit=("net_benefit", "mean"),
                               frac_above=("frac_above", "mean"),
                               n_above=("n_above", "mean"), n_total=("n_total", "mean"),
                               n_imp=("imp", "nunique")).reset_index())
                aggt["per_1000_above"] = 1000 * aggt.frac_above
                aggt["split"] = split; aggt["horizon_y"] = h
                aggt.to_csv(OUT_DCA / f"thresholds__{tag}.csv", index=False)
            print(f"  wrote curves for {tag}", flush=True)
    print("DONE step 09")
    return 0


if __name__ == "__main__":
    sys.exit(main())
