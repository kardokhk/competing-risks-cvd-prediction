#!/usr/bin/env python
"""P2-A step 12. Pool the concordant-smoother ICI, E50 and E90 from step 11b and
append them to main_metrics.csv and paired_contrasts.csv.

Six metric columns are added per model:

  ici_smoothfg   ici_smoothcsh   ici_smoothnet     and E50, E90 likewise
  ici_concordant_austin                            the one of the three that is
                                                   concordant with that model's
                                                   type, NaN where undetermined

`concordant_smoother` records which was used. Pooling is the same MI Boot (PS)
percentile interval plus the Rubin interval as everything else, using the same
`pool()` function and the same replicate indices, so these contrasts are paired
with the rest of the analysis.

Run after 11b: python src/v2/12_merge_concordant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v2"))
import importlib.util as _il

_sp = _il.spec_from_file_location("pool8", ROOT / "src/v2/08_pool_and_report.py")
_p8 = _il.module_from_spec(_sp); _sp.loader.exec_module(_p8)
pool, CONTRASTS, N_SUBJECTS = _p8.pool, _p8.CONTRASTS, _p8.N_SUBJECTS

OUT = ROOT / "results" / "metrics_v2"
CONC = OUT / "concordant"
BASE = ["ici", "e50", "e90"]
SMOOTHERS = ["smoothfg", "smoothcsh", "smoothnet"]
NAIVE = {"logistic", "cox_naive", "rsf_naive", "gbs_naive"}


def main() -> int:
    files = sorted(CONC.glob("*.csv"))
    if not files:
        print(f"no concordant-smoother output in {CONC}; run src/v2/11b_concordant_ici.R")
        return 1
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.drop_duplicates(subset=["split", "horizon_y", "imp", "b", "model"])
    # the concordant column for each model
    for b in BASE:
        d[f"{b}_concordant_austin"] = np.nan
        for sm in SMOOTHERS:
            m = d.concordant == sm
            d.loc[m, f"{b}_concordant_austin"] = d.loc[m, f"{b}_{sm}"]
    metrics = [f"{b}_{sm}" for b in BASE for sm in SMOOTHERS] + \
              [f"{b}_concordant_austin" for b in BASE]

    point = d[d.b < 0]
    boot = d[d.b >= 0]
    B = int(boot.b.nunique()) if len(boot) else 0
    print(f"{len(files)} chunk files, {d.imp.nunique()} imputations, B = {B}")

    main_rows, con_rows = [], []
    for (split, h), g in point.groupby(["split", "horizon_y"]):
        n_com = N_SUBJECTS[split] - 1
        gb = boot[(boot.split == split) & (boot.horizon_y == h)]
        imps = sorted(g.imp.unique())
        models = sorted(g.model.unique())
        pv = {(m, k): g[(g.model == m) & (g.imp == k)] for m in models for k in imps}
        bv = {(m, k): gb[(gb.model == m) & (gb.imp == k)].sort_values("b")
              for m in models for k in imps}

        for m in models:
            sm = g[g.model == m].concordant.iloc[0]
            for met in metrics:
                qm = np.array([pv[(m, k)][met].iloc[0] if len(pv[(m, k)]) else np.nan
                               for k in imps], float)
                if not np.isfinite(qm).any():
                    continue
                bl = [bv[(m, k)][met].to_numpy(float) for k in imps if len(bv[(m, k)])]
                vm = np.array([np.nanvar(x, ddof=1) for x in bl], float) if bl \
                    else np.full(len(imps), np.nan)
                bp = np.concatenate(bl) if bl else np.array([np.nan])
                r = pool(qm, vm, bp, n_com)
                main_rows.append(dict(split=split, model=m,
                                      family="naive" if m in NAIVE else "competing",
                                      horizon_y=h, metric=met,
                                      concordant_smoother=sm, **r))

        for a, b_, tier, matched, note in CONTRASTS:
            if a not in models or b_ not in models:
                continue
            for met in metrics:
                qm = np.array([
                    (pv[(a, k)][met].iloc[0] - pv[(b_, k)][met].iloc[0])
                    if len(pv[(a, k)]) and len(pv[(b_, k)]) else np.nan
                    for k in imps], float)
                if not np.isfinite(qm).any():
                    continue
                dbl = []
                for k in imps:
                    xa, xb = bv[(a, k)], bv[(b_, k)]
                    if len(xa) and len(xb) and len(xa) == len(xb):
                        dbl.append(xa[met].to_numpy(float) - xb[met].to_numpy(float))
                vm = np.array([np.nanvar(x, ddof=1) for x in dbl], float) if dbl \
                    else np.full(len(imps), np.nan)
                bp = np.concatenate(dbl) if dbl else np.array([np.nan])
                r = pool(qm, vm, bp, n_com)
                excl = (np.isfinite(r["lo"]) and np.isfinite(r["hi"])
                        and (r["lo"] > 0 or r["hi"] < 0))
                con_rows.append(dict(split=split, contrast=f"{a} - {b_}", model_a=a,
                                     model_b=b_, tier=tier, learner_matched=matched,
                                     horizon_y=h, metric=met, **r,
                                     ci_excludes_zero=bool(excl), note=note))

    mnew = pd.DataFrame(main_rows)
    cnew = pd.DataFrame(con_rows)
    mnew.to_csv(OUT / "concordant_metrics.csv", index=False)
    cnew.to_csv(OUT / "concordant_contrasts.csv", index=False)

    for f, new in [(OUT / "main_metrics.csv", mnew),
                   (OUT / "paired_contrasts.csv", cnew)]:
        if f.exists():
            old = pd.read_csv(f)
            old = old[~old.metric.isin(new.metric.unique())]
            pd.concat([old, new], ignore_index=True).to_csv(f, index=False)
            print(f"appended {len(new)} rows to {f.name}")
        else:
            new.to_csv(f, index=False)

    # the headline: does the naive-versus-competing ICI difference survive?
    print("\ndelta ICI, cox_naive - csc_cox, by smoother:")
    for met in ["ici_smoothfg", "ici_smoothcsh", "ici_smoothnet",
                "ici_concordant_austin"]:
        s = cnew[(cnew.model_a == "cox_naive") & (cnew.model_b == "csc_cox")
                 & (cnew.metric == met) & (cnew.split == "temporal")]
        if len(s):
            r = s.iloc[0]
            print(f"  {met:26s} {r.est:+.5f} ({r.lo:+.5f} to {r.hi:+.5f})  "
                  f"{'excludes zero' if r.ci_excludes_zero else 'INCLUDES ZERO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
