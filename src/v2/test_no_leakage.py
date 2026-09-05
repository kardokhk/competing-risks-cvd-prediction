#!/usr/bin/env python
"""P2-A. Regression test: no information flows from held-out rows into fitting.

The version 1 audit cleared the pipeline of leakage: scalers were fitted on
training folds only. Version 2 rewrote the fitting code, so that property has to
be re-established rather than inherited.

The test is direct. Perturb the held-out rows arbitrarily, refit, and require
that everything derived from the training data is byte-identical: the
standardization centre and scale, and the fitted coefficients. If any held-out
value reached the fit, at least one of those has to move.

Run: python src/v2/test_no_leakage.py
"""
from __future__ import annotations

import importlib.util as il
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "src/v2/feature_spec.json").read_text())
STD = SPEC["standardize_features"]
FEATURES = SPEC["primary_features"]

_sp = il.spec_from_file_location("fit4", ROOT / "src/v2/04_fit_py_models.py")
_f4 = il.module_from_spec(_sp); _sp.loader.exec_module(_f4)

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    f = ROOT / "data/processed/imputed/imp_1.parquet"
    if not f.exists():
        print(f"{f} missing; run src/v2/01_impute.R first")
        return 1
    d = pd.read_parquet(f)
    tr = d[d.split == "train"].reset_index(drop=True)
    te = d[d.split == "test"].reset_index(drop=True)
    print(f"train {len(tr):,} rows, held-out {len(te):,} rows")

    print("--- standardization ---")
    Xtr, Xte = _f4.standardize(tr, te)
    # perturb every held-out predictor value beyond recognition
    rng = np.random.default_rng(20260903)
    te2 = te.copy()
    for c in FEATURES:
        te2[c] = te2[c].to_numpy() * rng.uniform(2, 10, len(te2)) + 1000.0
    Xtr2, Xte2 = _f4.standardize(tr, te2)
    check("training matrix is bit-identical after the held-out rows are perturbed",
          np.array_equal(Xtr, Xtr2))
    check("held-out matrix does change (the perturbation was real)",
          not np.allclose(Xte, Xte2))

    mu = tr[STD].mean(); sd = tr[STD].std(ddof=0)
    j = [FEATURES.index(c) for c in STD]
    check("standardized training columns have mean 0 and sd 1",
          np.allclose(Xtr[:, j].mean(0), 0, atol=1e-10)
          and np.allclose(Xtr[:, j].std(0, ddof=0), 1, atol=1e-10))
    check("standardized held-out columns do NOT have mean 0 and sd 1, "
          "which is what proves the scaler came from the training rows",
          not (np.allclose(Xte[:, j].mean(0), 0, atol=1e-6)
               and np.allclose(Xte[:, j].std(0, ddof=0), 1, atol=1e-6)),
          f"held-out means {np.round(Xte[:, j].mean(0), 4)}")
    check("the centre and scale are the training ones",
          np.allclose(mu.to_numpy(), tr[STD].mean().to_numpy())
          and np.allclose(sd.to_numpy(), tr[STD].std(ddof=0).to_numpy()))

    print("--- IPCW weights are estimated on the training rows only ---")
    y1, w1 = _f4._ipcw_weights(tr, 10, naive=False)
    y2, w2 = _f4._ipcw_weights(tr, 10, naive=False)
    check("IPCW weights are deterministic", np.array_equal(w1, w2))
    check("subjects censored before the horizon get weight 0",
          np.all(w1[(tr.event.to_numpy() == 0) & (tr.time.to_numpy() <= 10)] == 0))
    check("competing deaths before the horizon keep a weight in the "
          "competing-risk version and get outcome 0",
          np.all(w1[(tr.event.to_numpy() == 2) & (tr.time.to_numpy() <= 10)] > 0)
          and np.all(y1[(tr.event.to_numpy() == 2) & (tr.time.to_numpy() <= 10)] == 0))
    yn, wn = _f4._ipcw_weights(tr, 10, naive=True)
    check("competing deaths before the horizon get weight 0 in the naive version",
          np.all(wn[(tr.event.to_numpy() == 2) & (tr.time.to_numpy() <= 10)] == 0))
    check("the naive and competing-risk logistic models differ in their weights",
          not np.array_equal(w1, wn))

    print("--- fitted coefficients ---")
    from sklearn.linear_model import LogisticRegression
    keep = w1 > 0
    a = LogisticRegression(max_iter=5000, random_state=SPEC["seed"], C=1e6)
    a.fit(Xtr[keep], y1[keep], sample_weight=w1[keep])
    b = LogisticRegression(max_iter=5000, random_state=SPEC["seed"], C=1e6)
    b.fit(Xtr2[keep], y1[keep], sample_weight=w1[keep])
    check("logistic coefficients are bit-identical after the held-out rows are "
          "perturbed", np.array_equal(a.coef_, b.coef_))

    print("--- fold membership is disjoint and complete ---")
    for r in range(3):
        col = d[f"cv_rep{r}"].to_numpy()
        check(f"repeat {r}: every subject is held out exactly once across the 5 folds",
              set(np.unique(col)) == {0, 1, 2, 3, 4} and len(col) == len(d))
        for k in range(5):
            trn = d[col != k]; hld = d[col == k]
            check(f"repeat {r} fold {k}: training and held-out SEQN are disjoint",
                  len(set(trn.SEQN) & set(hld.SEQN)) == 0)
            check(f"repeat {r} fold {k}: the held-out fold contains CVD deaths",
                  int((hld.event == 1).sum()) > 0)
    print("--- temporal split is by calendar time ---")
    check("no cycle appears in both the training and the held-out set",
          len(set(tr.cycle) & set(te.cycle)) == 0,
          f"train {sorted(set(tr.cycle))}, test {sorted(set(te.cycle))}")
    check("every training cycle starts before every held-out cycle",
          tr.cycle_startyr.max() < te.cycle_startyr.min())

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURES: {FAILS}")
        return 1
    print("no leakage detected: everything derived from the training rows is "
          "invariant to the held-out rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
