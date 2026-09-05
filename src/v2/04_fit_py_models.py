#!/usr/bin/env python
"""P2-A step 04. Fit the Python-side model families on every imputed dataset.

  gbs_naive     XGBoost gradient-boosted Cox survival, risk = 1 - S(t) from a
                Breslow baseline cumulative hazard
  logistic      IPCW binomial regression, competing death treated as CENSORING
  logistic_cr   IPCW binomial regression, competing death treated as a COMPETING
                EVENT (outcome 0, weight retained)

**The logistic family, and why version 1's version was not usable.**
Version 1 built a binary 10-year label as: 1 if CVD death by 10 y, 0 if followed
to 10 y, and missing otherwise, then fitted an unweighted logistic regression on
the non-missing rows. That drops everyone censored before 10 y and everyone who
died of a competing cause before 10 y, 18,276 of 35,309 rows on the complete-case
cohort. Those are not missing at random with respect to the outcome: they are
disproportionately the late cycles (administratively censored) and the old (who
die of competing causes). The resulting estimand is the CVD risk among people who
either survive a decade or die of cardiovascular disease, which is not a
prediction target.

The fix is inverse-probability-of-censoring weighting, the direct binomial
regression of Scheike, Zhang and Gerds. Define Y = 1 if a CVD death occurred by
t. Weight a subject who had any observed event before t by 1/G(T-), a subject
followed past t by 1/G(t), and a subject censored before t by 0. G is the
Kaplan-Meier censoring distribution estimated on the TRAINING rows and
stratified by NHANES cycle, because censoring here is administrative.

That construction also produces the matched pair the benchmark was missing.
Whether a competing death is censoring or an event is the only difference
between the two logistic models:

  logistic      G counts competing death as censoring, and a competing death
                before t gets weight 0. Target: 1 - S_1(t), the naive quantity.
  logistic_cr   G counts only true censoring, and a competing death before t is
                kept with Y = 0 and its IPCW weight. Target: CIF_1(t).

Same learner, same features, same weights machinery, opposite handling of the
competing risk. This is a third genuine learner-matched pair alongside
cox_naive/csc_cox and rsf_naive/rsf_cr, and it is an addition to the seven
families in the original plan. It is documented as such.

Because the model is horizon-specific, one fit is made per horizon.

**The gradient-boosting implementation changed, and why.** Version 1 used
`sksurv.ensemble.GradientBoostingSurvivalAnalysis(loss="coxph")`. Its Cox
partial-likelihood gradient is quadratic in the number of training rows: measured
here at 400 trees, 18 s at n = 2,000, 67 s at n = 4,000, 270 s at n = 8,000, which
extrapolates to about 76 minutes for one cross-validation fit at n = 32,921. Four
hundred and eighty such fits is roughly 600 core-hours for one family that is not
part of any learner-matched pair.

Version 2 fits the same model, gradient boosting on the Cox partial likelihood,
with `xgboost` (`objective="survival:cox"`), whose histogram-based splitting is
n log n. One fit takes 1.1 s instead of 894 s. The survival function is recovered
with a Breslow baseline cumulative hazard estimated on the training rows from the
fitted linear predictor, and `risk_t = 1 - exp(-H0(t) exp(f(x)))`. On the temporal
split at 10 y this gives mean predicted risk 0.0500 and a truncated concordance of
0.822, against 0.0491 and 0.824 for the Cox model on the same rows, so the change
is one of implementation and not of behaviour. It is recorded as a change from
version 1 because the learner library differs.

No hyperparameter tuning is performed for any family in version 2. Version 1
tuned the gradient-boosting `n_estimators` on an inner split for that family
only, which made the comparison asymmetric. Version 2 pre-specifies the
hyperparameters for every family and states them here.

Run:
    python src/v2/04_fit_py_models.py --imps 1-20 --splits temporal,cv --ncores 128
    python src/v2/04_fit_py_models.py --imps 1-2 --splits temporal --ncores 2   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v2"))
from eval_lib_v2 import censor_surv_stratified  # noqa: E402

SPEC = json.loads((ROOT / "src" / "v2" / "feature_spec.json").read_text())
FEATURES = SPEC["primary_features"]
STD = SPEC["standardize_features"]
IMPDIR = ROOT / "data" / "processed" / "imputed"
PARTS = ROOT / "results" / "metrics_v2" / "predictions" / "parts"
SEED = SPEC["seed"]
HORIZONS = [5, 10, 15]
REPEATS = [0, 1, 2]
FOLDS = [0, 1, 2, 3, 4]

# Pre-specified, not tuned. `eta`, `max_depth` and `subsample` are version 1's
# learning rate, depth and subsample; `min_child_weight` mirrors its
# `min_samples_leaf`.
GBS_ROUNDS = 400
GBS_PARAMS = dict(objective="survival:cox", eta=0.05, max_depth=3, subsample=0.8,
                  min_child_weight=15, nthread=1, verbosity=0)

MODELS = {
    "gbs_naive":   ("naive", "GradientBoostingSurvival"),   # xgboost survival:cox
    "logistic":    ("naive", "IPCWLogistic"),
    "logistic_cr": ("competing", "IPCWLogisticCR"),
}


def standardize(tr: pd.DataFrame, te: pd.DataFrame):
    """z-score fitted on the training rows of THIS fold only. No leakage."""
    mu = tr[STD].mean()
    sd = tr[STD].std(ddof=0).replace(0, 1.0)
    a = tr[FEATURES].copy(); b = te[FEATURES].copy()
    a[STD] = (a[STD] - mu) / sd
    b[STD] = (b[STD] - mu) / sd
    return a.to_numpy(float), b.to_numpy(float)


# --------------------------------------------------------------------------- #
def _breslow_baseline(time, event, lp):
    """Breslow baseline cumulative hazard from a fitted log hazard ratio.

    H0(u_k) = sum_{j <= k} d_j / sum_{i: T_i >= u_j} exp(lp_i), with d_j the
    number of events at u_j. Estimated on the TRAINING rows only.
    Returns (unique times, H0 at those times).
    """
    time = np.asarray(time, float); event = np.asarray(event, int)
    w = np.exp(np.asarray(lp, float) - np.max(lp))     # shift for numerical safety
    o = np.argsort(time, kind="mergesort")
    ts, es, ws = time[o], event[o], w[o]
    u, inv = np.unique(ts, return_inverse=True)
    d = np.bincount(inv, weights=(es == 1).astype(float), minlength=len(u))
    rev = np.cumsum(ws[::-1])[::-1]                    # sum of weights from i onwards
    at_risk = rev[np.searchsorted(ts, u, side="left")]
    dH = np.divide(d, at_risk, out=np.zeros_like(d), where=at_risk > 0)
    return u, np.cumsum(dH), float(np.max(lp))


def fit_gbs(tr, te, Xtr, Xte, seed):
    """Gradient boosting on the Cox partial likelihood, competing death censored.

    xgboost's `survival:cox` takes a signed time as the label: positive for an
    event, negative for a censored observation. `output_margin=True` returns the
    linear predictor f(x); the default `predict` would return exp(f(x)).
    """
    import xgboost as xgb
    t = tr["time"].to_numpy(float)
    e = tr["event_naive"].to_numpy(int)
    label = np.where(e == 1, t, -t)
    params = dict(GBS_PARAMS); params["seed"] = int(seed)
    bst = xgb.train(params, xgb.DMatrix(Xtr, label=label), num_boost_round=GBS_ROUNDS)
    lp_tr = bst.predict(xgb.DMatrix(Xtr), output_margin=True)
    lp_te = bst.predict(xgb.DMatrix(Xte), output_margin=True)
    u, H0, shift = _breslow_baseline(t, e, lp_tr)
    out = np.full((len(te), 3), np.nan)
    for j, h in enumerate(HORIZONS):
        k = np.searchsorted(u, h, side="right") - 1
        if k >= 0:
            out[:, j] = 1.0 - np.exp(-H0[k] * np.exp(lp_te - shift))
    return out


def _ipcw_weights(tr, t, naive):
    """IPCW weights and binary outcome for the direct binomial regression at t."""
    T = tr["time"].to_numpy(float)
    E = tr["event"].to_numpy(int)
    cyc = tr["cycle"].to_numpy()
    G_T = censor_surv_stratified(T, E, cyc, T, naive=naive)
    G_t = censor_surv_stratified(T, E, cyc, np.full(len(T), float(t)), naive=naive)
    w = np.zeros(len(T))
    y = np.zeros(len(T))
    past = T > t
    w[past] = 1.0 / G_t[past]
    if naive:
        # competing death is censoring: only cause-1 events before t carry weight
        ev = (E == 1) & (T <= t)
    else:
        # competing death is an event with outcome 0 and keeps its weight
        ev = (E > 0) & (T <= t)
    w[ev] = 1.0 / G_T[ev]
    y[(E == 1) & (T <= t)] = 1.0
    # subjects censored before t get weight 0 by construction
    return y, w


def fit_logistic(tr, te, Xtr, Xte, seed, naive: bool):
    from sklearn.linear_model import LogisticRegression
    out = np.full((len(te), 3), np.nan)
    for j, t in enumerate(HORIZONS):
        if tr["time"].max() < t:
            continue
        y, w = _ipcw_weights(tr, t, naive)
        keep = w > 0
        if y[keep].sum() < 10 or len(np.unique(y[keep])) < 2:
            continue
        clf = LogisticRegression(max_iter=5000, random_state=seed, C=1e6)
        clf.fit(Xtr[keep], y[keep], sample_weight=w[keep])
        out[:, j] = clf.predict_proba(Xte)[:, 1]
    return out


FITTERS = {
    "gbs_naive": lambda tr, te, a, b, s: fit_gbs(tr, te, a, b, s),
    "logistic": lambda tr, te, a, b, s: fit_logistic(tr, te, a, b, s, naive=True),
    "logistic_cr": lambda tr, te, a, b, s: fit_logistic(tr, te, a, b, s, naive=False),
}


# --------------------------------------------------------------------------- #
_CACHE: dict[int, pd.DataFrame] = {}


def get_imp(k: int) -> pd.DataFrame:
    if k not in _CACHE:
        _CACHE[k] = pd.read_parquet(IMPDIR / f"imp_{k}.parquet")
    return _CACHE[k]


def tag_of(task) -> str:
    m, k, sp, r, f = task
    return (f"{m}__temporal__imp{k}.parquet" if sp == "temporal"
            else f"{m}__cvr{r}f{f}__imp{k}.parquet")


def run_task(task):
    m, k, sp, r, f = task
    t0 = _time.time()
    d = get_imp(k)
    if sp == "temporal":
        tr = d[d.split == "train"]; te = d[d.split == "test"]
        seed = SEED + 1000 * k
    else:
        col = f"cv_rep{r}"
        tr = d[d[col] != f]; te = d[d[col] == f]
        seed = SEED + 1000 * k + 100 * r + f
    Xtr, Xte = standardize(tr, te)
    risk = FITTERS[m](tr, te, Xtr, Xte, seed)
    if sp == "temporal":
        risk[:, 2] = np.nan          # 15 y not estimable on 2007-2010
    risk = np.clip(risk, 0.0, 1.0)
    fam, learner = MODELS[m]
    o = pd.DataFrame({
        "SEQN": te.SEQN.to_numpy("int64"), "cycle": te.cycle.astype(str).to_numpy(),
        "time": te.time.to_numpy(float), "event": te.event.to_numpy("int64"),
        "risk_5": risk[:, 0], "risk_10": risk[:, 1], "risk_15": risk[:, 2],
        "model_name": m, "family": fam, "learner_class": learner,
        "imp": np.int64(k)})
    if sp == "cv":
        o["cv_rep"] = np.int64(r); o["cv_fold"] = np.int64(f)
    PARTS.mkdir(parents=True, exist_ok=True)
    o.to_parquet(PARTS / tag_of(task), index=False)
    return dict(task=tag_of(task), n=len(o),
                mean_risk10=float(np.nanmean(o.risk_10)), secs=_time.time() - t0)


def parse_range(s):
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imps", default="1-20")
    ap.add_argument("--splits", default="temporal,cv")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--ncores", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()

    imps = parse_range(a.imps)
    splits = a.splits.split(",")
    models = a.models.split(",")
    tasks = []
    for k in imps:
        for m in models:
            if "temporal" in splits:
                tasks.append((m, k, "temporal", -1, -1))
            if "cv" in splits:
                for r in REPEATS:
                    for f in FOLDS:
                        tasks.append((m, k, "cv", r, f))
    PARTS.mkdir(parents=True, exist_ok=True)
    done = {p.name for p in PARTS.glob("*.parquet")}
    todo = [t for t in tasks if tag_of(t) not in done]
    print(f"P2-A step 04: {len(tasks)} tasks, {len(tasks)-len(todo)} already on "
          f"disk, {len(todo)} to run on {a.ncores} cores", flush=True)

    t0 = _time.time()
    rows, errs = [], []
    if a.ncores > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=a.ncores) as ex:
            futs = {ex.submit(run_task, t): t for t in todo}
            for i, fu in enumerate(as_completed(futs)):
                try:
                    rows.append(fu.result())
                except Exception as e:
                    errs.append((tag_of(futs[fu]), repr(e)))
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(todo)} done, {_time.time()-t0:.0f}s",
                          flush=True)
    else:
        for t in todo:
            try:
                rows.append(run_task(t))
            except Exception as e:
                errs.append((tag_of(t), repr(e)))
    for tg, e in errs:
        print("FAILED:", tg, "|", e, flush=True)
    if rows:
        log = pd.DataFrame(rows)
        log.to_csv(ROOT / "results/metrics_v2/fit_log_py.csv", index=False)
        print(f"median fit {log.secs.median():.1f}s, total "
              f"{log.secs.sum()/3600:.2f} core-hours")
    print(f"DONE step 04 in {(_time.time()-t0)/60:.1f} min")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
