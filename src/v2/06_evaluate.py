#!/usr/bin/env python
"""P2-A step 06. Evaluate every model on every imputation, with a paired
shared-index bootstrap.

Metrics, per model per horizon, are the full Van Calster hierarchy as specified
in van Geloven et al., BMJ 2022;377:e069249 and implemented in
`src/v2/eval_lib_v2.py`. Nothing is computed and then left unreported: version 1
computed the calibration slope and reported only E/O and ICI, which are the two
levels that favoured its thesis.

**The shared-index bootstrap.** Replicate b resamples SUBJECTS with replacement
and every model sees exactly the same resample, so a contrast between two models
is a paired difference on a common index. The indices are generated from
`np.random.default_rng([20260903, split_id, b])`, which is a pure function of the
replicate number, so they are identical across models, across imputations and
across separate runs of this script without any index file having to be stored or
shipped between workers.

**Cross-validation and pseudoreplication.** For the cross-validated split, each
of the three repeats is a complete out-of-fold prediction for all 41,151
subjects. The statistic is the mean of the three per-repeat values, so repeat is
a nested factor. Version 1's `src/eval/run_eval_cv.R` stacked 5 repeats of some
models and 2 of others into one pool and treated the rows as independent, which
inflated the effective sample size roughly fivefold.

**Censoring weights** are estimated within NHANES cycle. Censoring here is
administrative and ranges from 0% to essentially 100% before 10 y across cycles,
so a marginal Kaplan-Meier averages hazards that do not overlap. The marginal
version of each discrimination metric is computed alongside so the size of the
change can be reported.

Output, one file per (split, imputation, chunk), so an interrupted run resumes:
    results/metrics_v2/boot/<split>__imp<k>__chunk<c>.npz
        boot   float32 (n_chunk, n_model, n_horizon, n_metric)
        point  float32 (n_model, n_horizon, n_metric)   from the unbootstrapped data
    plus results/metrics_v2/boot/schema.json naming every axis.

Run:
    python src/v2/06_evaluate.py --splits temporal,cv --imps 1-20 -B 2000 --ncores 128
    python src/v2/06_evaluate.py --splits temporal --imps 1-2 -B 50 --ncores 2   # smoke
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
from eval_lib_v2 import (aj_cif_at, at_risk_fraction, calib_intercept_slope,  # noqa: E402
                         censor_surv, censor_surv_stratified, cr_cindex,
                         flexible_calibration, ipcw_auc, km_net_risk_at,
                         net_benefit, net_benefit_all, pseudo_cif,
                         pseudo_net_risk)

SPEC = json.loads((ROOT / "src" / "v2" / "feature_spec.json").read_text())
SEED = SPEC["seed"]
PRED = ROOT / "results" / "metrics_v2" / "predictions"
BOOT = ROOT / "results" / "metrics_v2" / "boot"

HORIZONS = [5.0, 10.0, 15.0]
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.20]
SPLIT_ID = {"temporal": 0, "cv": 1}

# Two calibration targets are computed for every model.
#
#   *_shared      the observed probability is the Aalen-Johansen cumulative
#                 incidence for every model, naive and competing alike. This is
#                 what version 1 did and what makes the two families directly
#                 comparable on one scale.
#   *_concordant  the observed probability matches the estimand of the model
#                 being assessed: the Aalen-Johansen cumulative incidence for a
#                 competing-risk model, and 1 - S(t) with the competing event
#                 censored for a naive model, which is the quantity a naive model
#                 is actually estimating.
#
# Austin and Putter (Stat Med 2026;45:e70468, doi:10.1002/sim.70468) recommend
# that the smoothed event probabilities used for ICI, E50 and E90 come from a
# model concordant with the type of model whose calibration is being assessed. A
# naive model assessed against the Aalen-Johansen cumulative incidence is being
# assessed against a different estimand from the one it targets, which flatters
# the competing-risk arm by construction. Reporting only the shared version would
# put the paper's central ICI contrast at the mercy of that choice, so both are
# carried through the bootstrap and both are contrasted.
METRICS = (["mean_pred", "obs_cif", "eo", "calib_intercept", "calib_slope",
            "ici", "e50", "e90",
            "obs_concordant", "eo_concordant",
            "calib_intercept_concordant", "calib_slope_concordant",
            "ici_concordant", "e50_concordant", "e90_concordant",
            "auc", "cindex", "auc_marginal_ipcw",
            "cindex_marginal_ipcw", "at_risk_fraction"]
           + [f"nb_at_{t:g}" for t in THRESHOLDS]
           + [f"frac_above_{t:g}" for t in THRESHOLDS]
           + [f"nb_all_at_{t:g}" for t in THRESHOLDS])
MIDX = {m: i for i, m in enumerate(METRICS)}
NM = len(METRICS)

# The order here fixes the model axis of every saved array.
MODEL_ORDER = ["logistic", "cox_naive", "rsf_naive", "gbs_naive",
               "logistic_cr", "csc_cox", "fine_gray", "rsf_cr",
               "deephit", "nfg"]


# --------------------------------------------------------------------------- #
def load_split(split: str, imp: int):
    """Return (base, risks) for one imputation.

    base   DataFrame with SEQN, cycle, time, event, and for cv a `cv_rep` column,
           ordered canonically so that a bootstrap index is meaningful.
    risks  dict model -> array (n_rows, 3) aligned to `base`.
    """
    files = sorted(PRED.glob(f"*__{split}__imp{imp}.parquet"))
    if not files:
        raise FileNotFoundError(f"no prediction files for {split} imp {imp}")
    base = None
    risks: dict[str, np.ndarray] = {}
    for f in files:
        d = pd.read_parquet(f)
        key = ["cv_rep", "SEQN"] if split == "cv" else ["SEQN"]
        d = d.sort_values(key).reset_index(drop=True)
        if base is None:
            cols = ["SEQN", "cycle", "time", "event"] + (["cv_rep"] if split == "cv" else [])
            base = d[cols].copy()
        else:
            if not np.array_equal(base.SEQN.to_numpy(), d.SEQN.to_numpy()):
                raise ValueError(f"{f.name} rows do not align with the other models")
        risks[str(d.model_name.iloc[0])] = d[["risk_5", "risk_10", "risk_15"]].to_numpy(float)
    return base, risks


def subject_index(base: pd.DataFrame, split: str):
    """Canonical subject list and, for cv, the row positions of each subject in
    each repeat, so a resampled subject carries all of its repeats."""
    if split == "temporal":
        return base.SEQN.to_numpy(), [np.arange(len(base))]
    reps = sorted(base.cv_rep.unique())
    subs = np.unique(base.SEQN.to_numpy())
    pos = []
    for r in reps:
        m = base.cv_rep.to_numpy() == r
        idx = np.where(m)[0]
        order = np.argsort(base.SEQN.to_numpy()[idx], kind="mergesort")
        pos.append(idx[order])      # positions of subs, in the order of `subs`
    return subs, pos


def boot_index(split: str, b: int, n: int) -> np.ndarray:
    """Bootstrap positions for replicate b. A pure function of (split, b), so
    every model and every imputation sees the identical resample."""
    rng = np.random.default_rng([SEED, SPLIT_ID[split], b])
    return rng.integers(0, n, n)


# --------------------------------------------------------------------------- #
def metrics_one_set(time, event, cycle, risks_sub, hidx):
    """All metrics for every model on one dataset at one horizon index.

    Quantities that do not depend on the model (the observed CIF, the censoring
    distribution, the pseudo-observations) are computed once and shared, which is
    both faster and necessary: a paired contrast must difference two models
    against the same observed quantity, not against two separate estimates of it.
    """
    t = HORIZONS[hidx]
    out = np.full((len(MODEL_ORDER), NM), np.nan, dtype=np.float64)
    n = len(time)
    if n == 0 or not np.any((event == 1) & (time <= t)):
        return out

    obs = aj_cif_at(time, event, t, 1)
    arf = at_risk_fraction(time, event, t)
    ps = pseudo_cif(time, event, t, 1)
    # concordant target for the naive family: 1 - S(t) with the competing event
    # censored, and its exact jackknife pseudo-observations
    obs_net = km_net_risk_at(time, event, t, 1)
    ps_net = pseudo_net_risk(time, event, t, 1)
    G_T = censor_surv_stratified(time, event, cycle, time)
    G_t = censor_surv_stratified(time, event, cycle, np.full(n, t))
    G_T_marg = censor_surv(time, event, time)
    G_t_marg = censor_surv(time, event, np.full(n, t))
    # Treat-all is a property of the sample, not of any model, so it is computed
    # once per resample and copied onto every model's row. It is
    # CIF - (1 - CIF) * pt/(1 - pt): zero exactly at pt equal to the observed
    # cumulative incidence and negative above it. Version 1's docs/table1.md gave
    # treat-all at 7.5% as +0.0059 when the value is negative.
    nb_all = {th: net_benefit_all(time, event, th, t) for th in THRESHOLDS}

    naive_models = {"logistic", "cox_naive", "rsf_naive", "gbs_naive"}
    for mi, model in enumerate(MODEL_ORDER):
        r = risks_sub.get(model)
        if r is None:
            continue
        p = r[:, hidx]
        if not np.isfinite(p).any():
            continue
        ok = np.isfinite(p)
        pp = np.clip(p[ok], 1e-8, 1 - 1e-8)
        tt, ee, cc, pssub = time[ok], event[ok], cycle[ok], ps[ok]
        is_naive = model in naive_models
        ps_conc = (ps_net[ok] if is_naive else ps[ok])
        obs_conc = obs_net if is_naive else obs
        row = out[mi]
        row[MIDX["mean_pred"]] = float(np.mean(pp))
        row[MIDX["obs_cif"]] = obs
        row[MIDX["eo"]] = float(np.mean(pp)) / obs if obs > 0 else np.nan
        row[MIDX["at_risk_fraction"]] = arf
        icp, slp = calib_intercept_slope(pssub, pp)
        row[MIDX["calib_intercept"]] = icp
        row[MIDX["calib_slope"]] = slp
        fc = flexible_calibration(pssub, pp, nknots=3)
        row[MIDX["ici"]] = fc["ici"]; row[MIDX["e50"]] = fc["e50"]
        row[MIDX["e90"]] = fc["e90"]
        # concordant-target versions
        row[MIDX["obs_concordant"]] = obs_conc
        row[MIDX["eo_concordant"]] = (float(np.mean(pp)) / obs_conc
                                      if obs_conc > 0 else np.nan)
        ic2, sl2 = calib_intercept_slope(ps_conc, pp)
        row[MIDX["calib_intercept_concordant"]] = ic2
        row[MIDX["calib_slope_concordant"]] = sl2
        fc2 = flexible_calibration(ps_conc, pp, nknots=3)
        row[MIDX["ici_concordant"]] = fc2["ici"]
        row[MIDX["e50_concordant"]] = fc2["e50"]
        row[MIDX["e90_concordant"]] = fc2["e90"]
        row[MIDX["auc"]] = ipcw_auc(tt, ee, pp, t, G_T[ok], G_t[ok])
        row[MIDX["cindex"]] = cr_cindex(tt, ee, pp, t, G_T[ok])
        row[MIDX["auc_marginal_ipcw"]] = ipcw_auc(tt, ee, pp, t, G_T_marg[ok], G_t_marg[ok])
        row[MIDX["cindex_marginal_ipcw"]] = cr_cindex(tt, ee, pp, t, G_T_marg[ok])
        for th in THRESHOLDS:
            row[MIDX[f"nb_at_{th:g}"]] = net_benefit(tt, ee, pp, th, t)
            row[MIDX[f"frac_above_{th:g}"]] = float(np.mean(pp >= th))
            row[MIDX[f"nb_all_at_{th:g}"]] = nb_all[th]
    return out


def metrics_dataset(base, risks, split, rows_by_rep, sel=None):
    """Metrics averaged over cross-validation repeats. `sel` selects subjects."""
    acc = np.zeros((len(MODEL_ORDER), len(HORIZONS), NM))
    cnt = np.zeros((len(MODEL_ORDER), len(HORIZONS), NM))
    time_all = base.time.to_numpy(float)
    event_all = base.event.to_numpy(np.int64)
    cyc_all = base.cycle.to_numpy()
    for rows in rows_by_rep:
        take = rows if sel is None else rows[sel]
        rs = {m: v[take] for m, v in risks.items()}
        for hi in range(len(HORIZONS)):
            r = metrics_one_set(time_all[take], event_all[take], cyc_all[take], rs, hi)
            good = np.isfinite(r)
            acc[:, hi, :] += np.where(good, r, 0.0)
            cnt[:, hi, :] += good
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


# --------------------------------------------------------------------------- #
def run_chunk(args):
    split, imp, c0, c1, B = args
    out = BOOT / f"{split}__imp{imp}__chunk{c0}.npz"
    if out.exists():
        return dict(task=out.name, skipped=True, secs=0.0)
    t0 = _time.time()
    base, risks = load_split(split, imp)
    subs, rows_by_rep = subject_index(base, split)
    n = len(subs)
    point = metrics_dataset(base, risks, split, rows_by_rep, sel=None)
    bootarr = np.full((c1 - c0, len(MODEL_ORDER), len(HORIZONS), NM), np.nan,
                      dtype=np.float32)
    for j, b in enumerate(range(c0, c1)):
        sel = boot_index(split, b, n)
        bootarr[j] = metrics_dataset(base, risks, split, rows_by_rep, sel=sel)
    BOOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, boot=bootarr, point=point.astype(np.float32))
    return dict(task=out.name, skipped=False, secs=_time.time() - t0)


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
    ap.add_argument("--splits", default="temporal,cv")
    ap.add_argument("--imps", default="1-20")
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--ncores", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()

    BOOT.mkdir(parents=True, exist_ok=True)
    (BOOT / "schema.json").write_text(json.dumps({
        "axes": ["bootstrap_replicate", "model", "horizon", "metric"],
        "models": MODEL_ORDER, "horizons": HORIZONS, "metrics": METRICS,
        "thresholds": THRESHOLDS, "B": a.B, "seed": SEED,
        "boot_index_rule": "np.random.default_rng([seed, split_id, b]).integers(0, n, n)",
        "split_id": SPLIT_ID,
        "cv_repeats_averaged": True,
    }, indent=2))

    tasks = []
    for split in a.splits.split(","):
        for k in parse_range(a.imps):
            for c0 in range(0, a.B, a.chunk):
                tasks.append((split, k, c0, min(c0 + a.chunk, a.B), a.B))
    print(f"P2-A step 06: {len(tasks)} chunk tasks, B={a.B}, {a.ncores} cores",
          flush=True)

    t0 = _time.time()
    done = skipped = 0
    errs = []
    if a.ncores > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=a.ncores) as ex:
            futs = {ex.submit(run_chunk, t): t for t in tasks}
            for i, fu in enumerate(as_completed(futs)):
                try:
                    r = fu.result()
                    done += 1; skipped += int(r["skipped"])
                except Exception as e:
                    errs.append((futs[fu], repr(e)))
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(tasks)} chunks, {_time.time()-t0:.0f}s",
                          flush=True)
    else:
        for t in tasks:
            try:
                r = run_chunk(t); done += 1; skipped += int(r["skipped"])
            except Exception as e:
                errs.append((t, repr(e)))
    for t, e in errs:
        print("FAILED:", t, "|", e, flush=True)
    print(f"DONE step 06: {done} chunks ({skipped} already on disk), "
          f"{len(errs)} failures, {(_time.time()-t0)/60:.1f} min")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
