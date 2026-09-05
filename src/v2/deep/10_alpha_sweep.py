#!/usr/bin/env python
"""P2-C stage A. Sweep DeepHit's likelihood/ranking weight alpha.

pycox parameterises the DeepHit objective as

    loss = alpha * nll_pmf_cr + (1 - alpha) * rank_loss_deephit_cr(sigma)

so alpha = 0 is a pure ranking loss and alpha = 1 a pure likelihood. The prior work
used alpha = 0.2, that is 80% of the objective on the ranking term. If the mechanism
claim is right, calibration improves monotonically as alpha rises while discrimination
stays roughly flat.

Two protocols, both written to the same file and distinguished by the `protocol` column:

  inner_oof      PRIMARY. Out-of-fold predictions over the four training cycles
                 (1999-2006, N = 14,551), 5 stratified inner folds built from the
                 training rows only. No temporal test row is read.
  temporal_test  CONFIRMATORY, labelled as such. Fit on all training cycles, evaluate
                 on the 2007-2010 test rows. No configuration is chosen from it.

Neural Fine-Gray is run under the same protocols as a reference line; it has no alpha
and is recorded with alpha = NaN.

Output: results/deep/alpha_sweep.csv, appended row by row so a killed job resumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deep_lib as D  # noqa: E402

OUT = D.ROOT / "results" / "deep" / "alpha_sweep.csv"
KEY = ["model", "protocol", "alpha_s", "sigma_s", "seed", "imp"]

ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
SIGMAS = [0.1]                       # sigma held at the pycox default for the main sweep
SIGMA_EXTRA = [0.05, 0.25, 0.5]      # a secondary sigma grid at three alphas
SIGMA_EXTRA_ALPHAS = [0.0, 0.2, 1.0]
SEEDS = [0, 1, 2, 3, 4]
IMPS = [1, 2, 3]
BASE_SEED = 771000                   # offset so sweep seeds cannot collide with fit seeds


def task_seed(seed_i: int, imp: int) -> int:
    return D.PROJECT_SEED + BASE_SEED + 131 * seed_i + 7919 * imp


def run_one(t: dict) -> dict | None:
    imp = int(t["imp"])
    d = D.load_imp(imp)
    tr = d[d.split == "train"].reset_index(drop=True)
    cfg = dict(t["cfg"])
    seed = task_seed(int(t["seed"]), imp)
    try:
        if t["protocol"] == "inner_oof":
            folds = D.inner_folds(tr)
            oof = D.oof_train(t["model"], tr, folds, cfg, seed)
            ev_df = oof
        else:
            te = d[d.split == "test"].reset_index(drop=True)
            r, _ = D.fit_predict(t["model"], tr, te, cfg, seed)
            ev_df = te[D.META].copy()
            ev_df["risk_10"] = r[:, 1]
        m = D.metrics(ev_df.time.values, ev_df.event.values, ev_df.cycle.values,
                      ev_df.risk_10.values)
        m["brier"] = D.ipcw_brier(ev_df.time.values, ev_df.event.values,
                                  ev_df.risk_10.values)
    except Exception as e:  # a failed configuration is recorded, not silently dropped
        m = dict(error=f"{type(e).__name__}: {e}")
    row = dict(model=t["model"], protocol=t["protocol"], alpha_s=t["alpha_s"],
               sigma_s=t["sigma_s"], seed=t["seed"], imp=imp, seed_used=seed,
               horizon=D.PRIMARY_H, n_eval=len(ev_df) if "error" not in m else 0)
    row.update(m)
    return row


def build_tasks() -> list[dict]:
    tasks = []
    grid = [(a, s) for a in ALPHAS for s in SIGMAS]
    grid += [(a, s) for a in SIGMA_EXTRA_ALPHAS for s in SIGMA_EXTRA]
    for protocol in ("inner_oof", "temporal_test"):
        imps = IMPS if protocol == "inner_oof" else [1]
        for a, s in grid:
            for sd in SEEDS:
                for imp in imps:
                    tasks.append(dict(
                        model="deephit", protocol=protocol,
                        alpha_s=f"{a:g}", sigma_s=f"{s:g}", seed=sd, imp=imp,
                        cfg=dict(alpha=a, sigma=s)))
        for sd in SEEDS:
            for imp in imps:
                tasks.append(dict(model="nfg", protocol=protocol, alpha_s="NA",
                                  sigma_s="NA", seed=sd, imp=imp, cfg={}))
    return tasks


if __name__ == "__main__":
    tasks = build_tasks()
    print(f"alpha sweep: {len(tasks)} tasks", flush=True)
    D.run_tasks(run_one, tasks, OUT, KEY)
    if OUT.exists():
        df = pd.read_csv(OUT)
        q = df[(df.model == "deephit") & (df.protocol == "inner_oof")]
        if len(q):
            g = q.groupby("alpha_s")[["eo", "ici", "calib_slope", "cindex", "auc"]].median()
            print(g.to_string(), flush=True)
