#!/usr/bin/env python
"""P2-C stage C. Seed stability of the final and default configurations.

For each model and each of three configurations, refit under 12 random-initialisation
seeds and record the whole distribution of E/O, ICI, calibration slope and C-index
rather than one fit. Configurations:

  default       what the prior work ran (DeepHit alpha = 0.2, sigma = 0.1)
  tuned_brier   selected on the inner out-of-fold IPCW Brier score, the primary
  tuned_cindex  selected on the inner out-of-fold C-index, a labelled sensitivity that
                shows what tuning purely for discrimination does to calibration

Two protocols as in 10_alpha_sweep.py: `inner_oof` uses training rows only, and
`temporal_test` is the confirmatory read-out on the 2007-2010 rows. No configuration is
chosen here; the choice was already made in 11_tune.py from inner folds alone.

Output: results/deep/seed_stability.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deep_lib as D  # noqa: E402

OUTDIR = D.ROOT / "results" / "deep"
OUT = OUTDIR / "seed_stability.csv"
SEL = OUTDIR / "tuning_selected.json"
KEY = ["model", "config", "protocol", "seed", "imp"]

SEEDS = list(range(12))
IMP = 1
BASE_SEED = 553000


def load_configs() -> dict[str, dict[str, dict]]:
    with open(SEL) as fh:
        sel = json.load(fh)
    out = {}
    for model, s in sel.items():
        got = {"default": s["default_cfg"]["cfg"]}
        for name, key in (("tuned_brier", "brier"), ("tuned_cindex", "cindex")):
            if key in s["selected"]:
                got[name] = s["selected"][key]["cfg"]
        out[model] = got
    return out


def run_one(t: dict) -> dict:
    d = D.load_imp(IMP)
    tr = d[d.split == "train"].reset_index(drop=True)
    seed = D.PROJECT_SEED + BASE_SEED + 337 * int(t["seed"])
    row = dict(model=t["model"], config=t["config"], protocol=t["protocol"],
               seed=t["seed"], imp=IMP, seed_used=seed, horizon=D.PRIMARY_H,
               cfg=json.dumps(t["cfg"], sort_keys=True))
    try:
        if t["protocol"] == "inner_oof":
            ev_df = D.oof_train(t["model"], tr, D.inner_folds(tr), t["cfg"], seed)
        else:
            te = d[d.split == "test"].reset_index(drop=True)
            r, _ = D.fit_predict(t["model"], tr, te, t["cfg"], seed)
            ev_df = te[D.META].copy()
            ev_df["risk_10"] = r[:, 1]
        m = D.metrics(ev_df.time.values, ev_df.event.values, ev_df.cycle.values,
                      ev_df.risk_10.values)
        m["brier"] = D.ipcw_brier(ev_df.time.values, ev_df.event.values,
                                  ev_df.risk_10.values)
        row.update(m)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def summarise() -> pd.DataFrame:
    df = pd.read_csv(OUT)
    if "error" in df.columns:
        df = df[df.error.isna()]
    g = df.groupby(["model", "config", "protocol"])
    rows = []
    for k, q in g:
        r = dict(zip(["model", "config", "protocol"], k), n_seeds=len(q))
        for m in ("eo", "ici", "calib_slope", "cindex", "auc", "brier"):
            v = q[m].dropna().to_numpy()
            if len(v):
                r[f"{m}_median"] = float(np.median(v))
                r[f"{m}_p10"] = float(np.quantile(v, 0.10))
                r[f"{m}_p90"] = float(np.quantile(v, 0.90))
                r[f"{m}_min"] = float(v.min())
                r[f"{m}_max"] = float(v.max())
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfgs = load_configs()
    # `imp` is part of KEY, so it has to be on the task as well as on the result row,
    # otherwise the resume filter raises KeyError before a single fit is dispatched.
    tasks = [dict(model=m, config=name, cfg=c, protocol=p, seed=s, imp=IMP)
             for m, byname in cfgs.items() for name, c in byname.items()
             for p in ("inner_oof", "temporal_test") for s in SEEDS]
    print(f"seed stability: {len(tasks)} tasks", flush=True)
    D.run_tasks(run_one, tasks, OUT, KEY)
    s = summarise()
    s.to_csv(OUTDIR / "seed_stability_summary.csv", index=False)
    print(s.to_string(index=False), flush=True)
