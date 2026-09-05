#!/usr/bin/env python
"""P2-C stage B. Honest hyperparameter search for DeepHit and Neural Fine-Gray.

Protocol, and the guarantee that the temporal test set never touches selection:

  * Every fit in this script is trained on a subset of the rows with
    `split == "train"` (NHANES cycles 1999-2000 to 2005-2006, N = 14,551) and scored
    on the held-out fifth of those same rows. `df[df.split == "test"]` is never read;
    the string "test" does not index any dataframe here.
  * The five inner folds come from `deep_lib.inner_folds`, which is a deterministic
    stratified partition of the training rows keyed on SEQN and the project seed. It
    is not derived from `cv_rep0..4`, because those are defined on the full cohort and
    would place temporal-test rows inside a tuning fold.
  * Each candidate is scored on out-of-fold predictions pooled across the five inner
    folds, averaged over three seeds.

Selection criterion: the IPCW Brier score for the cause-1 CIF at 10 y, a proper
scoring rule. Selecting on the C-index would pick a ranking-dominated DeepHit by
construction; selecting on ICI or E/O would build the calibration conclusion into the
selection. Both of those alternative selections are recorded as secondary columns so
the effect of the criterion can be read off, and 12_seeds.py refits the C-index-selected
DeepHit as a labelled sensitivity analysis.

Output: results/deep/tuning_raw.csv (one row per config per seed) and
results/deep/tuning_selected.json.
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
RAW = OUTDIR / "tuning_raw.csv"
SEL = OUTDIR / "tuning_selected.json"
SPACE_JSON = OUTDIR / "tuning_space.json"
KEY = ["model", "cfg_id", "seed"]

TUNE_IMP = 1          # the search is run on one completed dataset; the selected
                      # configuration is then refitted on all 30
SEEDS = [0, 1, 2]
N_DEEPHIT = 80
N_NFG = 60
SEARCH_SEED = 20260903

DEEPHIT_SPACE = dict(
    trunk=[[32], [64], [64, 64], [128, 128], [64, 64, 64], [128, 64]],
    head=[[32], [64], [32, 32]],
    dropout=[0.0, 0.1, 0.2, 0.4],
    lr=[1e-3, 3e-3, 1e-2],
    batch_size=[128, 256, 512],
    alpha=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
    sigma=[0.05, 0.1, 0.25, 0.5],
    num_durations=[20, 40, 80],
)
NFG_SPACE = dict(
    layers=[[32], [32, 32], [64, 64], [100, 100], [50, 50, 50]],
    layers_surv=[[32], [64], [100], [50, 50]],
    dropout=[0.0, 0.1, 0.25],
    lr=[1e-4, 5e-4, 1e-3, 5e-3],
    batch_size=[128, 256, 512],
    weight_decay=[1e-4, 1e-3, 1e-2],
)


def sample_configs(space: dict, n: int, rng: np.random.Generator) -> list[dict]:
    """Random search over the grid, de-duplicated, with the prior work's default first."""
    out, seen = [], set()
    keys = sorted(space)
    while len(out) < n:
        c = {k: space[k][int(rng.integers(len(space[k])))] for k in keys}
        sig = json.dumps(c, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


def build_configs() -> dict[str, list[dict]]:
    rng = np.random.default_rng(SEARCH_SEED)
    cfgs = {}
    for model, space, n in (("deephit", DEEPHIT_SPACE, N_DEEPHIT),
                            ("nfg", NFG_SPACE, N_NFG)):
        base = {k: v for k, v in D.DEFAULTS[model].items() if k in space}
        rest = [c for c in sample_configs(space, n + 5, rng)
                if json.dumps(c, sort_keys=True) != json.dumps(base, sort_keys=True)]
        cfgs[model] = [base] + rest[: n - 1]
    return cfgs


def cfg_id(model: str, i: int) -> str:
    return f"{model}_{i:03d}"


def run_one(t: dict) -> dict:
    d = D.load_imp(TUNE_IMP)
    tr = d[d.split == "train"].reset_index(drop=True)   # training cycles only
    folds = D.inner_folds(tr)
    seed = D.PROJECT_SEED + 991 * (int(t["seed"]) + 1) + 17 * t["idx"]
    row = dict(model=t["model"], cfg_id=t["cfg_id"], seed=t["seed"], seed_used=seed,
               imp=TUNE_IMP, cfg=json.dumps(t["cfg"], sort_keys=True))
    try:
        oof = D.oof_train(t["model"], tr, folds, t["cfg"], seed)
        m = D.metrics(oof.time.values, oof.event.values, oof.cycle.values,
                      oof.risk_10.values)
        m["brier"] = D.ipcw_brier(oof.time.values, oof.event.values, oof.risk_10.values)
        row.update(m)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def select(cfgs: dict[str, list[dict]]) -> dict:
    df = pd.read_csv(RAW)
    df = df[df.get("error").isna()] if "error" in df.columns else df
    out = {}
    for model in ("deephit", "nfg"):
        q = df[df.model == model]
        if not len(q):
            continue
        agg = q.groupby("cfg_id").agg(
            brier=("brier", "mean"), ici=("ici", "mean"), eo=("eo", "mean"),
            slope=("calib_slope", "mean"), cindex=("cindex", "mean"),
            auc=("auc", "mean"), n_seeds=("seed", "size")).reset_index()
        agg = agg[agg.n_seeds >= len(SEEDS)]
        if not len(agg):
            continue
        cmap = {cfg_id(model, i): c for i, c in enumerate(cfgs[model])}
        idx = agg.set_index("cfg_id")

        def scores(cid):
            """Row of aggregate scores as plain Python, so json.dump accepts it."""
            if cid not in idx.index:
                return None
            return {k: (float(v) if isinstance(v, (np.floating, np.integer, float, int))
                        else v) for k, v in idx.loc[cid].to_dict().items()}

        picks = {}
        for name, col, better in (("brier", "brier", "min"), ("cindex", "cindex", "max"),
                                  ("ici", "ici", "min")):
            v = agg[col].dropna()
            if not len(v):
                continue
            picks[name] = agg.loc[v.idxmin() if better == "min" else v.idxmax(), "cfg_id"]
        if "brier" not in picks:
            raise RuntimeError(f"{model}: no configuration produced a usable Brier score")
        out[model] = {
            "primary_criterion": "ipcw_brier_10y_inner_oof",
            "n_configs_scored": int(len(agg)),
            "selected": {k: {"cfg_id": v, "cfg": cmap[v], "inner_oof": scores(v)}
                         for k, v in picks.items()},
            "default_cfg": {"cfg_id": cfg_id(model, 0), "cfg": cmap[cfg_id(model, 0)],
                            "inner_oof": scores(cfg_id(model, 0))},
        }
    return out


if __name__ == "__main__":
    cfgs = build_configs()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(SPACE_JSON, "w") as fh:
        json.dump({"space": {"deephit": DEEPHIT_SPACE, "nfg": NFG_SPACE},
                   "search_seed": SEARCH_SEED, "tune_imp": TUNE_IMP, "seeds": SEEDS,
                   "configs": {m: {cfg_id(m, i): c for i, c in enumerate(v)}
                               for m, v in cfgs.items()}}, fh, indent=1)
    tasks = [dict(model=m, cfg_id=cfg_id(m, i), idx=i, cfg=c, seed=s)
             for m, v in cfgs.items() for i, c in enumerate(v) for s in SEEDS]
    print(f"tuning: {len(tasks)} tasks", flush=True)
    D.run_tasks(run_one, tasks, RAW, KEY)
    sel = select(cfgs)
    with open(SEL, "w") as fh:
        json.dump(sel, fh, indent=1)
    print(json.dumps({m: {k: v["cfg_id"] for k, v in s["selected"].items()}
                      for m, s in sel.items()}, indent=1), flush=True)
