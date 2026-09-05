#!/usr/bin/env python
"""P2-C stage D. Production predictions for `deephit` and `nfg` in the frozen schema.

Writes results/metrics_v2/predictions/<model>__<split>__imp<k>.parquet for
model in {deephit, nfg}, split in {temporal, cv} and k = 1..30, exactly as
src/v2/PREDICTION_FORMAT.md section 3 requires.

Configuration: the one selected in 11_tune.py on the inner out-of-fold IPCW Brier score,
that is, on training rows only.

Seeding, and what was traded away. The schema prescribes one seed per fit,
seed = 20260903 + 1000*imp for temporal and + 100*cv_rep + cv_fold for cross-validation,
and that is what is used here: one network per fit, no averaging over seeds. Averaging
three seeds would have made these two families deep ensembles while the other eight
families remain single fits, which would confound the architecture contrast the paper
is making. The cost of the choice is that single-initialisation variability is not
averaged out of any one file; it is instead quantified separately in
results/deep/seed_stability.csv and, across the 30 imputations and 15 folds, enters the
pooled between-imputation variance, which is therefore mildly conservative.

Work is done at the level of one fit, cached under results/deep/cache/parts/, so a
resubmitted job resumes rather than restarts.

risk_15 is written as NaN on the temporal split, matching every other family, because
the 2007-2010 test cycles have at most 13.2 y of follow-up.
"""
from __future__ import annotations

import json
import time as _time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deep_lib as D  # noqa: E402

PRED = D.ROOT / "results" / "metrics_v2" / "predictions"
PARTS = D.ROOT / "results" / "deep" / "cache" / "parts"
SEL = D.ROOT / "results" / "deep" / "tuning_selected.json"
LOG = D.ROOT / "results" / "deep" / "predict_log.csv"
KEY = ["part"]

IMPS = list(range(1, 31))
REPS = [0, 1, 2]
FOLDS = [0, 1, 2, 3, 4]
COLS = ["SEQN", "cycle", "time", "event", "risk_5", "risk_10", "risk_15",
        "model_name", "family", "learner_class", "imp"]


def final_cfg(model: str) -> dict:
    with open(SEL) as fh:
        return json.load(fh)[model]["selected"]["brier"]["cfg"]


def part_path(model: str, split: str, imp: int, rep=None, fold=None) -> Path:
    tag = f"{model}__temporal__imp{imp}" if split == "temporal" else \
          f"{model}__cvr{rep}f{fold}__imp{imp}"
    return PARTS / f"{tag}.parquet"


def run_one(t: dict) -> dict:
    model, split, imp = t["model"], t["split"], int(t["imp"])
    p = part_path(model, split, imp, t.get("rep"), t.get("fold"))
    row = dict(part=p.stem, model=model, split=split, imp=imp,
               rep=t.get("rep", -1), fold=t.get("fold", -1))
    if p.exists():
        row["status"] = "cached"
        return row
    try:
        cfg = t["cfg"]
        d = D.load_imp(imp)
        if split == "temporal":
            tr = d[d.split == "train"].reset_index(drop=True)
            te = d[d.split == "test"].reset_index(drop=True)
            seed = D.temporal_seed(imp)
        else:
            col = f"cv_rep{t['rep']}"
            tr = d[d[col] != t["fold"]].reset_index(drop=True)
            te = d[d[col] == t["fold"]].reset_index(drop=True)
            seed = D.cv_seed(imp, int(t["rep"]), int(t["fold"]))
        t0 = _time.time()
        r, info = D.fit_predict(model, tr, te, cfg, seed)
        # the imputed files store SEQN, event and imp as int32; the schema requires
        # int64, so every column is built with its declared dtype rather than copied
        out = pd.DataFrame({
            "SEQN": te.SEQN.to_numpy("int64"),
            "cycle": te.cycle.astype(str).to_numpy(),
            "time": te.time.to_numpy("float64"),
            "event": te.event.to_numpy("int64"),
            "risk_5": r[:, 0].astype("float64"),
            "risk_10": r[:, 1].astype("float64"),
            "risk_15": (np.full(len(te), np.nan) if split == "temporal"
                        else r[:, 2].astype("float64")),
        })
        if split == "cv":
            out["cv_rep"] = np.int64(t["rep"])
            out["cv_fold"] = np.int64(t["fold"])
        row["secs"] = round(_time.time() - t0, 1)
        tmp = p.with_suffix(".parquet.tmp")
        out.to_parquet(tmp, index=False)
        tmp.replace(p)
        row["status"] = "fitted"
        row["seed_used"] = seed
        row["mean_risk10"] = float(np.nanmean(out.risk_10))
        row["n_epochs"] = info.get("n_epochs", -1)
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def assemble(model: str) -> list[str]:
    fam, learner = "competing", D.LEARNER[model]
    made = []
    for imp in IMPS:
        # temporal
        p = part_path(model, "temporal", imp)
        dst = PRED / f"{model}__temporal__imp{imp}.parquet"
        if p.exists() and not dst.exists():
            d = pd.read_parquet(p)
            d["model_name"] = model; d["family"] = fam
            d["learner_class"] = learner; d["imp"] = np.int64(imp)
            d = d[COLS]
            d.to_parquet(dst, index=False)
            made.append(dst.name)
        # cv
        parts = [part_path(model, "cv", imp, r, f) for r in REPS for f in FOLDS]
        dst = PRED / f"{model}__cv__imp{imp}.parquet"
        if all(q.exists() for q in parts) and not dst.exists():
            d = pd.concat([pd.read_parquet(q) for q in parts], ignore_index=True)
            d["model_name"] = model; d["family"] = fam
            d["learner_class"] = learner; d["imp"] = np.int64(imp)
            d = d[COLS + ["cv_rep", "cv_fold"]]
            d = d.sort_values(["cv_rep", "SEQN"], kind="mergesort").reset_index(drop=True)
            d.to_parquet(dst, index=False)
            made.append(dst.name)
    return made


if __name__ == "__main__":
    PARTS.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    models = [only] if only else ["deephit", "nfg"]
    cfgs = {m: final_cfg(m) for m in models}
    print("final configurations:", json.dumps(cfgs), flush=True)

    # A previous attempt's failures must be retried, so drop them from the log; every
    # other logged row corresponds to a part file that is already on disk.
    if LOG.exists():
        lg = pd.read_csv(LOG)
        if "status" in lg.columns:
            lg[lg.status != "error"].to_csv(LOG, index=False)

    tasks = []
    for m in models:
        for imp in IMPS:
            tasks.append(dict(model=m, split="temporal", imp=imp, cfg=cfgs[m],
                              part=part_path(m, "temporal", imp).stem))
            for rep in REPS:
                for fold in FOLDS:
                    tasks.append(dict(model=m, split="cv", imp=imp, rep=rep,
                                      fold=fold, cfg=cfgs[m],
                                      part=part_path(m, "cv", imp, rep, fold).stem))
    tasks = [t for t in tasks if not (PARTS / f"{t['part']}.parquet").exists()]
    # longest fits first so the tail of the job is short
    tasks.sort(key=lambda t: (t["split"] != "cv", t["imp"]))
    print(f"predictions: {len(tasks)} fits still to run", flush=True)
    D.run_tasks(run_one, tasks, LOG, KEY)
    for m in models:
        made = assemble(m)
        print(f"{m}: wrote {len(made)} prediction files", flush=True)
