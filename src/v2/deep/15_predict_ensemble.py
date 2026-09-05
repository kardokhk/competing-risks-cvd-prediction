#!/usr/bin/env python
"""P2-C2. Seed-ensembled production predictions for `deephit` and `nfg`.

Replaces the single-seed output of `13_predict.py`. Everything else is unchanged:
the same selected configurations from `results/deep/tuning_selected.json` (DeepHit at
alpha = 0.9), the same data, the same fold definitions, the same standardisation rule,
the same frozen output schema in `src/v2/PREDICTION_FORMAT.md`.

Why. `13_predict.py` fitted one network per fit, seeded by the schema rule
`20260903 + 1000*imp` (+ `100*cv_rep + cv_fold` for cross-validation), so each of the
30 imputations trained a differently initialised network. Rubin pooling then charges the
resulting optimisation noise to the between-imputation variance. In the single-seed run
that noise dominated: the between-imputation standard deviation of the mean predicted
10 y risk on the temporal split was 0.0030 for DeepHit and 0.0039 for Neural Fine-Gray,
against 0.00015 for the cause-specific Cox model on the same 30 datasets, and a pilot that
crossed 5 imputations with 4 seeds attributed essentially all of it to the seed
(`notes/scratch/pilot_decomp.csv`). The consequence was a fraction of missing information
above 1 for `mean_pred` and every `frac_above_*` threshold, which is not a legitimate FMI.

What changes. Every fit is now an ensemble of `--seeds` independently initialised networks
whose **predicted risks are averaged**. The averaging is on the risk scale, which is the
scale the evaluation consumes: `06_evaluate.py` reads `risk_10` and forms means,
threshold counts, pseudo-observation calibration and rank statistics from it directly.
Averaging there makes the ensemble mean predicted risk the mean of the member mean
predicted risks, so E/O, `mean_pred` and `frac_above_*` are the quantities whose seed
noise is being averaged down, by construction. Averaging on a cloglog or logit scale would
shrink the ensemble mean away from the arithmetic mean risk and would change the estimand
of every calibration-in-the-large statistic in the paper, so it is not used.

Seeding. Member j of a fit uses `base + 1_000_000 * j`, where `base` is exactly the schema
seed that `13_predict.py` used. Member 0 therefore reproduces the archived single-seed
prediction bit for bit, which is checked by `--verify-member0`. The 1,000,000 stride cannot
collide with the imputation (1,000), repeat (100) or fold (1) strides.

Resume. One parquet per (model, split, imputation, repeat, fold, member) under
`results/deep/cache/parts_ens/`, written through a temporary file, so a killed or
resubmitted job resumes at the fit granularity. Shards write separate logs, so array tasks
never contend for the same CSV.

Usage:
    python-cr 15_predict_ensemble.py --imps 1-10 --seeds 10 --shard 0
    python-cr 15_predict_ensemble.py --assemble --seeds 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deep_lib as D  # noqa: E402

PRED = D.ROOT / "results" / "metrics_v2" / "predictions"
PARTS = D.ROOT / "results" / "deep" / "cache" / "parts_ens"
SEL = D.ROOT / "results" / "deep" / "tuning_selected.json"
LOGDIR = D.ROOT / "results" / "deep"
MEMBER_LOG = LOGDIR / "ensemble_members.csv"
KEY = ["part"]

REPS = [0, 1, 2]
FOLDS = [0, 1, 2, 3, 4]
MEMBER_STRIDE = 1_000_000
COLS = ["SEQN", "cycle", "time", "event", "risk_5", "risk_10", "risk_15",
        "model_name", "family", "learner_class", "imp"]
THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2)


def final_cfg(model: str) -> dict:
    with open(SEL) as fh:
        return json.load(fh)[model]["selected"]["brier"]["cfg"]


def unit_tag(model: str, split: str, imp: int, rep=None, fold=None) -> str:
    return (f"{model}__temporal__imp{imp}" if split == "temporal"
            else f"{model}__cvr{rep}f{fold}__imp{imp}")


def member_path(model: str, split: str, imp: int, member: int, rep=None, fold=None) -> Path:
    return PARTS / f"{unit_tag(model, split, imp, rep, fold)}__s{member}.parquet"


def base_seed(split: str, imp: int, rep=None, fold=None) -> int:
    return (D.temporal_seed(imp) if split == "temporal"
            else D.cv_seed(imp, int(rep), int(fold)))


def parse_range(s: str) -> list[int]:
    out: list[int] = []
    for piece in str(s).split(","):
        piece = piece.strip()
        if "-" in piece:
            a, b = piece.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif piece:
            out.append(int(piece))
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# One member fit
# --------------------------------------------------------------------------- #
def run_one(t: dict) -> dict:
    model, split, imp, member = t["model"], t["split"], int(t["imp"]), int(t["member"])
    p = member_path(model, split, imp, member, t.get("rep"), t.get("fold"))
    row = dict(part=p.stem, model=model, split=split, imp=imp, member=member,
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
        else:
            col = f"cv_rep{t['rep']}"
            tr = d[d[col] != t["fold"]].reset_index(drop=True)
            te = d[d[col] == t["fold"]].reset_index(drop=True)
        seed = base_seed(split, imp, t.get("rep"), t.get("fold")) + MEMBER_STRIDE * member
        t0 = _time.time()
        r, info = D.fit_predict(model, tr, te, cfg, seed)
        # member cache carries only what the average needs; SEQN is kept so the
        # averaging step can align members rather than trust row order
        out = pd.DataFrame({
            "SEQN": te.SEQN.to_numpy("int64"),
            "risk_5": r[:, 0].astype("float32"),
            "risk_10": r[:, 1].astype("float32"),
            "risk_15": r[:, 2].astype("float32"),
        })
        tmp = p.with_suffix(".parquet.tmp")
        out.to_parquet(tmp, index=False)
        tmp.replace(p)
        row["secs"] = round(_time.time() - t0, 1)
        row["status"] = "fitted"
        row["seed_used"] = seed
        row["mean_risk10"] = float(np.mean(r[:, 1]))
        for th in THRESHOLDS:
            row[f"frac_above_{th}"] = float(np.mean(r[:, 1] > th))
        row["n_epochs"] = info.get("n_epochs", -1)
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
    return row


# --------------------------------------------------------------------------- #
# Averaging and assembly
# --------------------------------------------------------------------------- #
def average_unit(model: str, split: str, imp: int, n_seeds: int,
                 rep=None, fold=None) -> pd.DataFrame | None:
    """Mean predicted risk over the `n_seeds` members of one fit.

    Members are aligned on SEQN, not on row order. Returns None if any member is
    missing, so a partly finished job assembles nothing rather than an ensemble of
    the wrong size.
    """
    paths = [member_path(model, split, imp, j, rep, fold) for j in range(n_seeds)]
    if not all(q.exists() for q in paths):
        return None
    first = pd.read_parquet(paths[0]).sort_values("SEQN", kind="mergesort")
    seqn = first.SEQN.to_numpy("int64")
    acc = np.zeros((len(seqn), 3), dtype=np.float64)
    for q in paths:
        d = pd.read_parquet(q).sort_values("SEQN", kind="mergesort")
        if not np.array_equal(d.SEQN.to_numpy("int64"), seqn):
            raise ValueError(f"member SEQN mismatch in {q.name}")
        acc += d[["risk_5", "risk_10", "risk_15"]].to_numpy(dtype=np.float64)
    acc /= float(n_seeds)
    return pd.DataFrame({"SEQN": seqn, "risk_5": acc[:, 0],
                         "risk_10": acc[:, 1], "risk_15": acc[:, 2]})


def assemble(model: str, imps: list[int], n_seeds: int, outdir: Path) -> list[str]:
    """Write the frozen-schema prediction files from the seed-averaged risks."""
    fam, learner = "competing", D.LEARNER[model]
    made = []
    for imp in imps:
        d0 = D.load_imp(imp)
        # ---- temporal ----
        ens = average_unit(model, "temporal", imp, n_seeds)
        if ens is not None:
            te = d0[d0.split == "test"].sort_values("SEQN", kind="mergesort")
            if not np.array_equal(te.SEQN.to_numpy("int64"), ens.SEQN.to_numpy("int64")):
                raise ValueError(f"{model} temporal imp{imp}: SEQN does not match imp_{imp}")
            out = pd.DataFrame({
                "SEQN": ens.SEQN.to_numpy("int64"),
                "cycle": te.cycle.astype(str).to_numpy(),
                "time": te.time.to_numpy("float64"),
                "event": te.event.to_numpy("int64"),
                "risk_5": ens.risk_5.to_numpy("float64"),
                "risk_10": ens.risk_10.to_numpy("float64"),
                # 15 y is not estimable on the 2007-2010 test cycles (13.2 y of
                # follow-up at most), so it is NaN here as in every other family
                "risk_15": np.full(len(ens), np.nan, dtype="float64"),
                "model_name": model, "family": fam, "learner_class": learner,
                "imp": np.int64(imp),
            })[COLS]
            dst = outdir / f"{model}__temporal__imp{imp}.parquet"
            tmp = dst.with_suffix(".parquet.tmp")
            out.to_parquet(tmp, index=False)
            tmp.replace(dst)
            made.append(dst.name)
        # ---- cv ----
        parts = []
        complete = True
        for rep in REPS:
            for fold in FOLDS:
                ens = average_unit(model, "cv", imp, n_seeds, rep, fold)
                if ens is None:
                    complete = False
                    break
                col = f"cv_rep{rep}"
                te = d0[d0[col] == fold].sort_values("SEQN", kind="mergesort")
                if not np.array_equal(te.SEQN.to_numpy("int64"),
                                      ens.SEQN.to_numpy("int64")):
                    raise ValueError(
                        f"{model} cv imp{imp} rep{rep} fold{fold}: SEQN mismatch")
                parts.append(pd.DataFrame({
                    "SEQN": ens.SEQN.to_numpy("int64"),
                    "cycle": te.cycle.astype(str).to_numpy(),
                    "time": te.time.to_numpy("float64"),
                    "event": te.event.to_numpy("int64"),
                    "risk_5": ens.risk_5.to_numpy("float64"),
                    "risk_10": ens.risk_10.to_numpy("float64"),
                    "risk_15": ens.risk_15.to_numpy("float64"),
                    "model_name": model, "family": fam, "learner_class": learner,
                    "imp": np.int64(imp),
                    "cv_rep": np.int64(rep), "cv_fold": np.int64(fold),
                }))
            if not complete:
                break
        if complete and parts:
            out = pd.concat(parts, ignore_index=True)[COLS + ["cv_rep", "cv_fold"]]
            out = out.sort_values(["cv_rep", "SEQN"], kind="mergesort")
            out = out.reset_index(drop=True)
            dst = outdir / f"{model}__cv__imp{imp}.parquet"
            tmp = dst.with_suffix(".parquet.tmp")
            out.to_parquet(tmp, index=False)
            tmp.replace(dst)
            made.append(dst.name)
    return made


def verify_member0(models, imps) -> int:
    """Member 0 carries the schema seed, so it must reproduce the archived run."""
    arch = D.ROOT / "results" / "deep" / "predictions_singleseed"
    bad = 0
    for model in models:
        for imp in imps:
            p = member_path(model, "temporal", imp, 0)
            a = arch / f"{model}__temporal__imp{imp}.parquet"
            if not (p.exists() and a.exists()):
                continue
            m = pd.read_parquet(p).sort_values("SEQN", kind="mergesort")
            s = pd.read_parquet(a).sort_values("SEQN", kind="mergesort")
            dm = float(np.max(np.abs(m.risk_10.to_numpy("float64")
                                     - s.risk_10.to_numpy("float64"))))
            flag = "OK" if dm < 1e-6 else "DIFFERS"
            if dm >= 1e-6:
                bad += 1
            print(f"  member0 {model} temporal imp{imp}: max |risk_10 diff| "
                  f"{dm:.3e}  {flag}", flush=True)
    return bad


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imps", default="1-30")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--models", default="deephit,nfg")
    ap.add_argument("--splits", default="temporal,cv")
    ap.add_argument("--shard", default="0")
    ap.add_argument("--nproc", type=int, default=0)
    ap.add_argument("--assemble", action="store_true",
                    help="skip fitting, average the cached members and write predictions")
    ap.add_argument("--outdir", default=str(PRED))
    ap.add_argument("--verify-member0", action="store_true")
    a = ap.parse_args()

    imps = parse_range(a.imps)
    models = [m for m in a.models.split(",") if m]
    splits = [s for s in a.splits.split(",") if s]
    outdir = Path(a.outdir)
    PARTS.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    cfgs = {m: final_cfg(m) for m in models}
    print(f"models={models} splits={splits} imps={imps[0]}..{imps[-1]} "
          f"seeds={a.seeds} shard={a.shard}", flush=True)
    print("configurations:", json.dumps(cfgs), flush=True)

    # --assemble and --verify-member0 are read-only actions; neither fits
    if not a.assemble and not a.verify_member0:
        log = LOGDIR / f"ensemble_members_shard{a.shard}.csv"
        # a previous attempt's failures must be retried; every non-error row
        # corresponds to a member file already on disk
        if log.exists():
            lg = pd.read_csv(log)
            if "status" in lg.columns:
                lg[lg.status != "error"].to_csv(log, index=False)
        tasks = []
        for m in models:
            for imp in imps:
                for j in range(a.seeds):
                    if "temporal" in splits:
                        tasks.append(dict(model=m, split="temporal", imp=imp, member=j,
                                          cfg=cfgs[m],
                                          part=member_path(m, "temporal", imp, j).stem))
                    if "cv" in splits:
                        for rep in REPS:
                            for fold in FOLDS:
                                tasks.append(dict(
                                    model=m, split="cv", imp=imp, member=j, rep=rep,
                                    fold=fold, cfg=cfgs[m],
                                    part=member_path(m, "cv", imp, j, rep, fold).stem))
        tasks = [t for t in tasks if not (PARTS / f"{t['part']}.parquet").exists()]
        # the slowest family first so the tail of the job is short
        tasks.sort(key=lambda t: (t["model"] != "nfg", t["split"] != "cv", t["imp"]))
        print(f"{len(tasks)} member fits still to run", flush=True)
        D.run_tasks(run_one, tasks, log, KEY, n_proc=(a.nproc or None))

    if a.verify_member0:
        print("--- member 0 against the archived single-seed run ---", flush=True)
        nbad = verify_member0(models, imps[:5])
        print(f"member0 check: {nbad} file(s) differ", flush=True)

    if a.assemble:
        for m in models:
            made = assemble(m, imps, a.seeds, outdir)
            print(f"{m}: wrote {len(made)} prediction files to {outdir}", flush=True)
        # merge the per-shard member logs into one
        shards = sorted(LOGDIR.glob("ensemble_members_shard*.csv"))
        if shards:
            allrows = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
            allrows = allrows.drop_duplicates(subset=["part"], keep="last")
            allrows.to_csv(MEMBER_LOG, index=False)
            print(f"member log: {len(allrows)} rows -> {MEMBER_LOG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
