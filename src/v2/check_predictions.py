#!/usr/bin/env python
"""P2-A. Validator for the version 2 prediction schema.

Usage:
    python src/v2/check_predictions.py results/metrics_v2/predictions

Checks, per file: name, columns and their order, dtypes, risk range, closed
vocabularies for model_name / family / learner_class, that `imp` matches the
filename and is constant, row counts, uniqueness of SEQN (per repeat for cv),
and that SEQN, time and event agree row for row with the imputed dataset. Any
file that fails is not in the analysis. Exit status is non-zero if any file fails.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IMPDIR = ROOT / "data" / "processed" / "imputed"

COLS = ["SEQN", "cycle", "time", "event", "risk_5", "risk_10", "risk_15",
        "model_name", "family", "learner_class", "imp"]
CV_COLS = COLS + ["cv_rep", "cv_fold"]
VOCAB = {
    "logistic": ("naive", "IPCWLogistic"),
    "logistic_cr": ("competing", "IPCWLogisticCR"),
    "cox_naive": ("naive", "CoxPH"),
    "rsf_naive": ("naive", "RandomSurvivalForest"),
    "gbs_naive": ("naive", "GradientBoostingSurvival"),
    "csc_cox": ("competing", "CauseSpecificCox"),
    "fine_gray": ("competing", "FineGray"),
    "rsf_cr": ("competing", "RandomSurvivalForestCR"),
    "deephit": ("competing", "DeepHit"),
    "nfg": ("competing", "NeuralFineGray"),
}
N_TEMPORAL = 9329
N_FULL = 41151
REPEATS = {0, 1, 2}
PAT = re.compile(r"^(?P<m>[a-z_]+)__(?P<split>temporal|cv)__imp(?P<k>\d+)\.parquet$")


def check(path: Path, imp_cache: dict) -> list[str]:
    errs = []
    mt = PAT.match(path.name)
    if not mt:
        return [f"{path.name}: filename does not match <model>__<split>__imp<k>.parquet"]
    model, split, k = mt["m"], mt["split"], int(mt["k"])
    if model not in VOCAB:
        errs.append(f"{path.name}: model_name {model!r} not in the closed vocabulary")
    if not 1 <= k <= 30:
        errs.append(f"{path.name}: imp index {k} outside 1..30")

    df = pd.read_parquet(path)
    want = CV_COLS if split == "cv" else COLS
    if list(df.columns) != want:
        errs.append(f"{path.name}: columns are {list(df.columns)}, expected {want}")
        return errs
    for c, t in [("SEQN", "int64"), ("event", "int64"), ("imp", "int64")]:
        if str(df[c].dtype) != t:
            errs.append(f"{path.name}: {c} dtype {df[c].dtype}, expected {t}")
    for c in ["time", "risk_5", "risk_10", "risk_15"]:
        if str(df[c].dtype) != "float64":
            errs.append(f"{path.name}: {c} dtype {df[c].dtype}, expected float64")
    if df.risk_10.isna().all():
        errs.append(f"{path.name}: risk_10 is entirely missing; it is required")
    for c in ["risk_5", "risk_10", "risk_15"]:
        v = df[c].dropna()
        if len(v) and (v.min() < -1e-9 or v.max() > 1 + 1e-9):
            errs.append(f"{path.name}: {c} outside [0,1]: [{v.min():.4g}, {v.max():.4g}]")
    if model in VOCAB:
        fam, learner = VOCAB[model]
        if set(df.model_name.unique()) != {model}:
            errs.append(f"{path.name}: model_name not constant / wrong")
        if set(df.family.unique()) != {fam}:
            errs.append(f"{path.name}: family is {set(df.family.unique())}, expected {fam}")
        if set(df.learner_class.unique()) != {learner}:
            errs.append(f"{path.name}: learner_class is "
                        f"{set(df.learner_class.unique())}, expected {learner}")
    if set(df.imp.unique()) != {k}:
        errs.append(f"{path.name}: imp column {set(df.imp.unique())} does not match filename")

    if k not in imp_cache:
        f = IMPDIR / f"imp_{k}.parquet"
        if not f.exists():
            errs.append(f"{path.name}: {f} missing, cannot cross-check")
            return errs
        imp_cache[k] = pd.read_parquet(f)[["SEQN", "time", "event", "split",
                                           "cv_rep0", "cv_rep1", "cv_rep2"]]
    ref = imp_cache[k]

    if split == "temporal":
        if len(df) != N_TEMPORAL:
            errs.append(f"{path.name}: {len(df)} rows, expected {N_TEMPORAL}")
        if df.SEQN.duplicated().any():
            errs.append(f"{path.name}: duplicate SEQN")
        want_seqn = set(ref.loc[ref.split == "test", "SEQN"])
        if set(df.SEQN) != want_seqn:
            errs.append(f"{path.name}: SEQN set differs from the temporal test rows")
    else:
        if set(df.cv_rep.unique()) - REPEATS:
            errs.append(f"{path.name}: cv_rep values {set(df.cv_rep.unique())} outside 0..2")
        for r, g in df.groupby("cv_rep"):
            if g.SEQN.duplicated().any():
                errs.append(f"{path.name}: duplicate SEQN within repeat {r}")
            if len(g) != N_FULL:
                errs.append(f"{path.name}: repeat {r} has {len(g)} rows, expected {N_FULL}")
            fold_ref = ref.set_index("SEQN")[f"cv_rep{r}"]
            got = g.set_index("SEQN").cv_fold
            common = got.index.intersection(fold_ref.index)
            if not (got.loc[common] == fold_ref.loc[common]).all():
                errs.append(f"{path.name}: cv_fold in repeat {r} disagrees with the "
                            f"fold assignment in imp_{k}.parquet")

    m = df.merge(ref[["SEQN", "time", "event"]], on="SEQN", suffixes=("", "_ref"))
    if len(m):
        if not (m.time - m.time_ref).abs().lt(1e-9).all():
            errs.append(f"{path.name}: time does not match imp_{k}.parquet")
        if not (m.event == m.event_ref).all():
            errs.append(f"{path.name}: event does not match imp_{k}.parquet")
    return errs


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1
             else ROOT / "results/metrics_v2/predictions")
    files = sorted(p for p in d.glob("*.parquet"))
    if not files:
        print(f"no prediction files in {d}")
        return 1
    cache: dict = {}
    all_errs = []
    for p in files:
        e = check(p, cache)
        all_errs += e
    print(f"checked {len(files)} files in {d}")
    if all_errs:
        for e in all_errs:
            print("  FAIL:", e)
        print(f"{len(all_errs)} problems")
        return 1
    print("all files conform to src/v2/PREDICTION_FORMAT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
