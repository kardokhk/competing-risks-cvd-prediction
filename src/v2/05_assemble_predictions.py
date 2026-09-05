#!/usr/bin/env python
"""P2-A step 05. Concatenate the per-fold parts into the delivered prediction
files defined in src/v2/PREDICTION_FORMAT.md.

  parts/<model>__temporal__imp<k>.parquet   -> predictions/<model>__temporal__imp<k>.parquet
  parts/<model>__cvr<r>f<f>__imp<k>.parquet -> predictions/<model>__cv__imp<k>.parquet

Idempotent; safe to rerun. Parts are kept so an interrupted fitting run resumes.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "results" / "metrics_v2" / "predictions"
PARTS = PRED / "parts"

COLS = ["SEQN", "cycle", "time", "event", "risk_5", "risk_10", "risk_15",
        "model_name", "family", "learner_class", "imp"]
CV_COLS = COLS + ["cv_rep", "cv_fold"]
INT_COLS = ["SEQN", "event", "imp", "cv_rep", "cv_fold"]
FLOAT_COLS = ["time", "risk_5", "risk_10", "risk_15"]


def coerce(df):
    """R writes 32-bit integers and Python writes 64-bit. The delivered files are
    64-bit, so the schema is one thing regardless of which language fitted the model."""
    for c in INT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("int64")
    for c in FLOAT_COLS:
        df[c] = df[c].astype("float64")
    for c in ["cycle", "model_name", "family", "learner_class"]:
        df[c] = df[c].astype(str)
    return df


def main() -> int:
    temporal = list(PARTS.glob("*__temporal__imp*.parquet"))
    n = 0
    for p in temporal:
        out = PRED / p.name
        df = coerce(pd.read_parquet(p))[COLS]
        df.to_parquet(out, index=False)
        n += 1

    groups: dict[tuple[str, int], list[Path]] = defaultdict(list)
    pat = re.compile(r"^(?P<m>.+)__cvr(?P<r>\d+)f(?P<f>\d+)__imp(?P<k>\d+)\.parquet$")
    for p in PARTS.glob("*__cvr*__imp*.parquet"):
        mt = pat.match(p.name)
        if mt:
            groups[(mt["m"], int(mt["k"]))].append(p)

    for (m, k), ps in sorted(groups.items()):
        df = pd.concat([pd.read_parquet(p) for p in sorted(ps)], ignore_index=True)
        df = df.sort_values(["cv_rep", "cv_fold", "SEQN"]).reset_index(drop=True)
        coerce(df)[CV_COLS].to_parquet(PRED / f"{m}__cv__imp{k}.parquet", index=False)
        n += 1
    print(f"assembled {n} prediction files into {PRED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
