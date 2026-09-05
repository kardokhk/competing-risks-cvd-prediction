#!/usr/bin/env python
"""P2-A step 11a. Export the first B replicates of the shared bootstrap index so
the R side of the concordant-smoother calculation resamples exactly the same
subjects as the numpy side.

The index rule is `np.random.default_rng([seed, split_id, b]).integers(0, n, n)`,
a pure function of the replicate number. numpy's SeedSequence and PCG64 cannot be
reproduced in R, so the indices are written out rather than regenerated. Replicate
b here is the same replicate b as in `src/v2/06_evaluate.py`, which is what makes
the concordant-smoother contrasts paired with the rest.

Written as 1-based row positions into the canonical SEQN-sorted order of the
split, which is the order `07`/`11b` read the prediction files in.

Run: python src/v2/11a_export_boot_index.py --split temporal -B 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEED = json.loads((ROOT / "src/v2/feature_spec.json").read_text())["seed"]
SPLIT_ID = {"temporal": 0, "cv": 1}
N = {"temporal": 9329, "cv": 41151}
OUT = ROOT / "results" / "metrics_v2" / "boot"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="temporal")
    ap.add_argument("-B", type=int, default=300)
    a = ap.parse_args()
    n = N[a.split]
    idx = np.empty((a.B, n), dtype=np.int32)
    for b in range(a.B):
        rng = np.random.default_rng([SEED, SPLIT_ID[a.split], b])
        idx[b] = rng.integers(0, n, n) + 1        # 1-based for R
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"boot_index__{a.split}__B{a.B}.parquet"
    pd.DataFrame(idx.T, columns=[f"b{b}" for b in range(a.B)]).to_parquet(f, index=False)
    print(f"wrote {f}  shape {idx.shape}  (replicate 0 first five positions: "
          f"{idx[0][:5].tolist()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
