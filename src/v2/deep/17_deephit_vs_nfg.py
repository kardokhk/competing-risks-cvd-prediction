#!/usr/bin/env python
"""P2-C2. The paired DeepHit minus Neural Fine-Gray contrast on the pooled benchmark.

`src/v2/08_pool_and_report.py` writes only the five contrasts the manuscript's competing-risk
argument needs, and DeepHit against Neural Fine-Gray is not one of them, so P2-C's verdict
rested on `results/deep/seed_stability.csv`, which is twelve seeds on imputation 1. This
script forms the same contrast on the main benchmark instead: thirty imputations, the shared
bootstrap replicates, and the pooling of `08_pool_and_report.pool`, so the interval is paired
in exactly the way every other contrast in the paper is paired.

Nothing is refitted and nothing outside `results/deep/` is written. The contrast can be
computed from either state of the analysis by pointing `--boot` at the archived single-seed
chunks or at the current ones, which is how the before-and-after comparison is made.

Run:
  python src/v2/deep/17_deephit_vs_nfg.py --boot results/metrics_v2/boot --label ensemble
  python src/v2/deep/17_deephit_vs_nfg.py \
      --boot results/metrics_v2/boot_archive_singleseeddeep_<stamp> --label single_seed
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "results" / "metrics_v2" / "boot" / "schema.json"
OUTDIR = ROOT / "results" / "deep"
N_SUBJECTS = {"temporal": 9329, "cv": 41151}
PAT = re.compile(r"^(?P<split>\w+)__imp(?P<k>\d+)__chunk(?P<c>\d+)\.npz$")


def load_pool():
    """Reuse 08_pool_and_report.pool so the interval is formed identically."""
    p = ROOT / "src" / "v2" / "08_pool_and_report.py"
    spec = importlib.util.spec_from_file_location("_p08", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_p08"] = mod
    spec.loader.exec_module(mod)
    return mod.pool


def load_boot(bootdir: Path):
    point, chunks = {}, {}
    for f in sorted(Path(bootdir).glob("*.npz")):
        mt = PAT.match(f.name)
        if not mt:
            continue
        key = (mt["split"], int(mt["k"]))
        z = np.load(f)
        point.setdefault(key, z["point"])
        chunks.setdefault(key, []).append((int(mt["c"]), z["boot"]))
    boot = {k: np.concatenate([b for _, b in sorted(v)], axis=0)
            for k, v in chunks.items()}
    return point, boot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", default=str(ROOT / "results/metrics_v2/boot"))
    ap.add_argument("--label", default="current")
    ap.add_argument("--a", default="deephit")
    ap.add_argument("--b", default="nfg")
    a = ap.parse_args()

    schema = json.loads(SCHEMA.read_text())
    models, horizons, metrics = schema["models"], schema["horizons"], schema["metrics"]
    ia, ib = models.index(a.a), models.index(a.b)
    pool = load_pool()
    point, boot = load_boot(Path(a.boot))
    if not point:
        print(f"no bootstrap chunks under {a.boot}", file=sys.stderr)
        return 1
    splits = sorted({s for s, _ in point})
    rows = []
    for split in splits:
        imps = sorted(k for s, k in point if s == split)
        n_com = N_SUBJECTS[split] - 1
        for hi, h in enumerate(horizons):
            for mi, mt in enumerate(metrics):
                qm, var_m, allb = [], [], []
                for k in imps:
                    p = point[(split, k)]
                    d = float(p[ia, hi, mi] - p[ib, hi, mi])
                    bb = boot[(split, k)]
                    db = bb[:, ia, hi, mi] - bb[:, ib, hi, mi]
                    db = db[np.isfinite(db)]
                    qm.append(d)
                    var_m.append(float(np.var(db, ddof=1)) if db.size > 1 else np.nan)
                    allb.append(db)
                r = pool(np.array(qm), np.array(var_m),
                         np.concatenate(allb) if allb else np.array([]), n_com)
                r.update(state=a.label, split=split, contrast=f"{a.a} - {a.b}",
                         horizon_y=h, metric=mt,
                         ci_excludes_zero=bool(np.isfinite(r["lo"]) and np.isfinite(r["hi"])
                                               and (r["lo"] > 0 or r["hi"] < 0)))
                rows.append(r)
    d = pd.DataFrame(rows)
    cols = ["state", "split", "contrast", "horizon_y", "metric", "est", "lo", "hi",
            "lo_rubin", "hi_rubin", "se", "fmi", "df", "m", "ci_excludes_zero"]
    d = d[cols]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dst = OUTDIR / f"deephit_vs_nfg__{a.label}.csv"
    d.to_csv(dst, index=False)
    print(f"wrote {dst}: {len(d)} rows, splits {splits}, m = {int(d.m.max())}")
    key = d[(d.horizon_y == 10.0) &
            (d.metric.isin(["eo", "ici", "calib_slope", "cindex", "auc", "mean_pred"]))]
    print(key[["split", "metric", "est", "lo", "hi", "ci_excludes_zero"]]
          .to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
