#!/usr/bin/env python
"""P2-B step 5: fine empirical validation of the closed form against the
NON-PARAMETRIC estimand gap.

Within each stratum the Aalen-Johansen cause-1 and cause-2 CIFs and the naive
1-KM for cause 1 are all marginal quantities of the same stratum, so the only
assumption under test is that the stratum's two marginal cause-specific hazards
are proportional over time (equivalently kappa = 1/2).  Subject heterogeneity
does NOT enter here: the marginal 1-KM of a mixed group is not an average of
individual 1-KMs.  That makes this a clean test of the identity itself, separate
from the decomposition in cf2.

Strata families (cohort = data/processed/model_matrices/full.parquet, the 35,309
complete-case subjects the cross-validated predictions are defined on):
  A  5-year age band x horizon {5,10,15}
  B  sex x 10-year age band x horizon
  C  decile of out-of-fold predicted csc_cox risk at the horizon x horizon

Reported per stratum: n, cause-1 and cause-2 events by the horizon, fraction
still at risk at the horizon, observed F1 / F2 / naive / gap, theoretical gap,
relative error, non-parametric kappa, and a percentile bootstrap CI for the
relative error.  Strata with fewer than 15 cause-1 events are flagged.

Seed 20260903.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cf_lib import SEED, gap_exact_ph, cif_and_naive, kappa_nelson_aalen  # noqa: E402

OUT = ROOT / "results" / "closed_form"
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = [5, 10, 15]
MIN_EVENTS = 15
N_BOOT = 500


def oof_risk() -> pd.DataFrame:
    """csc_cox out-of-fold risk, averaged over CV repeats per subject."""
    p = pd.read_parquet(ROOT / "results/predictions/csc_cox__cv.parquet")
    return p.groupby("SEQN", as_index=False)[[f"risk_{h}" for h in HORIZONS]].mean()


def build() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data/processed/model_matrices/full.parquet")[
        ["SEQN", "age", "sex", "time", "event"]].copy()
    d = d.merge(oof_risk(), on="SEQN", how="left")
    d["age5"] = (np.floor(d["age"] / 5) * 5).astype(int)
    d["age10"] = (np.floor(d["age"] / 10) * 10).astype(int)
    d["sexlab"] = np.where(d["sex"].to_numpy() == 1, "male", "female")
    return d


def stratum_row(sub: pd.DataFrame, h: int, family: str, label: str,
                rng: np.random.Generator) -> dict:
    t = sub["time"].to_numpy(float)
    e = sub["event"].to_numpy(int)
    o = cif_and_naive(t, e, h)
    Gt = float(gap_exact_ph(o["F1"], o["F2"]))
    rel = (Gt - o["G_obs"]) / o["G_obs"] if o["G_obs"] > 0 else np.nan
    n = len(sub)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ix = rng.integers(0, n, n)
        ob = cif_and_naive(t[ix], e[ix], h)
        draws[b] = ((float(gap_exact_ph(ob["F1"], ob["F2"])) - ob["G_obs"]) / ob["G_obs"]
                    if ob["G_obs"] > 0 else np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return dict(
        family=family, stratum=label, horizon=h, n=n,
        n_ev1=o["n_ev1"], n_ev2=o["n_ev2"], at_risk_frac=o["at_risk_frac"],
        F1_obs=o["F1"], F2_obs=o["F2"], A_naive_obs=o["A_naive"],
        gap_obs=o["G_obs"], gap_theory=Gt, rel_err=rel,
        rel_err_lo=lo, rel_err_hi=hi,
        kappa_nonparam=kappa_nelson_aalen(t, e, h),
        sufficient=bool(o["n_ev1"] >= MIN_EVENTS),
    )


def main() -> None:
    d = build()
    rng = np.random.default_rng(SEED)
    rows = []
    for h in HORIZONS:
        for a, sub in d.groupby("age5"):
            rows.append(stratum_row(sub, h, "age5", f"age {a}-{a+4}", rng))
        for (s, a), sub in d.groupby(["sexlab", "age10"]):
            rows.append(stratum_row(sub, h, "sex_x_age10", f"{s} age {a}-{a+9}", rng))
        r = d[f"risk_{h}"]
        dec = pd.qcut(r, 10, labels=False, duplicates="drop")
        for q, sub in d.groupby(dec):
            rows.append(stratum_row(sub, h, "risk_decile", f"csc_cox decile {int(q)+1}", rng))

    v = pd.DataFrame(rows)
    v.to_csv(OUT / "validation.csv", index=False)

    ok = v[v["sufficient"] & v["gap_obs"].gt(0)]
    print(f"{len(v)} strata built, {len(ok)} with >= {MIN_EVENTS} cause-1 events "
          f"and a positive observed gap")
    q = ok["rel_err"].describe(percentiles=[.05, .25, .5, .75, .95])
    print(q.to_string())
    print("\nrelative error, distribution over sufficient strata")
    print(f"  median            {ok['rel_err'].median():+.4f}")
    print(f"  IQR               {ok['rel_err'].quantile(.25):+.4f} to "
          f"{ok['rel_err'].quantile(.75):+.4f}")
    print(f"  5th-95th          {ok['rel_err'].quantile(.05):+.4f} to "
          f"{ok['rel_err'].quantile(.95):+.4f}")
    for thr in (0.05, 0.10, 0.20):
        print(f"  |rel err| <= {thr:.0%}   {(ok['rel_err'].abs() <= thr).mean():.1%}")
    print(f"  CI covers 0       {((ok['rel_err_lo'] <= 0) & (ok['rel_err_hi'] >= 0)).mean():.1%}")
    print(f"  kappa median      {ok['kappa_nonparam'].median():.4f} "
          f"(IQR {ok['kappa_nonparam'].quantile(.25):.4f}-"
          f"{ok['kappa_nonparam'].quantile(.75):.4f})")
    print("\nby family and horizon (median relative error)")
    print(ok.pivot_table(index="family", columns="horizon", values="rel_err",
                         aggfunc="median").to_string())
    print("\nR^2 of theory vs observed gap over sufficient strata (for contrast only)")
    ss = 1 - ((ok["gap_theory"] - ok["gap_obs"]) ** 2).sum() / \
             ((ok["gap_obs"] - ok["gap_obs"].mean()) ** 2).sum()
    print(f"  R^2 = {ss:.5f}   <- do not quote this without the error distribution")


if __name__ == "__main__":
    main()
