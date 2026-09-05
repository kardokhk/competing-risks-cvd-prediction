#!/usr/bin/env python
"""P2-B step 1-3: decompose the naive-versus-competing prediction gap on the
temporal test set into

    change of estimand (pooled)  +  heterogeneity (Jensen)  +  hazard timing
        +  implementation residual

Definitions at horizon h, over the n test subjects i:

    N_i        cox_naive risk_h                 (benchmark naive model, sksurv)
    C_i        csc_cox risk_h                   (benchmark competing model, R)
    F1_i, F2_i cause-1 / cause-2 CIF from the refit CSC (cf1_cause2_fit.R)
    V_i        1 - exp(-Lambda_1i(h)) from the SAME refit CSC: the naive
               estimand with the model held fixed

    D_obs   = mean(N) - mean(C)                    what the benchmark shows
    D_model = mean(V) - mean(F1)                   exact estimand gap, one fit
    T_het   = mean_i G(F1_i, F2_i)                 closed form, per subject
    T_pool  = G(mean F1, mean F2)                  closed form, pooled

    change of estimand   = T_pool
    heterogeneity        = T_het   - T_pool
    hazard timing        = D_model - T_het
    implementation       = D_obs   - D_model

with G(F1,F2) = 1 - (1-F1-F2)^(F1/(F1+F2)) - F1 (cf_lib.gap_exact_ph).

Seed 20260903.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cf_lib import SEED, gap_exact_ph, naive_exact_ph, cif_and_naive, kappa_nelson_aalen  # noqa: E402

OUT = ROOT / "results" / "closed_form"
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = [5, 10, 15]
N_BOOT = 2000


def load() -> pd.DataFrame:
    cf = pd.read_parquet(OUT / "cause2_predictions.parquet")
    nv = pd.read_parquet(ROOT / "results/predictions/cox_naive__temporal.parquet")
    cs = pd.read_parquet(ROOT / "results/predictions/csc_cox__temporal.parquet")
    d = cf.merge(nv[["SEQN"] + [f"risk_{h}" for h in HORIZONS]].rename(
                     columns={f"risk_{h}": f"N_{h}" for h in HORIZONS}), on="SEQN")
    d = d.merge(cs[["SEQN"] + [f"risk_{h}" for h in HORIZONS]].rename(
                    columns={f"risk_{h}": f"C_{h}" for h in HORIZONS}), on="SEQN")
    assert len(d) == len(cf), "merge lost rows"
    return d


def terms(d: pd.DataFrame, h: int, idx: np.ndarray) -> dict:
    F1 = d[f"risk1_{h}"].to_numpy()[idx]
    F2 = d[f"risk2_{h}"].to_numpy()[idx]
    V = d[f"naive1_{h}"].to_numpy()[idx]
    N = d[f"N_{h}"].to_numpy()[idx]
    C = d[f"C_{h}"].to_numpy()[idx]
    D_obs = N.mean() - C.mean()
    D_model = V.mean() - F1.mean()
    T_het = float(np.mean(gap_exact_ph(F1, F2)))
    T_pool = float(gap_exact_ph(F1.mean(), F2.mean()))
    return dict(
        mean_N=N.mean(), mean_C=C.mean(), mean_F1=F1.mean(), mean_F2=F2.mean(),
        mean_V=V.mean(),
        D_obs=D_obs, D_model=D_model, T_het=T_het, T_pool=T_pool,
        estimand=T_pool, heterogeneity=T_het - T_pool,
        timing=D_model - T_het, implementation=D_obs - D_model,
        # implementation split
        impl_naive=N.mean() - V.mean(), impl_csc=F1.mean() - C.mean(),
        # shares of D_obs
        sh_estimand=T_pool / D_obs,
        sh_heterogeneity=(T_het - T_pool) / D_obs,
        sh_timing=(D_model - T_het) / D_obs,
        sh_implementation=(D_obs - D_model) / D_obs,
        # diagnostics
        corr_F1F2=float(np.corrcoef(F1, F2)[0, 1]),
        EF1F2=float(np.mean(F1 * F2)), EF1_EF2=float(F1.mean() * F2.mean()),
        kappa_implied=float((V.mean() - F1.mean()) / np.mean(F1 * F2)),
    )


def main() -> None:
    d = load()
    n = len(d)
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    full = np.arange(n)

    rows = []
    for h in HORIZONS:
        pt = terms(d, h, full)
        draws = pd.DataFrame([terms(d, h, b) for b in boot_idx])
        for k, v in pt.items():
            lo, hi = np.nanpercentile(draws[k].to_numpy(), [2.5, 97.5])
            rows.append(dict(horizon=h, quantity=k, value=v, lo=lo, hi=hi))
        print(f"h={h}: D_obs={pt['D_obs']:.5f} = estimand {pt['estimand']:.5f}"
              f" + heterogeneity {pt['heterogeneity']:.5f}"
              f" + timing {pt['timing']:.5f}"
              f" + implementation {pt['implementation']:.5f}")
        print(f"      shares  {pt['sh_estimand']:.3f} / {pt['sh_heterogeneity']:.3f}"
              f" / {pt['sh_timing']:.3f} / {pt['sh_implementation']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "decomposition.csv", index=False)

    # ---- observed calibration context: E/O against the Aalen-Johansen CIF ----
    an = pd.read_parquet(ROOT / "data/processed/analytic.parquet")[["SEQN", "age"]]
    d2 = d.merge(an, on="SEQN")
    eo_rows = []
    for h in HORIZONS:
        obs = cif_and_naive(d2["time"].to_numpy(), d2["event"].to_numpy(), h)
        kap = kappa_nelson_aalen(d2["time"].to_numpy(), d2["event"].to_numpy(), h)
        eo_rows.append(dict(
            horizon=h, n=len(d2), n_ev1=obs["n_ev1"], n_ev2=obs["n_ev2"],
            at_risk_frac=obs["at_risk_frac"],
            F1_obs=obs["F1"], F2_obs=obs["F2"], A_naive_obs=obs["A_naive"],
            G_obs_nonparam=obs["G_obs"],
            G_theory_at_obs=float(gap_exact_ph(obs["F1"], obs["F2"])),
            kappa_nonparam=kap,
            EO_naive=d2[f"N_{h}"].mean() / obs["F1"] if obs["F1"] > 0 else np.nan,
            EO_csc=d2[f"C_{h}"].mean() / obs["F1"] if obs["F1"] > 0 else np.nan,
            inflation_theory_pooled=float(
                naive_exact_ph(obs["F1"], obs["F2"]) / obs["F1"]) if obs["F1"] > 0 else np.nan,
        ))
    pd.DataFrame(eo_rows).to_csv(OUT / "temporal_eo_context.csv", index=False)
    print(pd.DataFrame(eo_rows).to_string(index=False))

    meta = dict(seed=SEED, n_boot=N_BOOT, n_test=n,
                numpy=np.__version__, pandas=pd.__version__)
    (OUT / "decomposition_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
