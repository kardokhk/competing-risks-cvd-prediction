#!/usr/bin/env python
"""P2-B step 6: nomogram grid for the exact closed form, plus the real-cohort
overlay and the heterogeneity warning that has to travel with it.

nomogram_data.csv  grid over F_1 in 0.5-30% and F_2 in 0.5-60% giving
   A_exact       1 - (1 - F1 - F2)^(F1/(F1+F2))
   inflation     A_exact / F_1
   gap_abs       A_exact - F_1
These are POOLED quantities: they say what happens to a group whose members all
carry the same F_1 and F_2.  In a cohort with heterogeneous risk the mean of the
per-subject gaps is larger, because the gap is convex in F_2 and F_1i and F_2i
are strongly positively correlated.  Columns gap_abs_m2 and gap_abs_m3 give the
gap multiplied by 2 and by 3 to bracket the heterogeneity multiplier observed
here (2.5 to 3.3 across horizons on NHANES 1999-2018).  Read the plain columns
as a LOWER BOUND for a heterogeneous cohort.

nomogram_overlay.csv  where real strata sit on that grid, with the observed
gap alongside the theoretical one.

Seed 20260903.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cf_lib import naive_exact_ph, gap_exact_ph  # noqa: E402

OUT = ROOT / "results" / "closed_form"


def main() -> None:
    f1 = np.round(np.arange(0.005, 0.3001, 0.005), 4)
    f2 = np.round(np.arange(0.005, 0.6001, 0.005), 4)
    F1, F2 = np.meshgrid(f1, f2, indexing="ij")
    ok = (F1 + F2) < 1.0
    A = naive_exact_ph(F1, F2)
    G = gap_exact_ph(F1, F2)
    g = pd.DataFrame(dict(
        F1=F1.ravel(), F2=F2.ravel(),
        p=(F1 + F2).ravel(), pi=(F1 / (F1 + F2)).ravel(),
        A_exact=A.ravel(), inflation=(A / F1).ravel(), gap_abs=G.ravel(),
        gap_abs_m2=(2 * G).ravel(), gap_abs_m3=(3 * G).ravel(),
    ))[ok.ravel()]
    g.to_csv(OUT / "nomogram_data.csv", index=False)
    print(f"grid rows {len(g)}; inflation range "
          f"{g.inflation.min():.4f} to {g.inflation.max():.4f}")

    # ---- real-cohort overlay -------------------------------------------------
    rows = []
    v = pd.read_csv(OUT / "validation.csv")
    v = v[v["sufficient"] & v["gap_obs"].gt(0)]
    for _, r in v.iterrows():
        rows.append(dict(source="nhanes_stratum", label=r["stratum"],
                         family=r["family"], horizon=r["horizon"], n=r["n"],
                         F1=r["F1_obs"], F2=r["F2_obs"],
                         gap_obs=r["gap_obs"], gap_theory=r["gap_theory"],
                         inflation_theory=float(
                             naive_exact_ph(r["F1_obs"], r["F2_obs"]) / r["F1_obs"]),
                         heterogeneity_multiplier=np.nan))
    ctx = pd.read_csv(OUT / "temporal_eo_context.csv")
    dec = pd.read_csv(OUT / "decomposition.csv")
    piv = dec.pivot(index="horizon", columns="quantity", values="value")
    for _, r in ctx.iterrows():
        h = int(r["horizon"])
        rows.append(dict(source="nhanes_temporal_test_pooled", label=f"{h} y pooled",
                         family="pooled", horizon=h, n=r["n"],
                         F1=r["F1_obs"], F2=r["F2_obs"],
                         gap_obs=r["G_obs_nonparam"], gap_theory=r["G_theory_at_obs"],
                         inflation_theory=r["inflation_theory_pooled"],
                         heterogeneity_multiplier=piv.loc[h, "T_het"] / piv.loc[h, "T_pool"]))
    ov = pd.DataFrame(rows)
    ov.to_csv(OUT / "nomogram_overlay.csv", index=False)
    print("heterogeneity multiplier T_het / T_pool by horizon:")
    print((piv["T_het"] / piv["T_pool"]).to_string())

    # ---- diagnostic PNGs (not publication figures) --------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    P = g.pivot(index="F1", columns="F2", values="inflation")
    cs = ax[0].contour(P.columns * 100, P.index * 100, P.values,
                       levels=[1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0], colors="k")
    ax[0].clabel(cs, inline=True, fontsize=7)
    s = ov[ov.source == "nhanes_stratum"]
    ax[0].scatter(s.F2 * 100, s.F1 * 100, s=10, alpha=.6, label="NHANES strata")
    ax[0].scatter(ov[ov.source != "nhanes_stratum"].F2 * 100,
                  ov[ov.source != "nhanes_stratum"].F1 * 100,
                  s=60, marker="*", color="crimson", label="pooled test set")
    ax[0].set_xlabel("competing-death CIF F2 (%)"); ax[0].set_ylabel("CVD CIF F1 (%)")
    ax[0].set_title("pooled naive inflation A / F1"); ax[0].legend(fontsize=7)
    vv = pd.read_csv(OUT / "validation.csv"); vv = vv[vv.sufficient & vv.gap_obs.gt(0)]
    ax[1].scatter(vv.kappa_nonparam, vv.rel_err, s=14, c=vv.horizon, cmap="viridis")
    ax[1].axhline(0, lw=.5); ax[1].axvline(0.5, lw=.5, ls="--")
    ax[1].set_xlabel("non-parametric kappa"); ax[1].set_ylabel("relative error of theory")
    ax[1].set_title("error tracks departure of kappa from 1/2")
    fig.tight_layout(); fig.savefig(OUT / "diag_nomogram.png", dpi=120)
    print("wrote diag_nomogram.png")


if __name__ == "__main__":
    main()
