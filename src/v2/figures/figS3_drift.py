"""Figure S3. The shared over-prediction on the temporal split is secular drift,
not a competing-risk artefact.

A. Expected over observed for the two Cox models under two validation designs.
   Out of fold on the whole 1999-2018 cohort, where training and test periods
   overlap, the cause-specific Cox model is essentially calibrated in the large
   at every horizon. Under the temporal split both models over-predict by a
   similar amount, and the competing-risk model over-predicts too, so the shared
   component cannot be attributed to the naive estimand.
B. The naive-to-competing E/O ratio, which is what the closed form predicts. It
   is stable across the two designs; the shared level shift is not.

The 15-year temporal rows are NOT estimable: maximum follow-up on the temporal
test set is 13.25 years and the fraction still at risk at 15 years is zero. They
are drawn as open markers with a hatched exclusion band and must not be quoted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as F
from figlib import plt

PT = 6.0
DESIGNS = [
    ("cv_oof_1999_2018", "Out of fold, 1999-2018", "s", -0.16),
    ("temporal_2007_2010", "Temporal, trained 1999-2006", "o", 0.16),
]
MODELS = [("cox_naive", F.C_NAIVE), ("csc_cox", F.C_CR)]
HORIZONS = [5, 10, 15]


def build():
    d = pd.read_csv(F.CF / "drift_check.csv")
    dec = pd.read_csv(F.CF / "decomposition.csv")
    # Per-subject closed-form prediction of the naive-to-competing E/O ratio at
    # 10 years: (mean F1 + mean per-subject gap) / mean F1.
    mean_c = F.one(dec, horizon=10, quantity="mean_C").value
    t_het = F.one(dec, horizon=10, quantity="T_het").value
    cf_ratio = (mean_c + t_het) / mean_c
    fig = plt.figure(figsize=(F.W_ONEHALF / F.MM, 66.0 / F.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.16)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    xs = np.arange(len(HORIZONS), dtype=float)
    rows = []
    for design, dlabel, mk, doff in DESIGNS:
        for mi, (mdl, col) in enumerate(MODELS):
            moff = (mi - 0.5) * 0.14
            for j, h in enumerate(HORIZONS):
                r = F.one(d, model=mdl, design=design, horizon=h)
                est = bool(r.at_risk_frac > 0)
                x = xs[j] + doff + moff
                axA.plot([x, x], [r.EO_lo, r.EO_hi], color=col, lw=0.8,
                         ls="-" if est else (0, (1, 1)), zorder=3)
                axA.plot([x], [r.EO], marker=mk, ms=3.0, color=col,
                         markerfacecolor=col if est else "white",
                         markeredgecolor=col, markeredgewidth=0.6, zorder=4)
                rows.append(dict(panel="A", model=mdl, design=design,
                                 horizon_y=h, quantity="EO", est=r.EO,
                                 lo=r.EO_lo, hi=r.EO_hi, n=r.n,
                                 n_events=r.n_ev1, at_risk_fraction=r.at_risk_frac,
                                 estimable=est))
    axA.axhline(1.0, color=F.C_GREY, lw=0.5, ls=(0, (3, 2)), zorder=2)
    # Only the TEMPORAL rows at 15 y are inestimable; the out-of-fold rows at
    # the same horizon are fine, so the flag points at the two markers it
    # applies to rather than shading the whole horizon.
    axA.annotate("open markers: not estimable,\nfraction at risk = 0",
                 xy=(2.16, 1.95), xytext=(1.35, 2.24), fontsize=PT,
                 color=F.C_GREY, ha="left", va="top", linespacing=1.3,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=F.C_GREY,
                                 shrinkA=1.0, shrinkB=2.0))
    axA.set_xticks(xs)
    axA.set_xticklabels([str(h) for h in HORIZONS])
    axA.set_xlim(-0.5, 2.5)
    axA.set_ylim(0.85, 2.35)
    axA.set_xlabel("Horizon (years)")
    axA.set_ylabel("Expected / observed")
    F.thin_spines(axA)

    for design, dlabel, mk, _o in DESIGNS:
        axA.plot([], [], marker=mk, ms=3.0, color=F.C_GREY, ls="none",
                 label=dlabel)
    for mdl, col in MODELS:
        axA.plot([], [], "_", color=col, ls="none", label=F.LABEL[mdl])
    axA.legend(loc="lower left", bbox_to_anchor=(0.0, 1.00), fontsize=PT,
               handlelength=1.2, borderpad=0.1, labelspacing=0.24,
               borderaxespad=0.0, ncol=2, columnspacing=1.0)

    for design, dlabel, mk, doff in DESIGNS:
        vals, keep = [], []
        for h in HORIZONS:
            a = F.one(d, model="cox_naive", design=design, horizon=h)
            b = F.one(d, model="csc_cox", design=design, horizon=h)
            vals.append(a.EO / b.EO)
            keep.append(bool(a.at_risk_frac > 0))
            rows.append(dict(panel="B", model="cox_naive / csc_cox",
                             design=design, horizon_y=h, quantity="EO_ratio",
                             est=vals[-1], lo=np.nan, hi=np.nan, n=a.n,
                             n_events=a.n_ev1, at_risk_fraction=a.at_risk_frac,
                             estimable=keep[-1]))
        for j, (v, k) in enumerate(zip(vals, keep)):
            axB.plot([xs[j] + doff], [v], marker=mk, ms=3.0, color="black",
                     markerfacecolor="black" if k else "white",
                     markeredgecolor="black", markeredgewidth=0.6, zorder=4)
    axB.axhline(cf_ratio, color=F.C_ACC, lw=0.8, ls="-", zorder=2)
    axB.text(2.45, cf_ratio + 0.006,
             f"predicted by the per-subject\nclosed form at 10 y ({cf_ratio:.3f})",
             ha="right", va="bottom", fontsize=PT, color=F.C_ACC,
             linespacing=1.3)
    rows.append(dict(panel="B", model="closed form, per subject",
                     design="reference", horizon_y=10, quantity="EO_ratio",
                     est=cf_ratio, lo=np.nan, hi=np.nan, n=np.nan,
                     n_events=np.nan, at_risk_fraction=np.nan, estimable=True))
    axB.set_xticks(xs)
    axB.set_xticklabels([str(h) for h in HORIZONS])
    axB.set_xlim(-0.5, 2.5)
    axB.set_ylim(1.00, 1.36)
    axB.set_xlabel("Horizon (years)")
    axB.set_ylabel("E/O ratio, naive / competing risk")
    F.thin_spines(axB)

    F.panel_letters([axA], ["A"], dx=-28.0, dy=3.0)
    F.panel_letters([axB], ["B"], dx=-28.0, dy=3.0)
    return fig, pd.DataFrame(rows)


if __name__ == "__main__":
    fig, data = build()
    F.report(F.save(fig, "figS3_drift", data=data, width_mm=F.W_ONEHALF))
