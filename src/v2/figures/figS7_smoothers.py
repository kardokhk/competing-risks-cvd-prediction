"""Figure S7. The naive-versus-competing ICI difference survives a concordant
smoother, at a smaller magnitude.

Austin and Putter (Stat Med 2026;45:e70468) show that estimating the smoothed
observed probability behind ICI with a model discordant with the type of model
being assessed manufactures apparent miscalibration, always away from zero. The
manuscript leads with a naive-versus-competing ICI difference, so the choice of
smoother has to be tested rather than assumed.

A. Paired difference in ICI for the primary learner-matched pair under six
   smoothers, both splits.
B. The two models' own ICI under each smoother, temporal split, so that the
   reader can see whether a smoother moves one model or both.

Only the primary learner-matched pair is shown. Neither deep model belongs to a
naive-versus-competing pair, so adding them would not bear on the contrast this
figure tests, and eight more series would crowd a 136 mm panel.

The cross-validated split is excluded because its smoothed-calibration intervals
are not usable: on that split the pooled point estimate falls outside its own
percentile interval for 90 rows of `main_metrics.csv` and 45 of
`paired_contrasts.csv`, with a median gap of 0.69 of the interval width.

Smooth-Net is an extension, not in Austin and Putter: a naive model predicts
1 - S(t), so the smoother that estimates that same quantity is the concordant one
for it. `ici_concordant_austin` is whichever of Smooth-FG and Smooth-CSH is
concordant with that model's own type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as F
from figlib import plt

H = 10
PT = 6.0
SMOOTHERS = [
    ("ici", "Pseudo-\nobservation"),
    ("ici_concordant", "Estimand-\nmatched"),
    ("ici_smoothfg", "Smooth-FG"),
    ("ici_smoothcsh", "Smooth-CSH"),
    ("ici_smoothnet", "Smooth-Net"),
    ("ici_concordant_austin", "Concordant\n(Austin)"),
]
# Only the temporal split is plotted. The pooled point estimate falls OUTSIDE its
# own percentile interval for 90 rows of `main_metrics.csv` and 45 of
# `paired_contrasts.csv` (the rows flagged `interval_valid = FALSE`); every one is
# on `cv` and every one is a smoothed-calibration metric. The median gap is 0.69
# of the interval width, so it is not rounding.
SPLITS = [("temporal", "Temporal split", "o", 0.0)]


# The four Austin-concordant smoother metrics come from the standalone concordant
# tables rather than from the merged ones; see figlib.concordant_metrics().
FROM_CONCORDANT = {"ici_smoothfg", "ici_smoothcsh", "ici_smoothnet",
                   "ici_concordant_austin"}


def build():
    # Take the four smoother-specific metrics from the standalone concordant
    # tables and everything else from the merged ones. The merged tables sometimes
    # carry these metrics and sometimes do not, depending on whether the merge has
    # been rerun, so selecting by source keeps the figure correct either way. The
    # two sources agree exactly where they overlap; checked.
    def _pick(merged, standalone):
        return pd.concat([merged[~merged.metric.isin(FROM_CONCORDANT)],
                          standalone[standalone.metric.isin(FROM_CONCORDANT)]],
                         ignore_index=True)

    pc = _pick(F.contrasts(), F.concordant_contrasts())
    mm = _pick(F.main_metrics(), F.concordant_metrics())
    fig = plt.figure(figsize=(F.W_ONEHALF / F.MM, 72.0 / F.MM))
    gs = fig.add_gridspec(2, 1, hspace=0.28)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])

    xs = np.arange(len(SMOOTHERS), dtype=float)
    rows = []
    for split, slabel, mk, off in SPLITS:
        sub = F.pick(pc, split=split, horizon_y=float(H),
                     contrast="cox_naive - csc_cox")
        for j, (metric, _lab) in enumerate(SMOOTHERS):
            r = F.one(sub, metric=metric)
            x = xs[j] + off
            col = F.C_NAIVE if split == "temporal" else F.C_CR
            axA.plot([x, x], [r.lo * 1000, r.hi * 1000], color=col, lw=0.8,
                     zorder=3)
            axA.plot([x], [r.est * 1000], marker=mk, ms=3.2, color=col,
                     markerfacecolor=col if r.ci_excludes_zero else "white",
                     markeredgecolor=col, markeredgewidth=0.7, zorder=4)
            rows.append(dict(panel="A", split=split, smoother=metric,
                             model="cox_naive - csc_cox", est=r.est * 1000,
                             lo=r.lo * 1000, hi=r.hi * 1000,
                             ci_excludes_zero=bool(r.ci_excludes_zero),
                             source=("concordant_contrasts.csv"
                                     if metric in FROM_CONCORDANT
                                     else "paired_contrasts.csv"),
                             units="ICI per 1,000"))
    F.zero_line(axA, vertical=False)
    axA.set_xticks(xs)
    axA.set_xticklabels([])
    axA.set_xlim(-0.5, len(SMOOTHERS) - 0.5)
    axA.set_xlabel("")
    axA.set_ylabel("Δ ICI, naive − competing\nrisk (per 1,000)")
    F.thin_spines(axA)
    for split, slabel, mk, _o in SPLITS:
        col = F.C_NAIVE if split == "temporal" else F.C_CR
        axA.plot([], [], marker=mk, ms=3.2, color=col, ls="none", label=slabel)
    axA.legend(loc="upper right", fontsize=PT, handlelength=1.2, borderpad=0.1,
               labelspacing=0.24, borderaxespad=0.3, ncol=2, columnspacing=1.0)
    axA.text(0.015, 0.02,
             "Primary learner-matched pair, temporal split. The cross-validated "
             "intervals for these\nsmoothers are not usable and are withheld; "
             "see the caption.",
             transform=axA.transAxes, fontsize=PT, color=F.C_GREY, ha="left",
             va="bottom", linespacing=1.3)

    for mdl, mk, off in [("cox_naive", "o", -0.16), ("csc_cox", "s", 0.16)]:
        col = F.FAM_STYLE[F.FAMILY_OF[mdl]]["color"]
        sub = F.pick(mm, split="temporal", horizon_y=float(H), model=mdl)
        for j, (metric, _lab) in enumerate(SMOOTHERS):
            r = F.pick(sub, metric=metric)
            if r.empty or pd.isna(r.iloc[0].est):
                continue
            r = r.iloc[0]
            x = xs[j] + off
            axB.plot([x, x], [r.lo * 1000, r.hi * 1000], color=col, lw=0.8,
                     zorder=3)
            axB.plot([x], [r.est * 1000], marker=mk, ms=3.2, color=col,
                     zorder=4)
            rows.append(dict(panel="B", split="temporal", smoother=metric,
                             model=mdl, est=r.est * 1000, lo=r.lo * 1000,
                             hi=r.hi * 1000, ci_excludes_zero=np.nan,
                             source=("concordant_metrics.csv"
                                     if metric in FROM_CONCORDANT
                                     else "main_metrics.csv"),
                             units="ICI per 1,000"))
    axB.set_xticks(xs)
    axB.set_xticklabels([lab for _m, lab in SMOOTHERS])
    axB.set_xlim(-0.5, len(SMOOTHERS) - 0.5)
    axB.set_ylim(0, 24)
    axB.set_xlabel("Smoother used for the observed probability")
    axB.set_ylabel("ICI, temporal split\n(per 1,000)")
    F.thin_spines(axB)
    for mdl, mk in [("cox_naive", "o"), ("csc_cox", "s")]:
        axB.plot([], [], marker=mk, ms=3.2,
                 color=F.FAM_STYLE[F.FAMILY_OF[mdl]]["color"], ls="none",
                 label=F.LABEL[mdl])
    axB.legend(loc="upper right", fontsize=PT, handlelength=1.2, borderpad=0.1,
               labelspacing=0.24, borderaxespad=0.3, ncol=2, columnspacing=1.0)

    F.panel_letters([axA, axB], ["A", "B"], dx=-11.0, dy=5.0)
    return fig, pd.DataFrame(rows).assign(horizon_y=H)


if __name__ == "__main__":
    fig, data = build()
    F.report(F.save(fig, "figS7_smoothers", data=data, width_mm=F.W_ONEHALF))
