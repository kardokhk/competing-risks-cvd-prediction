"""Supplementary Figure S1. Discrimination is the same for every model.

A. IPCW time-dependent AUC at 10 years for all ten models, both splits.
B. Truncated IPCW concordance for the same models and splits.
C. The paired contrasts on both discrimination measures, temporal split. The
   two contrasts that are not learner-matched are drawn in grey and labelled,
   because a difference between models that do not share a learner says nothing
   about competing risks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as F
from figlib import plt

H = 10
PT = 6.0
ORDER = ["logistic", "cox_naive", "rsf_naive", "gbs_naive",
         "logistic_cr", "csc_cox", "fine_gray", "rsf_cr", "deephit", "nfg"]
# Marker SHAPE carries the family (as in Figures 1 to 3) and marker FILL
# carries the split, so neither distinction depends on colour.
SPLITS = [("temporal", "Temporal", True, -0.17),
          ("cv", "Cross-validated", False, 0.17)]
CONTRASTS = [
    ("cox_naive - csc_cox", "Cox", True),
    ("rsf_naive - rsf_cr", "Random survival forest", True),
    ("logistic - logistic_cr", "IPCW binomial", True),
    ("cox_naive - fine_gray", "Cox naive vs Fine-Gray", False),
    ("gbs_naive - deephit", "Gradient boosting vs DeepHit", False),
]


def _levels(ax, mm, metric, title, rows):
    ys = np.arange(len(ORDER), dtype=float)[::-1]
    for split, slabel, filled, off in SPLITS:
        sub = F.pick(mm, split=split, horizon_y=float(H), metric=metric)
        for i, mdl in enumerate(ORDER):
            r = F.one(sub, model=mdl)
            fam = F.FAMILY_OF[mdl]
            col = F.FAM_STYLE[fam]["color"]
            mk = F.FAM_STYLE[fam]["marker"]
            y = ys[i] + off
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=0.8, zorder=3)
            ax.plot([r.est], [y], marker=mk, ms=2.8, color=col,
                    markerfacecolor=col if filled else "white",
                    markeredgecolor=col, markeredgewidth=0.6, zorder=4)
            rows.append(dict(panel=title, metric=metric, split=split,
                             model=mdl, family=fam, est=r.est, lo=r.lo,
                             hi=r.hi, learner_matched=np.nan))
    n_naive = sum(1 for m in ORDER if F.FAMILY_OF[m] == "naive")
    ax.axhline(len(ORDER) - n_naive - 0.5, color=F.C_GREY, lw=0.5,
               ls=(0, (3, 2)), zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([F.LABEL[m] for m in ORDER])
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.set_xlabel(title)
    F.thin_spines(ax)


def build():
    mm = F.main_metrics()
    pc = F.pick(F.contrasts(), split="temporal", horizon_y=float(H))

    fig = plt.figure(figsize=(F.W_FULL / F.MM, 78.0 / F.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.80, 1.15], wspace=0.10)
    axA, axB, axC = (fig.add_subplot(gs[0, j]) for j in range(3))

    rows = []
    _levels(axA, mm, "auc", "IPCW time-dependent AUC at 10 y", rows)
    _levels(axB, mm, "cindex", "Truncated IPCW concordance", rows)
    axB.set_yticklabels([])

    ys = np.arange(len(CONTRASTS), dtype=float)[::-1]
    for k, (metric, mlabel, mk, off) in enumerate(
            [("auc", "AUC", "D", -0.17), ("cindex", "Concordance", "^", 0.17)]):
        for i, (contrast, clabel, matched) in enumerate(CONTRASTS):
            r = F.one(pc, contrast=contrast, metric=metric)
            col = "black" if matched else F.C_LIGHT
            y = ys[i] + off
            axC.plot([r.lo, r.hi], [y, y], color=col, lw=0.8, zorder=3)
            axC.plot([r.est], [y], marker=mk, ms=2.8, color=col,
                     markerfacecolor=col if r.ci_excludes_zero else "white",
                     markeredgecolor=col, markeredgewidth=0.6, zorder=4)
            rows.append(dict(panel="C", metric=metric, split="temporal",
                             model=contrast, family="contrast", est=r.est,
                             lo=r.lo, hi=r.hi, learner_matched=matched))
    F.zero_line(axC)
    axC.set_yticks(ys)
    axC.set_yticklabels([c[1] + ("" if c[2] else " *") for c in CONTRASTS])
    axC.set_ylim(-0.7, len(CONTRASTS) - 0.3)
    axC.set_xlabel("Paired difference, naive − competing risk")
    axC.yaxis.tick_right()
    axC.yaxis.set_label_position("right")
    axC.spines["left"].set_visible(False)
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(True)
    axC.text(0.02, 0.02, "* not learner-matched", transform=axC.transAxes,
             fontsize=PT, color=F.C_GREY, ha="left", va="bottom")

    axA.plot([], [], "o", ms=2.8, color=F.C_NAIVE, ls="none", label="Naive")
    axA.plot([], [], "s", ms=2.8, color=F.C_CR, ls="none",
             label="Competing risk")
    axA.plot([], [], "o", ms=2.8, color=F.C_GREY, ls="none",
             label="Temporal (filled)")
    axA.plot([], [], "o", ms=2.8, color=F.C_GREY, markerfacecolor="white",
             ls="none", label="Cross-validated (open)")
    axA.legend(loc="lower left", bbox_to_anchor=(0.0, 1.00), fontsize=PT,
               handlelength=1.2, borderpad=0.1, labelspacing=0.24,
               borderaxespad=0.0, ncol=2, columnspacing=1.0)

    axC.plot([], [], "D", ms=2.8, color="black", ls="none", label="AUC")
    axC.plot([], [], "^", ms=2.8, color="black", ls="none",
             label="Concordance")
    axC.legend(loc="lower left", bbox_to_anchor=(0.0, 1.00), fontsize=PT,
               handlelength=1.2, borderpad=0.1, labelspacing=0.24,
               borderaxespad=0.0, ncol=2, columnspacing=1.0)

    F.panel_letters([axA], ["A"], dx=-52.0, dy=3.0)
    F.panel_letters([axB], ["B"], dx=-8.0, dy=3.0)
    F.panel_letters([axC], ["C"], dx=-8.0, dy=3.0)
    return fig, pd.DataFrame(rows).assign(horizon_y=H)


if __name__ == "__main__":
    fig, data = build()
    F.report(F.save(fig, "figS1_discrimination", data=data, width_mm=F.W_FULL))
