"""Figure 4. DeepHit's calibration is set by one hyperparameter, and the cost of
its default shows up in the production fits.

Top row, the mechanism. The alpha in DeepHit's loss weights the likelihood term
against the ranking term. Sweeping it from 0 (pure ranking) to 1 (pure
likelihood), with the architecture, optimiser and early-stopping rule held fixed,
moves calibration by more than an order of magnitude and leaves concordance flat.

Bottom row, the consequence. The paired difference between the two deep
competing-risk models as they are fitted everywhere else in this paper, DeepHit
minus Neural Fine-Gray. Both are 10-member seed ensembles, so this contrast
isolates the training objective, a ranking-weighted likelihood against a monotone
cumulative-incidence parameterisation, rather than the competing-risk assumption.
It is not learner-matched in the sense used for the naive-versus-competing pairs.

THE TWO ROWS COME FROM DIFFERENT FITS. The sweep is a separate experiment run
before ensembling, 15 single fits at each setting; the bottom row is the
production ensembles. Their ICI values are not comparable with each other, and
only the bottom row is comparable with Figure 2 and the tables. That warning, and
the filled-means-excludes-zero convention, are carried by the caption in
figures/v2/captions.md.

Columns are paired by metric: A with D, B with E, C with F.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as F
from figlib import plt

SIGMA = 0.1
H = 10
PT = 6.0
CONTRAST = "deephit - nfg"

PROTOCOLS = [
    ("inner_oof", "Inner out-of-fold", F.C_CR, "s", "-"),
    ("temporal_test", "Temporal test set", F.C_NAIVE, "o", "--"),
]
# Bottom-row splits, coloured and shaped to match the protocols above them.
SPLITS = [("temporal", "Temporal", F.C_NAIVE, "o"),
          ("cv", "Cross-validated", F.C_CR, "s")]

# (sweep metric, sweep axis label, log y, reference line) paired with
# (contrast metric, contrast axis label, scale)
COLUMNS = [
    (("ici", "Integrated calibration index", True, None),
     ("ici", "Δ ICI (per 1,000)", 1000.0)),
    (("eo", "Expected / observed", True, 1.0),
     ("eo", "Δ E/O", 1.0)),
    (("cindex", "Truncated IPCW concordance", False, None),
     ("cindex", "Δ concordance", 1.0)),
]


def _sweep(ax, sw, metric, title, logy, ref, rows):
    for proto, plabel, col, mk, ls in PROTOCOLS:
        g = sw[sw.protocol == proto]
        a = g["alpha"].values
        med, lo, hi = (g[f"{metric}_{k}"].values for k in ("median", "p10", "p90"))
        ax.plot(a, med, color=col, lw=1.0, ls=ls, zorder=3)
        for x, l, h in zip(a, lo, hi):
            ax.plot([x, x], [l, h], color=col, lw=0.6, zorder=3)
        ax.plot(a, med, marker=mk, ms=2.6, color=col, ls="none", zorder=4)
        for x, m, l, h, n in zip(a, med, lo, hi, g["n_rep"].values):
            rows.append(dict(row="sweep", panel_metric=metric, protocol=proto,
                             alpha=x, sigma=SIGMA, est=m, lo=l, hi=h,
                             n_replicates=n, ci_excludes_zero=np.nan,
                             units=title))
    if ref is not None:
        ax.axhline(ref, color=F.C_GREY, lw=0.5, ls=(0, (3, 2)), zorder=2)
    if logy:
        ax.set_yscale("log")
    ax.axvline(0.2, color=F.C_GREY, lw=0.5, ls=(0, (1, 2)), zorder=1)
    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks([0, 0.2, 0.5, 0.7, 1.0])
    ax.set_xlabel("Weight on the likelihood term, alpha")
    ax.set_ylabel(title)
    F.thin_spines(ax)


def _contrast(ax, pc, metric, title, scale, rows):
    ys = np.arange(len(SPLITS), dtype=float)[::-1]
    for i, (split, slabel, col, mk) in enumerate(SPLITS):
        r = F.one(pc, split=split, horizon_y=float(H), contrast=CONTRAST,
                  metric=metric)
        y = ys[i]
        ax.plot([r.lo * scale, r.hi * scale], [y, y], color=col, lw=0.9,
                solid_capstyle="butt", zorder=3)
        for cap in (r.lo, r.hi):
            ax.plot([cap * scale] * 2, [y - 0.13, y + 0.13], color=col, lw=0.7,
                    zorder=3)
        ax.plot([r.est * scale], [y], marker=mk, ms=3.2, color=col,
                markerfacecolor=col if r.ci_excludes_zero else "white",
                markeredgecolor=col, markeredgewidth=0.7, zorder=4)
        rows.append(dict(row="contrast", panel_metric=metric, protocol=split,
                         alpha=np.nan, sigma=np.nan, est=r.est * scale,
                         lo=r.lo * scale, hi=r.hi * scale, n_replicates=np.nan,
                         ci_excludes_zero=bool(r.ci_excludes_zero), units=title))
    F.zero_line(ax)
    ax.set_ylim(-0.55, len(SPLITS) - 0.45)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[1].replace("Cross-validated", "Cross-val.")
                        for s in SPLITS])
    ax.set_xlabel(title)
    F.thin_spines(ax)


def build():
    sw = pd.read_csv(F.DEEP / "alpha_sweep_summary.csv")
    sw = sw[(sw.model == "deephit") & (sw.sigma == SIGMA)].sort_values("alpha")
    pc = F.contrasts()

    fig = plt.figure(figsize=(F.W_FULL / F.MM, 100.0 / F.MM))
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.03)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.30, 0.66], wspace=0.22,
                          hspace=0.32)
    top = [fig.add_subplot(gs[0, j]) for j in range(3)]
    bot = [fig.add_subplot(gs[1, j]) for j in range(3)]
    rows = []
    for j, ((sm, stitle, logy, ref), (cm, ctitle, scale)) in enumerate(COLUMNS):
        _sweep(top[j], sw, sm, stitle, logy, ref, rows)
        _contrast(bot[j], pc, cm, ctitle, scale, rows)
        if j:
            bot[j].set_yticklabels([])

    top[0].set_yticks([0.003, 0.01, 0.03, 0.1])
    top[0].set_yticklabels(["0.003", "0.01", "0.03", "0.10"])
    top[0].minorticks_off()
    top[1].set_yticks([1, 2, 5])
    top[1].set_yticklabels(["1", "2", "5"])
    top[1].minorticks_off()
    top[2].set_ylim(0.795, 0.845)
    top[0].text(0.22, 0.98, "alpha = 0.2,\nprior work",
                transform=top[0].get_xaxis_transform(), ha="left", va="top",
                fontsize=PT, color=F.C_GREY)

    for proto, plabel, col, mk, ls in PROTOCOLS:
        top[2].plot([], [], marker=mk, ms=2.6, color=col, lw=1.0, ls=ls,
                    label=plabel)
    top[2].legend(loc="lower right", fontsize=PT, handlelength=1.8,
                  borderpad=0.1, labelspacing=0.24, borderaxespad=0.3)
    # The fill convention, and the warning that the two rows come from different
    # fits, are carried by the caption rather than by a box on the image.
    F.panel_letters(top, ["A", "B", "C"], dx=-13.0, dy=4.0)
    F.panel_letters(bot, ["D", "E", "F"], dx=-13.0, dy=14.0)
    return fig, pd.DataFrame(rows).assign(contrast=CONTRAST, horizon_y=H)


if __name__ == "__main__":
    fig, data = build()
    F.report(F.save(fig, "fig04_alpha_sweep", data=data, width_mm=F.W_FULL))
