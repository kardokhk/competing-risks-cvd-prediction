"""Figure S8. What a single deep fit per imputation registers as between-imputation
spread is almost entirely the random seed.

A. For each statistic, the share of the spread across a single-fit-per-imputation
   run that is attributable to the seed rather than to the imputation:
   sd_seed / sd_single_seed_run. It sits between 93% and 102% for every statistic
   in both deep models. Values above 100% are possible because these are
   standard-deviation ratios of separately estimated components, not a partition.
B. The fraction of missing information for every pooled estimand, before and
   after replacing the single fit with a 10-member seed ensemble. Points below
   the identity line are estimands whose FMI fell.

The seed is not a nuisance to be averaged away quietly: before ensembling it was
inflating the between-imputation variance of these two models, and therefore
their Rubin intervals, by an amount that had nothing to do with missing data.

Sources: `results/deep/instability_decomposition.csv`,
`results/deep/fmi_before_after.csv`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as F
from figlib import plt

PT = 6.0
STATS = [
    ("mean_risk10", "Mean\nrisk"),
    ("frac_above_0.01", "> 1%"),
    ("frac_above_0.02", "> 2%"),
    ("frac_above_0.03", "> 3%"),
    ("frac_above_0.05", "> 5%"),
    ("frac_above_0.075", "> 7.5%"),
    ("frac_above_0.2", "> 20%"),
]
MODELS = [("deephit", "DeepHit", F.C_NAIVE, "o", -0.15),
          ("nfg", "Neural Fine-Gray", F.C_CR, "s", 0.15)]
SPLITS = [("temporal", "Temporal", True), ("cv", "Cross-validated", False)]


def _seed_share(ax, dec, rows):
    xs = np.arange(len(STATS), dtype=float)
    ax.axhline(100, color=F.C_GREY, lw=0.5, ls=(0, (3, 2)), zorder=2)
    for mdl, label, col, mk, off in MODELS:
        for j, (stat, _lab) in enumerate(STATS):
            r = F.one(dec, model=mdl, split="temporal", statistic=stat)
            ax.plot([xs[j] + off], [r.seed_share * 100], marker=mk, ms=3.2,
                    color=col, zorder=4)
            rows.append(dict(panel="A", model=mdl, split="temporal",
                             statistic=stat, quantity="seed_share_pct",
                             value=r.seed_share * 100, sd_seed=r.sd_seed,
                             sd_imputation=r.sd_imputation,
                             sd_single_seed_run=r.sd_single_seed_run,
                             n_seed=r.n_seed, n_imp=r.n_imp))
    lo, hi = dec.seed_share.min() * 100, dec.seed_share.max() * 100
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _s, lab in STATS])
    ax.set_xlim(-0.55, len(STATS) - 0.45)
    ax.set_ylim(86, 111)
    ax.set_xlabel("Statistic at 10 years, temporal split")
    ax.set_ylabel("Share of the single-fit spread\nattributable to the seed (%)")
    F.thin_spines(ax)
    ax.text(0.02, 0.045,
            f"Range over both models and all seven statistics: "
            f"{lo:.0f}% to {hi:.0f}%.\n"
            "Pure imputation component of the mean predicted risk at 10 y: "
            "0.00 per 1,000\n(DeepHit) and 0.64 per 1,000 (Neural Fine-Gray).",
            transform=ax.transAxes, fontsize=PT, color=F.C_GREY, ha="left",
            va="bottom", linespacing=1.35)
    for mdl, label, col, mk, _o in MODELS:
        ax.plot([], [], marker=mk, ms=3.2, color=col, ls="none", label=label)
    ax.legend(loc="upper left", fontsize=PT, handlelength=1.2, borderpad=0.1,
              labelspacing=0.24, borderaxespad=0.3, ncol=2, columnspacing=1.0)


def _fmi(ax, f, rows):
    ax.plot([0, 1.1], [0, 1.1], color=F.C_GREY, lw=0.5, ls=":", zorder=1)
    for mdl, label, col, mk, _o in MODELS:
        for split, slabel, filled in SPLITS:
            g = f[(f.model == mdl) & (f.split == split)]
            ax.plot(g.fmi_single, g.fmi_ens, marker=mk, ms=2.2, ls="none",
                    color=col, markerfacecolor=col if filled else "white",
                    markeredgecolor=col, markeredgewidth=0.5, zorder=3)
    for _, r in f.iterrows():
        rows.append(dict(panel="B", model=r.model, split=r.split,
                         statistic=f"{r.metric}@{r.horizon_y:g}y",
                         quantity="fmi", value=r.fmi_ens,
                         sd_seed=r.fmi_single, sd_imputation=np.nan,
                         sd_single_seed_run=np.nan,
                         n_seed=r.m_required_von_hippel_ens,
                         n_imp=r.m_required_von_hippel_single))

    keep = f[f.fmi_ens > 0.9]
    ax.plot(keep.fmi_single, keep.fmi_ens, "o", ms=5.0, markerfacecolor="none",
            markeredgecolor="black", markeredgewidth=0.6, zorder=5)
    ax.annotate("all four are the fraction of the\n"
                "sample above 1%, and stay at\n"
                f"{keep.fmi_ens.min():.2f} to {keep.fmi_ens.max():.2f}",
                (keep.fmi_single.mean(), keep.fmi_ens.mean()),
                textcoords="offset points", xytext=(-152.0, -12.0),
                fontsize=PT, color="black", ha="left", va="top",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="black",
                                shrinkA=1.0, shrinkB=5.0))

    n_red = int((f.fmi_ens < f.fmi_single - 1e-9).sum())
    n_up = int((f.fmi_ens > f.fmi_single + 1e-9).sum())
    ax.text(0.03, 0.60,
            "Shapes as in A: circle DeepHit, square Neural Fine-Gray.\n"
            f"{len(f)} pooled estimands.\n"
            f"FMI fell for {n_red}, rose for {n_up}.\n"
            f"Median {f.fmi_single.median():.3f} to {f.fmi_ens.median():.3f}.\n"
            f"Needing m > 30 by von Hippel's rule: "
            f"{int((f.m_required_von_hippel_single > 30).sum())} to "
            f"{int((f.m_required_von_hippel_ens > 30).sum())}.",
            transform=ax.transAxes, fontsize=PT, color=F.C_GREY, ha="left",
            va="top", linespacing=1.35)

    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("Fraction of missing information,\nsingle fit per imputation")
    ax.set_ylabel("Fraction of missing information,\n10-member seed ensemble")
    F.thin_spines(ax)
    for split, slabel, filled in SPLITS:
        ax.plot([], [], "o", ms=2.6, color=F.C_GREY,
                markerfacecolor=F.C_GREY if filled else "white",
                markeredgecolor=F.C_GREY, ls="none", label=slabel)
    ax.plot([], [], "o", ms=5.0, markerfacecolor="none", markeredgecolor="black",
            ls="none", label="FMI still > 0.9")
    ax.legend(loc="upper left", fontsize=PT, handlelength=1.2, borderpad=0.1,
              labelspacing=0.24, borderaxespad=0.3)


def build():
    dec = pd.read_csv(F.DEEP / "instability_decomposition.csv")
    f = pd.read_csv(F.DEEP / "fmi_before_after.csv")

    fig = plt.figure(figsize=(F.W_FULL / F.MM, 78.0 / F.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.20, 1.0], wspace=0.16)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    rows = []
    _seed_share(axA, dec, rows)
    _fmi(axB, f, rows)
    F.panel_letters([axA], ["A"], dx=-14.0, dy=5.0)
    F.panel_letters([axB], ["B"], dx=-14.0, dy=5.0)
    return fig, pd.DataFrame(rows)


if __name__ == "__main__":
    fig, data = build()
    F.report(F.save(fig, "figS8_instability", data=data, width_mm=F.W_FULL))
