"""Figure 3. The closed form for the naive minus competing-risk gap.

A. What the gap is made of at 10 years, with the 5- and 15-year shares overlaid.
   The difference in mean predicted risk between the naive Cox model and the
   cause-specific Cox model is decomposed into a change of estimand evaluated at
   the cohort's mean incidences, a heterogeneity (Jensen) term, a hazard-timing
   term and an implementation term.
B. The pooled nomogram. Predicted naive inflation A / F1 as a function of the two
   cumulative incidences, with the 65 sufficient NHANES strata, the three pooled
   temporal-test rows and one external published cohort overlaid. The external
   check is at the same pooled/marginal level as the grid: it tests whether the
   identity reproduces a published Kaplan-Meier versus Aalen-Johansen
   discrepancy, NOT whether it predicts a model-to-model difference, which needs
   the heterogeneity term as well.

The pooled-versus-per-subject caveat and the external check's numbers are carried
by the caption in figures/v2/captions.md rather than by a box on the image.
C. Why the nomogram is a lower bound. The closed form evaluated at mean
   incidences recovers about a third of the observed gap; evaluated per subject
   and averaged it recovers all of it, a factor of about 3.25 at every horizon.

No mathtext is used anywhere: the mathtext fallback chain reaches DejaVu for the
cursive family and would silently mix typefaces.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

import figlib as F
from figlib import plt

HORIZONS = [5, 10, 15]
PRIMARY_H = 10
PT = 6.0

COMPONENTS = [
    ("sh_estimand", "Change of\nestimand"),
    ("sh_heterogeneity", "Hetero-\ngeneity"),
    ("sh_timing", "Hazard\ntiming"),
    ("sh_implementation", "Implemen-\ntation"),
]
OTHER_MK = {5: "v", 15: "^"}


def _decomp(ax, dec, boot, rows):
    """Share of the observed difference carried by each component.

    Bars are used because these are shares of a whole with a meaningful zero and
    the bar length is the quantity; the value axis therefore includes zero. The
    whisker is the refit bootstrap interval on the share of D_model. The
    implementation term is deterministic and carries no interval.
    """
    xs = np.arange(len(COMPONENTS) + 1, dtype=float)
    for i, (q, _label) in enumerate(COMPONENTS):
        est = F.one(dec, horizon=PRIMARY_H, quantity=q).value * 100
        ax.bar(xs[i], est, width=0.62, color="white", edgecolor="black",
               linewidth=0.6, zorder=3)
        b = boot[boot.quantity == f"{q}_h{PRIMARY_H}"]
        lo = hi = np.nan
        if len(b):
            lo, hi = b.iloc[0].lo * 100, b.iloc[0].hi * 100
            ax.plot([xs[i]] * 2, [lo, hi], color="black", lw=0.7, zorder=4)
            for cap in (lo, hi):
                ax.plot([xs[i] - 0.13, xs[i] + 0.13], [cap] * 2, color="black",
                        lw=0.6, zorder=4)
        rows.append(dict(panel="A", quantity=q, horizon_y=PRIMARY_H, est=est,
                         lo=lo, hi=hi,
                         units="% of the observed difference in mean predicted risk"))
        for h in (5, 15):
            v = F.one(dec, horizon=h, quantity=q).value * 100
            ax.plot([xs[i]], [v], OTHER_MK[h], ms=2.8, markerfacecolor="white",
                    markeredgecolor=F.C_CR, markeredgewidth=0.7, zorder=5)
            rows.append(dict(panel="A", quantity=q, horizon_y=h, est=v,
                             lo=np.nan, hi=np.nan,
                             units="% of the observed difference in mean predicted risk"))
    ax.bar(xs[-1], 100.0, width=0.62, color="white", edgecolor="black",
           linewidth=0.6, hatch="////", zorder=3)
    rows.append(dict(panel="A", quantity="total", horizon_y=PRIMARY_H, est=100.0,
                     lo=np.nan, hi=np.nan, units="% (definition)"))

    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _q, lab in COMPONENTS] + ["Observed\ndifference"])
    ax.set_xlabel("Component of the decomposition")
    ax.set_ylabel("Share of the naive\nminus competing-risk\ndifference (%)",
                  labelpad=3)
    ax.set_ylim(-16, 128)
    ax.set_xlim(-0.62, len(COMPONENTS) + 0.62)
    F.thin_spines(ax)
    ax.plot([], [], "v", ms=2.8, markerfacecolor="white",
            markeredgecolor=F.C_CR, ls="none", label="5 y")
    ax.plot([], [], "^", ms=2.8, markerfacecolor="white",
            markeredgecolor=F.C_CR, ls="none", label="15 y")
    ax.legend(loc="upper left", fontsize=PT, handlelength=1.0, borderpad=0.1,
              labelspacing=0.22, borderaxespad=0.2, ncol=2, columnspacing=0.7,
              title="Bars: 10 y", title_fontsize=PT)


def _nomogram(ax, grid, ov, ext, rows):
    piv = grid.pivot(index="F2", columns="F1", values="inflation")
    X, Y = np.meshgrid(piv.columns.values * 100, piv.index.values * 100)
    Z = piv.values
    levels = [1.05, 1.10, 1.20, 1.40, 1.60]
    cs = ax.contour(X, Y, Z, levels=levels, colors=["#333333"], linewidths=0.6,
                    zorder=2)
    # Every contour is labelled at F1 = 27%, a column that carries no stratum
    # (the largest stratum F1 is 24.5%), no pooled row, the external check point,
    # nor the legend, and at which all five levels exist.
    f1, f2 = piv.columns.values * 100, piv.index.values * 100
    col = int(np.argmin(np.abs(f1 - 27.0)))
    manual = [(f1[col], f2[int(np.argmin(np.abs(Z[:, col] - lv)))])
              for lv in levels]
    ax.clabel(cs, fmt=lambda v: f"{v:.2f}", fontsize=PT, inline=True,
              inline_spacing=2, manual=manual)

    st = ov[ov.source == "nhanes_stratum"]
    po = ov[ov.source != "nhanes_stratum"]
    ax.plot(st.F1 * 100, st.F2 * 100, "o", ms=2.2, markerfacecolor="none",
            markeredgecolor=F.C_CR, markeredgewidth=0.55, zorder=4,
            label=f"NHANES stratum (n = {len(st)})")
    ax.plot(po.F1 * 100, po.F2 * 100, "D", ms=3.4, color=F.C_NAIVE,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5,
            label="Temporal test set, pooled")
    for _, r in po.iterrows():
        dx, dy, va = {5: (5.0, -6.0, "top"),
                      10: (6.0, -1.0, "center"),
                      15: (5.0, 7.0, "bottom")}[int(r.horizon)]
        ax.annotate(f"{int(r.horizon)} y", (r.F1 * 100, r.F2 * 100),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=PT, color=F.C_NAIVE_TXT, va=va)
    for _, r in ov.iterrows():
        rows.append(dict(panel="B", quantity=f"overlay:{r.source}:{r.label}",
                         horizon_y=r.horizon, est=r.inflation_theory,
                         lo=r.F1 * 100, hi=r.F2 * 100,
                         units="est = pooled inflation; lo = F1 (%); hi = F2 (%)"))

    # External check on a published cohort. Pooled/marginal, like the grid.
    e = ext.iloc[0]
    ax.plot([e.F1 * 100], [e.F2 * 100], "*", ms=6.0, color=F.C_ACC,
            markeredgecolor="black", markeredgewidth=0.5, zorder=6,
            label="External check (Truchot 2025)")
    # Only the study name goes in the panel; the numbers and the pooled-level
    # caveat are in the caption, so they cannot collide with a contour label.
    # Below the marker, not beside it: the 1.05 contour runs horizontally
    # through the star and would strike through a label placed on that line.
    ax.annotate(e.study, (e.F1 * 100, e.F2 * 100), textcoords="offset points",
                xytext=(3.0, -5.5), fontsize=PT, color=F.C_ACC, va="top",
                ha="left")
    rows.append(dict(panel="B", quantity=f"external_check:{e.study}",
                     horizon_y=np.nan, est=e.closed_form_inflation,
                     lo=e.F1 * 100, hi=e.F2 * 100,
                     units=("est = pooled inflation from the identity; lo = F1 (%); "
                            f"hi = F2 (%); published KM/AJ ratio {e.published_ratio}")))

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 60)
    ax.set_xlabel("Cumulative incidence of the event of interest, F1 (%)")
    ax.set_ylabel("Cumulative incidence of the competing event, F2 (%)")
    F.thin_spines(ax)
    ax.legend(loc="upper left", fontsize=PT, handlelength=1.2, borderpad=0.15,
              labelspacing=0.26, borderaxespad=0.4,
              title="Contour: pooled inflation A / F1", title_fontsize=PT)


def _lower_bound(ax, dec, boot, rows):
    """How much of the observed difference each way of using the closed form
    recovers. Evaluated at the cohort's mean incidences it recovers about a
    third; evaluated per subject and averaged it recovers all of it. The gap
    between the two series is the heterogeneity multiplier, annotated.

    The refit bootstrap gives an interval for the pooled share only; the
    per-subject share is the sum of two shares whose joint distribution is not
    recorded, so it is plotted without an interval.
    """
    xs = np.arange(len(HORIZONS), dtype=float)
    pooled, persub, mults = [], [], []
    for h in HORIZONS:
        d_obs = F.one(dec, horizon=h, quantity="D_obs").value
        t_pool = F.one(dec, horizon=h, quantity="T_pool").value
        t_het = F.one(dec, horizon=h, quantity="T_het").value
        b = boot[boot.quantity == f"sh_estimand_h{h}"]
        lo, hi = (b.iloc[0].lo * 100, b.iloc[0].hi * 100) if len(b) else (np.nan,) * 2
        pooled.append((t_pool / d_obs * 100, lo, hi))
        persub.append((t_het / d_obs * 100, np.nan, np.nan))
        mults.append(t_het / t_pool)
        rows.append(dict(panel="C", quantity="recovered_pooled", horizon_y=h,
                         est=pooled[-1][0], lo=lo, hi=hi,
                         units="% of the observed difference recovered"))
        rows.append(dict(panel="C", quantity="recovered_per_subject",
                         horizon_y=h, est=persub[-1][0], lo=np.nan, hi=np.nan,
                         units="% of the observed difference recovered"))
        rows.append(dict(panel="C", quantity="heterogeneity_multiplier",
                         horizon_y=h, est=mults[-1], lo=np.nan, hi=np.nan,
                         units="ratio, per-subject to pooled"))

    ax.axhline(100, color=F.C_GREY, lw=0.5, ls=(0, (3, 2)), zorder=2)
    for j, h in enumerate(HORIZONS):
        p, lo, hi = pooled[j]
        q = persub[j][0]
        ax.plot([xs[j], xs[j]], [p, q], color=F.C_GREY, lw=0.5, zorder=2)
        ax.plot([xs[j]] * 2, [lo, hi], color=F.C_CR, lw=0.8, zorder=3)
        for cap in (lo, hi):
            ax.plot([xs[j] - 0.07, xs[j] + 0.07], [cap] * 2, color=F.C_CR,
                    lw=0.6, zorder=3)
        ax.plot([xs[j]], [p], "s", ms=3.2, color=F.C_CR, zorder=4)
        ax.plot([xs[j]], [q], "^", ms=3.4, color=F.C_ACC, zorder=4)
        ax.text(xs[j] + 0.10, (p + q) / 2, f"x {mults[j]:.2f}", ha="left",
                va="center", fontsize=PT, color=F.C_GREY)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{h}" for h in HORIZONS])
    ax.set_xlim(-0.42, 2.58)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Observed difference\nrecovered (%)", labelpad=3)
    F.thin_spines(ax)
    ax.plot([], [], "s", ms=3.2, color=F.C_CR, ls="none",
            label="Closed form at mean incidences")
    ax.plot([], [], "^", ms=3.4, color=F.C_ACC, ls="none",
            label="Closed form per subject, averaged")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.00), fontsize=PT,
              handlelength=1.2, borderpad=0.1, labelspacing=0.24,
              borderaxespad=0.0, ncol=2, columnspacing=1.0)


def build():
    dec = pd.read_csv(F.CF / "decomposition.csv")
    boot = pd.read_csv(F.CF / "decomposition_refit_boot.csv")
    grid = pd.read_csv(F.CF / "nomogram_data.csv")
    ov = pd.read_csv(F.CF / "nomogram_overlay.csv")
    ext = pd.read_csv(F.CF / "external_check_truchot.csv")

    fig = plt.figure(figsize=(F.W_FULL / F.MM, 108.0 / F.MM))
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.03)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.90, 1.10],
                          height_ratios=[1.0, 1.0], wspace=0.20, hspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[1, 0])
    axB = fig.add_subplot(gs[:, 1])

    rows = []
    _decomp(axA, dec, boot, rows)
    _nomogram(axB, grid, ov, ext, rows)
    _lower_bound(axC, dec, boot, rows)

    F.panel_letters([axA, axC], ["A", "C"], dx=-13.0, dy=4.0)
    F.panel_letters([axB], ["B"], dx=-24.0, dy=4.0)
    return fig, pd.DataFrame(rows)


if __name__ == "__main__":
    fig, data = build()
    F.report(F.save(fig, "fig03_closed_form", data=data, width_mm=F.W_FULL))
