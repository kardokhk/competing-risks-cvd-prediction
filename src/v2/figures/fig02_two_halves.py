"""Figure 2. The two halves of the claim, side by side.

Left (A-C): the paired calibration contrasts for the three learner-matched
pairs. Every interval excludes zero.
Right (D-E): the same three contrasts on net benefit, with the fraction of the
sample above each threshold below.

The main-text version of this figure shows the four primary thresholds and the
secondary one. The 20% threshold is defined on incident atherosclerotic disease,
not on the mortality endpoint used here, the manuscript puts it in the supplement,
and its contrasts are five to twenty times larger than the primary ones, so
including it on a common axis compresses the thresholds that carry the argument
into a band of dots. It is restored, at the same scale as everything else, in the
supplementary version of this figure (`figS2_two_halves_full.py`), so nothing is
hidden: the reader sees it, on its own page, where the manuscript already puts
it.

Contrast is always naive minus competing risk, so a positive value on A means the
naive model over-predicts more, and a negative value on D means the naive model
buys fewer net true positives.

Marker fill is the same encoding throughout the figure: filled means the 95%
interval excludes zero, open means it contains zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as F
from figlib import plt

SPLIT, H = "temporal", 10
PT = 6.0
# Main text: primary and secondary thresholds. The supplementary build passes
# F.THRESHOLDS to restore the supplement-only 20% threshold.
MAIN_THRESHOLDS = [t for t in F.THRESHOLDS if F.TIER[t] != "supplement"]

PAIR_STYLE = [
    ("cox_naive - csc_cox", "Cox", "#000000", "o"),
    ("rsf_naive - rsf_cr", "Random survival forest", "#009E73", "^"),
    ("logistic - logistic_cr", "IPCW binomial", "#CC79A7", "D"),
]
CAL_PANELS = [
    ("eo", "\u0394 E/O (naive \u2212 competing risk)", 1.0, "ratio"),
    ("calib_slope", "\u0394 calibration slope", 1.0, "dimensionless"),
    ("ici", "\u0394 ICI (per 1,000)", 1000.0, "per 1,000"),
]


def _forest(ax, pc, metric, title, scale, rows):
    for i, (contrast, label, col, mk) in enumerate(PAIR_STYLE):
        r = F.one(pc, contrast=contrast, metric=metric)
        y = len(PAIR_STYLE) - 1 - i
        ax.plot([r.lo * scale, r.hi * scale], [y, y], color=col, lw=0.9,
                solid_capstyle="butt", zorder=3)
        for cap in (r.lo, r.hi):
            ax.plot([cap * scale] * 2, [y - 0.16, y + 0.16], color=col, lw=0.7,
                    zorder=3)
        ax.plot([r.est * scale], [y], marker=mk, ms=3.2, color=col,
                markerfacecolor=col if r.ci_excludes_zero else "white",
                markeredgecolor=col, markeredgewidth=0.7, zorder=4)
        rows.append(dict(panel_group="calibration", metric=metric,
                         contrast=contrast, label=label, x=np.nan,
                         est=r.est * scale, lo=r.lo * scale, hi=r.hi * scale,
                         ci_excludes_zero=bool(r.ci_excludes_zero),
                         units=title))
    F.zero_line(ax)
    ax.set_ylim(-0.62, len(PAIR_STYLE) - 0.38)
    ax.set_yticks(range(len(PAIR_STYLE)))
    ax.set_yticklabels([s[1].replace("Random survival forest", "RSF")
                        .replace("IPCW binomial", "Binomial")
                        for s in PAIR_STYLE][::-1])
    ax.set_xlabel(title)
    F.thin_spines(ax)


def build(split=SPLIT, h=H, thresholds=None):
    pc = F.pick(F.contrasts(), split=split, horizon_y=float(h),
                learner_matched=True)
    mm = F.pick(F.main_metrics(), split=split, horizon_y=float(h))
    obs = F.one(mm, model="cox_naive", metric="obs_cif")
    thresholds = list(thresholds if thresholds is not None else MAIN_THRESHOLDS)

    fig = plt.figure(figsize=(F.W_FULL / F.MM, 104.0 / F.MM))
    gs = fig.add_gridspec(6, 2, width_ratios=[0.86, 1.14], wspace=0.10,
                          hspace=0.35)
    axA = fig.add_subplot(gs[0:2, 0])
    axB = fig.add_subplot(gs[2:4, 0])
    axC = fig.add_subplot(gs[4:6, 0])
    axD = fig.add_subplot(gs[0:4, 1])
    axE = fig.add_subplot(gs[4:6, 1])

    rows = []
    for ax, (metric, title, scale, _u) in zip((axA, axB, axC), CAL_PANELS):
        _forest(ax, pc, metric, title, scale, rows)

    # ---- D: paired net benefit across thresholds -------------------------
    xs = np.arange(len(thresholds), dtype=float)
    tiers = [F.TIER[t] for t in thresholds]
    F.tier_rules(axD, tiers)
    offs = np.linspace(-0.20, 0.20, len(PAIR_STYLE))
    for k, (contrast, label, col, mk) in enumerate(PAIR_STYLE):
        for j, t in enumerate(thresholds):
            r = F.one(pc, contrast=contrast, metric=f"nb_at_{t:g}")
            x = xs[j] + offs[k]
            axD.plot([x, x], [r.lo * 1000, r.hi * 1000], color=col, lw=0.9,
                     solid_capstyle="butt", zorder=3)
            for cap in (r.lo, r.hi):
                axD.plot([x - 0.055, x + 0.055], [cap * 1000] * 2, color=col,
                         lw=0.7, zorder=3)
            axD.plot([x], [r.est * 1000], marker=mk, ms=3.2, color=col,
                     markerfacecolor=col if r.ci_excludes_zero else "white",
                     markeredgecolor=col, markeredgewidth=0.7, zorder=4)
            rows.append(dict(panel_group="net_benefit", metric=f"nb_at_{t:g}",
                             contrast=contrast, label=label, x=t,
                             est=r.est * 1000, lo=r.lo * 1000, hi=r.hi * 1000,
                             ci_excludes_zero=bool(r.ci_excludes_zero),
                             units="net true positives per 1,000 screened"))
    F.zero_line(axD, vertical=False)
    axD.set_xlim(-0.5, len(thresholds) - 0.5)
    lim = 6.3 if 0.20 in thresholds else 1.55
    axD.set_ylim(-lim, lim * 0.24 + 1.05)
    axD.set_xticks(xs)
    axD.set_xticklabels([])
    axD.set_xlabel("")
    axD.set_ylabel("\u0394 net benefit\n(net true positives per 1,000 screened)")
    F.thin_spines(axD)
    spans = [("primary", -0.5, 3.5), ("secondary", 3.5, 4.5)]
    if 0.20 in thresholds:
        spans.append(("supplement\nonly", 4.5, 5.5))
    ytop = axD.get_ylim()[1]
    for name, x0, x1 in spans:
        axD.text((x0 + x1) / 2, ytop * 0.985, name, ha="center", va="top",
                 fontsize=PT, color=F.C_GREY)
    axD.text(0.015, 0.055,
             f"Observed {h}-year cumulative incidence "
             f"{obs.est*1000:.1f} per 1,000",
             transform=axD.transAxes, ha="left", va="bottom", fontsize=PT,
             color=F.C_GREY)

    # symbol key inside the artwork
    for contrast, label, col, mk in PAIR_STYLE:
        axD.plot([], [], marker=mk, ms=3.2, color=col, lw=0.9, label=label)
    axD.plot([], [], marker="s", ms=3.2, color=F.C_GREY, ls="none",
             markerfacecolor="white", markeredgecolor=F.C_GREY,
             label="95% interval contains zero")
    axD.plot([], [], marker="s", ms=3.2, color=F.C_GREY, ls="none",
             label="95% interval excludes zero")
    axD.legend(loc="lower left", fontsize=PT, handlelength=1.6, borderpad=0.1,
               labelspacing=0.26, borderaxespad=0.3,
               bbox_to_anchor=(0.0, 0.09))

    # ---- E: how many people each threshold concerns ----------------------
    F.tier_rules(axE, tiers)
    fr = {}
    for t in thresholds:
        sub = F.pick(mm, metric=f"frac_above_{t:g}")
        fr[t] = sub.set_index("model")["est"]
    for j, t in enumerate(thresholds):
        v = fr[t]
        axE.plot([xs[j], xs[j]], [v.min() * 100, v.max() * 100],
                 color=F.C_LIGHT, lw=0.9, zorder=2, solid_capstyle="butt")
        for cap in (v.min(), v.max()):
            axE.plot([xs[j] - 0.10, xs[j] + 0.10], [cap * 100] * 2,
                     color=F.C_LIGHT, lw=0.7, zorder=2)
        axE.plot([xs[j] - 0.10], [v["cox_naive"] * 100], "o", ms=3.0,
                 color=F.C_NAIVE, markerfacecolor="white",
                 markeredgecolor=F.C_NAIVE, markeredgewidth=0.7, zorder=4)
        axE.plot([xs[j] + 0.10], [v["csc_cox"] * 100], "s", ms=3.0,
                 color=F.C_CR, zorder=4)
        for mdl in v.index:
            rows.append(dict(panel_group="frac_above", metric=f"frac_above_{t:g}",
                             contrast=mdl, label=F.LABEL[mdl], x=t,
                             est=v[mdl] * 100, lo=np.nan, hi=np.nan,
                             ci_excludes_zero=False, units="% of sample"))
    axE.set_xlim(-0.5, len(thresholds) - 0.5)
    axE.set_ylim(0, 78)
    axE.set_xticks(xs)
    axE.set_xticklabels([f"{t*100:g}" for t in thresholds])
    axE.set_xlabel(f"Decision threshold (% predicted {h}-year risk)")
    axE.set_ylabel("Sample above\nthreshold (%)")
    F.thin_spines(axE)
    axE.plot([], [], "o", ms=3.0, color=F.C_NAIVE, markerfacecolor="white",
             markeredgecolor=F.C_NAIVE, ls="none", label="Cox, naive")
    axE.plot([], [], "s", ms=3.0, color=F.C_CR, ls="none",
             label="Cause-specific Cox")
    axE.plot([], [], color=F.C_LIGHT, lw=0.9, label="Range over all 10 models")
    axE.legend(loc="upper right", fontsize=PT, handlelength=1.6, borderpad=0.1,
               labelspacing=0.26, borderaxespad=0.3, ncol=1)

    axB.set_xticks([-0.20, -0.15, -0.10, -0.05, 0.0])
    axB.set_xticklabels(["-0.20", "-0.15", "-0.10", "-0.05", "0.00"])
    F.panel_letters([axA, axB, axC], ["A", "B", "C"], dx=-42.0, dy=2.0)
    F.panel_letters([axD, axE], ["D", "E"], dx=-30.0, dy=2.0)
    return fig, pd.DataFrame(rows).assign(split=split, horizon_y=h)


if __name__ == "__main__":
    fig, data = build()
    F.report(F.save(fig, "fig02_two_halves", data=data, width_mm=F.W_FULL))
