"""Figure S4. Stratified calibration in the large, with the information each
stratum actually carries shown beside it.

Temporal split, 10-year horizon. Top row: expected over observed for the primary
learner-matched pair in each stratum. Bottom row: the number of cardiovascular
deaths by the horizon and the fraction of the stratum still informative at it.
A stratum with fewer than 15 events, or with a low at-risk fraction, is drawn as
an open marker and its label is starred; it is shown rather than dropped so that
the reader can see what was excluded from interpretation and why.

No interval is available for a stratified E/O: the shared-index bootstrap was run
on the whole sample, not within stratum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as F
from figlib import plt

SPLIT, H = "temporal", 10
PT = 6.0
BYS = [("age_group", "Age group (years)"), ("sex_label", "Sex"),
       ("cycle", "NHANES cycle")]
MODELS = [("cox_naive", F.C_NAIVE, "o", -0.14), ("csc_cox", F.C_CR, "s", 0.14)]


def build():
    fig = plt.figure(figsize=(F.W_FULL / F.MM, 82.0 / F.MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 0.75, 0.85],
                          height_ratios=[1.35, 1.0], wspace=0.16, hspace=0.30)
    rows = []
    top = [fig.add_subplot(gs[0, j]) for j in range(3)]
    bot = [fig.add_subplot(gs[1, j]) for j in range(3)]

    for j, (by, title) in enumerate(BYS):
        d = F.strata(SPLIT, H, by).sort_values("stratum").reset_index(drop=True)
        xs = np.arange(len(d), dtype=float)
        ax, axb = top[j], bot[j]
        ax.axhline(1.0, color=F.C_GREY, lw=0.5, ls=(0, (3, 2)), zorder=2)
        for mdl, col, mk, off in MODELS:
            for i, r in d.iterrows():
                ok = bool(r.interpretable)
                ax.plot([xs[i] + off], [r[f"eo_{mdl}"]], marker=mk, ms=3.2,
                        color=col, markerfacecolor=col if ok else "white",
                        markeredgecolor=col, markeredgewidth=0.7, zorder=4)
                rows.append(dict(panel="top", stratifier=by, stratum=r.stratum,
                                 model=mdl, quantity="eo", value=r[f"eo_{mdl}"],
                                 n=r.n, n_cvd_by_horizon=r.n_cvd_by_horizon,
                                 at_risk_fraction=r.at_risk_fraction,
                                 flag_sparse=bool(r.flag_sparse),
                                 flag_low_at_risk=bool(r.flag_low_at_risk),
                                 interpretable=ok))
        labs = [s + ("" if bool(o) else " *")
                for s, o in zip(d.stratum, d.interpretable)]
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=30, ha="right")
        ax.set_xlim(-0.55, len(d) - 0.45)
        ax.set_ylim(0.9, 1.85)
        F.thin_spines(ax)
        if j == 0:
            ax.set_ylabel("Expected / observed")
        else:
            ax.set_yticklabels([])

        axb.bar(xs, d.n_cvd_by_horizon, width=0.60, color="white",
                edgecolor="black", linewidth=0.6, zorder=3)
        axb.axhline(15, color=F.C_GREY, lw=0.5, ls=(0, (3, 2)), zorder=2)
        for i, r in d.iterrows():
            axb.text(xs[i], r.n_cvd_by_horizon + 6,
                     f"{r.at_risk_fraction*100:.0f}%", ha="center", va="bottom",
                     fontsize=PT, color=F.C_CR)
            rows.append(dict(panel="bottom", stratifier=by, stratum=r.stratum,
                             model="", quantity="n_cvd_by_horizon",
                             value=r.n_cvd_by_horizon, n=r.n,
                             n_cvd_by_horizon=r.n_cvd_by_horizon,
                             at_risk_fraction=r.at_risk_fraction,
                             flag_sparse=bool(r.flag_sparse),
                             flag_low_at_risk=bool(r.flag_low_at_risk),
                             interpretable=bool(r.interpretable)))
        axb.set_xticks(xs)
        axb.set_xticklabels(labs, rotation=30, ha="right")
        axb.set_xlim(-0.55, len(d) - 0.45)
        axb.set_ylim(0, 275)
        axb.set_xlabel(title)
        F.thin_spines(axb)
        if j == 0:
            axb.set_ylabel("Cardiovascular\ndeaths by 10 y (n)", labelpad=3)
        else:
            axb.set_yticklabels([])

    top[0].plot([], [], "o", ms=3.2, color=F.C_NAIVE, ls="none",
                label="Cox, naive")
    top[0].plot([], [], "s", ms=3.2, color=F.C_CR, ls="none",
                label="Cause-specific Cox")
    top[0].plot([], [], "o", ms=3.2, color=F.C_GREY, markerfacecolor="white",
                ls="none", label="* not interpretable")
    top[0].legend(loc="upper left", fontsize=PT, handlelength=1.2,
                  borderpad=0.1, labelspacing=0.24, borderaxespad=0.3)
    # Wording must not depend on colour: the figure has to read in greyscale.
    bot[0].text(0.02, 0.93, "percentage above each bar: fraction of the stratum\n"
                            "still informative at 10 y. Dashed rule: 15 events",
                transform=bot[0].transAxes, fontsize=PT, color=F.C_GREY,
                ha="left", va="top", linespacing=1.3)

    F.panel_letters(top, ["A", "B", "C"], dx=-26.0, dy=3.0)
    F.panel_letters(bot, ["D", "E", "F"], dx=-26.0, dy=6.0)
    return fig, pd.DataFrame(rows).assign(split=SPLIT, horizon_y=H)


if __name__ == "__main__":
    fig, data = build()
    F.report(F.save(fig, "figS4_strata", data=data, width_mm=F.W_FULL))
