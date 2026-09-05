"""Figure 1. Flexible calibration for the three learner-matched pairs.

Temporal split, 10-year horizon. Predicted 10-year cumulative incidence of
cardiovascular death against observed. Each panel holds one learner: the naive
member censors competing death, the competing member models it, and nothing else
differs between them.

Sparse deciles (fewer than 15 cause-1 events, `flag_sparse` in the source file)
are drawn as open markers with dotted intervals so that a reader cannot mistake
them for solid points. They are shown, not dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figlib as F
from figlib import plt

SPLIT, H = "temporal", 10
AXMAX = 0.32          # covers every decile point and the curve to its 98th pct
TRUNC_BIN = 97        # 0-based; keeps percentile bins 1..98 of the smooth curve
PT = 6.0


def _pair_panel(ax, rug, dec, smo, met, naive, cr, learner, letter, rows):
    ax.plot([0, AXMAX], [0, AXMAX], color=F.C_GREY, lw=0.5, ls=":", zorder=1)

    for mdl in (naive, cr):
        fam = F.FAMILY_OF[mdl]
        st = F.FAM_STYLE[fam]

        g = smo[smo.model == mdl].sort_values("pred_bin")
        gk = g[g.pred_bin <= TRUNC_BIN]
        ax.plot(gk.pred, gk.obs, color=st["color"], ls=st["ls"], lw=1.0,
                zorder=3, solid_capstyle="round")
        rows.append(pd.DataFrame(dict(
            panel=letter, layer="smooth", model=mdl, family=fam,
            x=gk.pred, y=gk.obs, lo=np.nan, hi=np.nan, flag_sparse=False,
            n=np.nan, n_events=np.nan, at_risk_fraction=np.nan)))

        d = dec[dec.model == mdl].sort_values("decile")
        for _, r in d.iterrows():
            ls = (0, (1, 1)) if r.flag_sparse else "-"
            ax.plot([r.mean_pred, r.mean_pred], [r.obs_lo, r.obs_hi],
                    color=st["color"], lw=0.5, ls=ls, zorder=3.5)
        for sparse in (False, True):
            dd = d[d.flag_sparse == sparse]
            if dd.empty:
                continue
            ax.plot(dd.mean_pred, dd.obs_cif, st["marker"], ms=2.6,
                    markerfacecolor="white" if sparse else st["color"],
                    markeredgecolor=st["color"], markeredgewidth=0.6,
                    linestyle="none", zorder=4, clip_on=False)
        rows.append(pd.DataFrame(dict(
            panel=letter, layer="decile", model=mdl, family=fam,
            x=d.mean_pred, y=d.obs_cif, lo=d.obs_lo, hi=d.obs_hi,
            flag_sparse=d.flag_sparse, n=d.n, n_events=d.n_events,
            at_risk_fraction=d.at_risk_fraction)))

        # Quantile rug. The smooth grid is percentile based, so its x values are
        # the percentiles of predicted risk for that model.
        y0 = 0.55 if fam == "naive" else 0.05
        rug.vlines(g.pred, y0, y0 + 0.40, color=st["color"], lw=0.4)

    for k, (mdl, tag) in enumerate([(naive, "Naive"), (cr, "Competing")]):
        eo = F.one(met, model=mdl, metric="eo")
        ic = F.one(met, model=mdl, metric="ici")
        ax.text(0.985, 0.040 + (1 - k) * 0.150,
                f"{tag}  E/O {eo.est:.2f} ({eo.lo:.2f}-{eo.hi:.2f})\n"
                f"ICI {ic.est:.4f} ({ic.lo:.4f}-{ic.hi:.4f})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=PT,
                linespacing=1.35,
                color=F.C_NAIVE_TXT if F.FAMILY_OF[mdl] == "naive"
                else F.C_CR_TXT)

    ax.set_xlim(0, AXMAX)
    ax.set_ylim(0, AXMAX)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xticklabels([])
    ax.set_yticklabels(["0", "10", "20", "30"])
    ax.set_title(learner, fontsize=7.0, pad=2.5)
    ax.set_anchor("S")          # square axes hug the rug strip below them
    F.thin_spines(ax)

    rug.set_xlim(0, AXMAX)
    rug.set_ylim(0, 1)
    rug.set_yticks([])
    rug.set_xticks([0, 0.1, 0.2, 0.3])
    rug.set_xticklabels(["0", "10", "20", "30"])
    rug.set_anchor("N")
    rug.set_xlabel("Predicted risk (%)")
    for s in ("top", "right", "left"):
        rug.spines[s].set_visible(False)


def build(split=SPLIT, h=H):
    dec = F.deciles(split, h)
    smo = F.smooth(split, h)
    met = F.pick(F.main_metrics(), split=split, horizon_y=float(h))

    fig = plt.figure(figsize=(F.W_FULL / F.MM, 76.0 / F.MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[6.0, 1.0], hspace=0.02,
                          wspace=0.14)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    rugs = [fig.add_subplot(gs[1, j]) for j in range(3)]

    rows = []
    for j, (naive, cr, learner, _tier) in enumerate(F.PAIRS):
        _pair_panel(axes[j], rugs[j], dec, smo, met, naive, cr, learner,
                    chr(65 + j), rows)
        if j:
            axes[j].tick_params(labelleft=False)
            axes[j].set_yticklabels([])

    axes[0].set_ylabel(f"Observed {h}-year cumulative\nincidence (%)")

    # Symbol key inside the artwork, in
    # the region above the identity line, which no series occupies.
    ax = axes[0]
    ax.plot([], [], color=F.C_NAIVE, ls="--", lw=1.0, marker="o", ms=2.6,
            label="Naive, competing death censored")
    ax.plot([], [], color=F.C_CR, ls="-", lw=1.0, marker="s", ms=2.6,
            label="Competing risk, death modelled")
    ax.plot([], [], color=F.C_GREY, ls="none", marker="o", ms=2.6,
            markerfacecolor="white", markeredgecolor=F.C_GREY,
            label="Decile with < 15 events")
    ax.plot([], [], color=F.C_GREY, ls=":", lw=0.5, label="Perfect calibration")
    ax.legend(loc="upper left", fontsize=PT, handlelength=1.7, borderpad=0.1,
              labelspacing=0.26, borderaxespad=0.25)

    F.panel_letters(axes, dx=-26.0, dy=3.0)
    data = pd.concat(rows, ignore_index=True)
    data["split"] = split
    data["horizon_y"] = h
    return fig, data


if __name__ == "__main__":
    fig, data = build()
    F.report(F.save(fig, "fig01_calibration", data=data, width_mm=F.W_FULL))
