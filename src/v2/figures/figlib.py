"""Shared style, palette and IO for the version-2 publication figures.

Target venue: JAMIA, Research and Applications (see docs/venue_shortlist.md).
JAMIA wants figures as
separate files cited in numerical order and mandates alt text for every image;
the alt text lives in figures/v2/captions.md and figures/v2/alt_text.csv.

Neither journal states a figure width, so the house default applies: 183 mm full
width, 89 mm single column, 170 mm maximum height, 7 pt sans-serif body text,
8 pt bold panel letters, nothing below 0.5 pt, RGB. The 800 dpi TIFF and the
no-tints and key-inside-the-artwork rules were adopted for Statistics in Medicine
and are kept: they cost nothing, they are above what JAMIA asks, and re-cutting
them would be churn. Every figure is written as a PDF master, a 600 dpi PNG for
inspection and an 800 dpi flattened LZW TIFF for submission.

Every figure regenerates from the CSVs under results/ with no manual step. The
seed below is fixed only so that any jitter is reproducible; nothing here
resamples.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260904
np.random.seed(SEED)

# Repo root is two levels above src/v2/figures/ .
ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
MET = RESULTS / "metrics_v2"
CF = RESULTS / "closed_form"
DEEP = RESULTS / "deep"
OUT = ROOT / "figures" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

# figstyle.py / figqa.py / fontguard.py live in the user's academic assets.
_ASSETS = Path.home() / ".claude" / "academic" / "assets"
if str(_ASSETS) not in sys.path:
    sys.path.insert(0, str(_ASSETS))

os.environ.setdefault("XDG_CACHE_HOME", "${CACHE_DIR}")
os.environ.setdefault("MPLCONFIGDIR", "${CACHE_DIR}/matplotlib")

import figstyle  # noqa: E402
import figqa  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------

FAMILY = figstyle.use_print_style()          # raises rather than reaching DejaVu
MM = figstyle.MM_PER_IN
W_FULL, W_HALF, W_ONEHALF = 183.0, 89.0, 136.0

# Okabe-Ito. The naive/competing distinction is carried by colour AND by marker
# shape AND by line style, so the figures survive greyscale printing.
C_NAIVE = "#D55E00"        # vermillion
C_CR = "#0072B2"           # blue
C_GREY = "#4D4D4D"
C_LIGHT = "#8C8C8C"
# Darkened variants used for TEXT only. #D55E00 gives 3.9:1 against white, below
# the 4.5:1 contrast floor; the line and marker colours are unchanged because the
# contrast requirement applies to text.
C_NAIVE_TXT = "#A64000"
C_CR_TXT = "#005B8F"
C_ACC = "#009E73"          # bluish green, third series where one is needed
C_ACC2 = "#CC79A7"         # reddish purple
C_ACC3 = "#E69F00"         # orange

OKABE = figstyle.OKABE_ITO

FAM_STYLE = {
    "naive": dict(color=C_NAIVE, marker="o", ls="--"),
    "competing": dict(color=C_CR, marker="s", ls="-"),
}

# Learner-matched pairs, in the order the manuscript reports them.
PAIRS = [
    ("cox_naive", "csc_cox", "Cox", "primary"),
    ("rsf_naive", "rsf_cr", "Random survival forest", "secondary"),
    ("logistic", "logistic_cr", "IPCW binomial", "secondary"),
]
PAIR_CONTRAST = {
    "cox_naive - csc_cox": "Cox",
    "rsf_naive - rsf_cr": "Random survival forest",
    "logistic - logistic_cr": "IPCW binomial",
    "cox_naive - fine_gray": "Cox naive vs Fine-Gray",
    "gbs_naive - deephit": "Gradient boosting vs DeepHit",
}

LABEL = {
    "cox_naive": "Cox, naive",
    "csc_cox": "Cause-specific Cox",
    "rsf_naive": "RSF, naive",
    "rsf_cr": "RSF, competing",
    "logistic": "Binomial, naive",
    "logistic_cr": "Binomial, competing",
    "fine_gray": "Fine-Gray",
    "gbs_naive": "Gradient boosting, naive",
    "deephit": "DeepHit",
    "nfg": "Neural Fine-Gray",
}
FAMILY_OF = {
    "cox_naive": "naive", "rsf_naive": "naive", "logistic": "naive",
    "gbs_naive": "naive",
    "csc_cox": "competing", "rsf_cr": "competing", "logistic_cr": "competing",
    "fine_gray": "competing", "deephit": "competing", "nfg": "competing",
}

# Decision thresholds and the tier each one belongs to. 7.5% and 20% are defined
# on incident ASCVD; the endpoint here is cardiovascular mortality with a 10-year
# cumulative incidence near 3.3%, so those two thresholds interrogate a nearly
# empty region and are demoted.
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.20]
TIER = {0.01: "primary", 0.02: "primary", 0.03: "primary", 0.05: "primary",
        0.075: "secondary", 0.20: "supplement"}
TIER_COLOR = {"primary": "#FFFFFF", "secondary": "#F2F2F2",
              "supplement": "#E3E3E3"}


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------

def main_metrics() -> pd.DataFrame:
    return pd.read_csv(MET / "main_metrics.csv")


def contrasts() -> pd.DataFrame:
    return pd.read_csv(MET / "paired_contrasts.csv")


def concordant_metrics() -> pd.DataFrame:
    """Step 11b/12 output: the Austin-concordant smoother columns.

    These live in their own file as well as being merged into main_metrics.csv
    by src/v2/12_merge_concordant.py. Read them from here, because a re-run of
    regenerating main_metrics.csv without rerunning the merge silently drops
    them, which has happened. This file is the source of record.
    """
    return pd.read_csv(MET / "concordant_metrics.csv")


def concordant_contrasts() -> pd.DataFrame:
    """Step 11b/12 output: paired contrasts on the concordant smoother columns."""
    return pd.read_csv(MET / "concordant_contrasts.csv")


def deciles(split: str, h: int) -> pd.DataFrame:
    return pd.read_csv(MET / "calibration" / f"deciles__{split}__h{h}.csv")


def smooth(split: str, h: int) -> pd.DataFrame:
    return pd.read_csv(MET / "calibration" / f"smooth__{split}__h{h}.csv")


def dca_curves(split: str, h: int) -> pd.DataFrame:
    return pd.read_csv(MET / "dca" / f"curves__{split}__h{h}.csv")


def dca_thresholds(split: str, h: int) -> pd.DataFrame:
    return pd.read_csv(MET / "dca" / f"thresholds__{split}__h{h}.csv")


def strata(split: str, h: int, by: str) -> pd.DataFrame:
    return pd.read_csv(MET / "strata" / f"{split}__h{h}__{by}.csv")


def pick(df: pd.DataFrame, **kw) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k].isin(v) if isinstance(v, (list, tuple, set)) else df[k] == v
    return df.loc[m]


def one(df: pd.DataFrame, **kw) -> pd.Series:
    r = pick(df, **kw)
    if len(r) != 1:
        raise ValueError(f"expected 1 row for {kw}, got {len(r)}")
    return r.iloc[0]


# ----------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------

def panel_letters(axes, letters=None, *, dx=-30.0, dy=3.0, size=8.0):
    """Bold panel letters placed a fixed number of POINTS from the axes corner.

    Offsetting in points rather than in axes fractions keeps the letters the same
    distance from the axis in every panel, whatever the panel's width, and stops
    them colliding with a two-line y label.
    """
    letters = letters or [chr(65 + i) for i in range(len(axes))]
    for ax, letter in zip(axes, letters):
        ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=size, fontweight="bold", va="bottom", ha="left",
                    annotation_clip=False)


def zero_line(ax, *, vertical=True, **kw):
    """A clearly marked zero reference, heavier than the data rules."""
    style = dict(color="black", lw=0.8, ls="-", zorder=1.5)
    style.update(kw)
    (ax.axvline if vertical else ax.axhline)(0.0, **style)


def tier_rules(ax, tiers):
    """Separate the threshold tiers with a dotted rule.

    A rule and a label rather than a shaded band, because the previous target,
    Statistics in Medicine, did not accept tints. Kept under JAMIA: a rule reads
    at least as well and survives greyscale printing.
    """
    for i in range(1, len(tiers)):
        if tiers[i] != tiers[i - 1]:
            ax.axvline(i - 0.5, color=C_GREY, lw=0.5, ls=(0, (1, 2)), zorder=0)


def thin_spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------
# save + QA
# ----------------------------------------------------------------------------

def save(fig, stem: str, *, data: pd.DataFrame | None = None,
         width_mm: float | None = None, min_pt: float = 6.0) -> dict:
    """Write the PDF master, a 600 dpi PNG, an 800 dpi flattened LZW TIFF, the
    plotted data, and the two derived images the visual pass needs.

    Returns the figqa report so the caller can print it.
    """
    import logging

    from PIL import Image

    stem_p = OUT / stem

    # matplotlib reports a font substitution through this logger, not through
    # the warnings module, so a warnings filter never sees it. Attach the
    # handler before the first draw.
    subs: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, rec):
            subs.append(rec.getMessage())

    lg = logging.getLogger("matplotlib.font_manager")
    h = _Catch()
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)

    problems = figqa.report(fig, min_pt=min_pt)
    fig.savefig(stem_p.with_suffix(".pdf"))
    fig.savefig(stem_p.with_suffix(".png"), dpi=600)
    # Stat Med asks for 800 dpi line art; matplotlib's TIFF writer emits RGBA,
    # which Wiley's checker rejects, so flatten onto white and use LZW.
    tmp = stem_p.parent / f".{stem}_800.png"
    fig.savefig(tmp, dpi=800)
    im = Image.open(tmp)
    Image.alpha_composite(Image.new("RGBA", im.size, "white"),
                          im.convert("RGBA")).convert("RGB").save(
        stem_p.with_suffix(".tif"), compression="tiff_lzw", dpi=(800, 800))
    im.close()
    tmp.unlink()
    lg.removeHandler(h)
    for m in dict.fromkeys(x for x in subs
                           if "falling back" in x.lower()
                           or "not found" in x.lower()):
        problems.insert(0, f"FONT SUBSTITUTION: {m}")

    if data is not None:
        data.to_csv(OUT / f"{stem}_data.csv", index=False)

    w_mm = width_mm or fig.get_size_inches()[0] * MM
    figqa.greyscale_and_downscale(str(stem_p.with_suffix(".png")), w_mm)
    return {"stem": stem, "problems": problems}


def report(res: dict) -> None:
    print(f"\n=== {res['stem']}")
    for p in res["problems"]:
        print("   ", p)
