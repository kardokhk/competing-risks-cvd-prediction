"""Alt text for the version-2 figures, and the writer that emits it.

JAMIA requires alt text for every image. Alt text is not the caption: it says
what is visually present for a reader who cannot see the image, namely the panel
layout, the axes and their ranges, the direction and rough magnitude of what is
plotted, and the one thing each panel shows. Numbers appear only where the number
is the visual point.

The text lives here, in one place, so that `figures/v2/alt_text.csv` and the
`**Alt text.**` paragraphs in `figures/v2/captions.md` cannot drift apart. Run
this module to rewrite both.

    python src/v2/figures/alt_text.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "figures" / "v2"

# figure label -> (caption heading prefix, file stem, alt text)
ALT: list[tuple[str, str, str, str]] = [
    ("Figure 1", "## Figure 1.", "fig01_calibration",
     "Three square panels side by side, each plotting predicted against observed "
     "10-year cumulative incidence from 0 to 30% with a dotted diagonal for "
     "perfect calibration. In every panel a dashed orange curve with circles runs "
     "below the diagonal and a solid blue curve with squares runs closer to it, "
     "so the naive model over-predicts in all three. Open markers flag deciles "
     "with too few events. A rug strip under each panel puts most predicted risk "
     "below 10%."),

    ("Figure 2", "## Figure 2.", "fig02_two_halves",
     "Five panels. Three small forest plots down the left show differences in "
     "E/O, calibration slope and calibration index for three model pairs; all "
     "nine intervals sit clear of the vertical zero line, positive for the first "
     "and third, negative for slope. The large panel at right plots the "
     "difference in net benefit at five thresholds: every interval crosses the "
     "horizontal zero line and every marker is open. Below it, the fraction above "
     "threshold falls from about 60% to 20%."),

    ("Figure 3", "## Figure 3.", "fig03_closed_form",
     "Three panels. Top left, five bars: a hatched reference bar at 100%, then "
     "heterogeneity tallest of the four components near 72%, change of estimand "
     "about a third, and two near zero. Right, a tall scatter of competing-event "
     "against event incidence in percent, crossed by five gently sloping contours "
     "labelled 1.05 to 1.60; open circles rise diagonally, with three filled "
     "diamonds low down and one star at 14 by 9%. Bottom left, at three horizons "
     "a square near 30% and a triangle near 100%, joined by a line marked about "
     "3.25 times."),

    ("Figure 4", "## Figure 4.", "fig04_alpha_sweep",
     "Six panels in two rows of three, paired by column. The top row plots three "
     "measures against a loss weight from 0 to 1: on log axes the calibration "
     "index and the expected-over-observed ratio fall steeply then flatten, while "
     "concordance stays flat near 0.83 throughout. The bottom row shows the same "
     "measures as paired differences for two validation designs; the calibration "
     "markers sit right of the vertical zero line, one interval still crossing "
     "it, and the concordance markers straddle zero."),

    ("Figure S1", "## Figure S1.", "figS1_discrimination",
     "Three panels. The first two are forest plots of time-dependent AUC and of "
     "concordance for ten models, every point clustered between 0.80 and 0.85 "
     "with heavily overlapping intervals, and a dashed horizontal rule separating "
     "the four naive models above from the six competing-risk models below. The "
     "third plots paired differences for five contrasts: every interval crosses "
     "the vertical zero line, and the two contrasts that are not learner-matched "
     "are drawn in grey at the bottom with much wider intervals."),

    ("Figure S2", "## Figure S2.", "figS2_two_halves_full",
     "The same five-panel layout as Figure 2 with a sixth decision threshold "
     "added at the right. The three left-hand forest plots are unchanged. In the "
     "net-benefit panel the five lower thresholds again all cross the zero line "
     "with open markers, while at the 20% threshold, set off by a dotted rule and "
     "labelled supplement only, all three markers are filled and sit well below "
     "zero. The panel beneath shows only about 2 to 7% of the sample above that "
     "threshold."),

    ("Figure S3", "## Figure S3.", "figS3_drift",
     "Two panels. Left, expected over observed for two Cox models under two "
     "validation designs at three horizons: the out-of-fold points sit on the "
     "dashed reference line at 1 while the temporal points sit well above it, and "
     "two open markers with dotted intervals at 15 years are flagged as not "
     "estimable. Right, the ratio between the two models rises from about 1.05 at "
     "5 years to about 1.28 at 15, with a horizontal reference line near 1.15."),

    ("Figure S4", "## Figure S4.", "figS4_strata",
     "Six panels in two rows of three, by age group, sex and survey cycle. In the "
     "top row the naive model's marker sits above its competing-risk partner in "
     "every stratum, and both rise across middle age then fall in the oldest "
     "band, all above the dashed line at 1. The bottom row gives cardiovascular "
     "deaths per stratum as open bars with the informative fraction printed "
     "above each; only the youngest age band falls below the dashed rule at 15 "
     "events."),

    ("Figure S5", "## Figure S5.", "figS5_calibration_cv",
     "The same three-panel calibration layout as Figure 1, on the cross-validated "
     "split. The solid blue curves with squares now track the dotted diagonal "
     "closely across the whole range, while the dashed orange curves with circles "
     "sit below it and the gap widens as predicted risk rises. Fewer low deciles "
     "carry the open sparse marker than in Figure 1. Rug strips again place most "
     "predicted risk below 10%."),

    ("Figure S6", "## Figure S6.", "figS6_two_halves_cv",
     "The same five-panel layout as Figure 2 on the cross-validated split, with "
     "all six decision thresholds. The three calibration forest plots sit clear "
     "of the zero line and are visibly tighter than their temporal counterparts. "
     "In the net-benefit panel almost every interval crosses zero; two filled "
     "markers, both the binomial pair, sit below the line at the 7.5% and 20% "
     "thresholds. The fraction above threshold falls from about 60% to under 5%."),

    ("Figure S7", "## Figure S7.", "figS7_smoothers",
     "Two stacked panels sharing six smoother names along the x axis. The upper "
     "panel plots the paired difference in the integrated calibration index: all "
     "six markers sit above the heavy horizontal zero line, four clustered near "
     "6.6 per 1,000, one lower at about 4.7, and the Austin-concordant smoother "
     "at the right lowest of all near 1.8. The lower panel plots each model's own "
     "index, the naive model above its competing-risk partner at every smoother."),

    ("Figure S8", "## Figure S8.", "figS8_instability",
     "Two panels. Left, seven statistics along the x axis, each carrying two "
     "markers that all fall between 93 and 102% against a dashed rule at 100%, so "
     "the seed accounts for nearly the whole of the spread. Right, a square "
     "scatter of the fraction of missing information before ensembling against "
     "after, 0 to 1.1 on both axes: almost every point lies well below the dotted "
     "diagonal, and four circled points remain near the top right corner."),
]


def write_csv() -> Path:
    path = OUT / "alt_text.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "file", "alt_text"])
        for label, _heading, stem, text in ALT:
            w.writerow([label, f"{stem}.pdf", " ".join(text.split())])
    return path


def write_captions() -> Path:
    """Insert or replace an `**Alt text.**` paragraph in each caption section."""
    path = OUT / "captions.md"
    s = path.read_text()
    for label, heading, _stem, text in ALT:
        start = s.index(heading)
        # The final section has no trailing rule, so fall back to end of file.
        end = s.find("\n---\n", start)
        if end == -1:
            end = len(s)
        block = s[start:end]
        block = re.sub(r"\n\*\*Alt text\.\*\*.*?(?=\n\n|\Z)", "", block,
                       flags=re.S)
        para = "\n\n**Alt text.** " + " ".join(text.split()) + "\n"
        s = s[:start] + block.rstrip("\n") + para + s[end:]
    path.write_text(s)
    return path


if __name__ == "__main__":
    for p in (write_csv(), write_captions()):
        print("wrote", p)
    n = [len(" ".join(t.split()).split()) for *_r, t in ALT]
    print(f"{len(ALT)} figures, alt text {min(n)}-{max(n)} words "
          f"(median {sorted(n)[len(n) // 2]})")
