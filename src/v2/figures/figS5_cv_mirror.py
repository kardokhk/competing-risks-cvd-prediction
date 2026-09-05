"""Figures S5 and S6. The cross-validated split, mirroring Figures 1 and 2.

Figure S6 keeps all six decision thresholds, including the supplement-only 20%,
which the main-text Figure 2 omits; see figS2_two_halves_full.py for the
temporal-split equivalent.

Same panels, same estimators, same scales, on the repeated out-of-fold split of
the whole 1999-2018 cohort instead of the temporal test set. The calibration
contrasts are larger and tighter here, and no net-benefit contrast excludes zero
at any threshold.

Both figures are produced by calling the Figure 1 and Figure 2 builders with a
different split, so the two versions cannot drift apart.
"""
from __future__ import annotations

import figlib as F
import fig01_calibration as f1
import fig02_two_halves as f2

SPLIT, H = "cv", 10


def build_calibration():
    return f1.build(split=SPLIT, h=H)


def build_two_halves():
    # The supplement keeps all six thresholds, including the supplement-only
    # 20%, which the main-text Figure 2 omits.
    return f2.build(split=SPLIT, h=H, thresholds=F.THRESHOLDS)


if __name__ == "__main__":
    fig, data = build_calibration()
    F.report(F.save(fig, "figS5_calibration_cv", data=data,
                    width_mm=F.W_FULL))
    fig, data = build_two_halves()
    F.report(F.save(fig, "figS6_two_halves_cv", data=data, width_mm=F.W_FULL))
