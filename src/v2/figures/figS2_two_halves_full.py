"""Figure S2. Figure 2 with the supplement-only 20% threshold restored.

Same panels, same estimators, same split (temporal, 10 years) and the same
marker encoding as Figure 2, with the 20% decision threshold added back at the
same scale as everything else. The main-text figure stops at 7.5% because the
20% contrasts are five to twenty times larger and compress the thresholds that
carry the argument; this figure is where the 20% evidence lives, which is where
the manuscript already puts it.

Three of the eighteen learner-matched net-benefit contrasts exclude zero, and
all three are at 20%: a threshold defined on incident atherosclerotic disease,
not on the cardiovascular-mortality endpoint used here, above which 2.2 to 6.9%
of the sample sits and at which the treat-all reference has a net benefit of
-0.209.
"""
from __future__ import annotations

import figlib as F
import fig02_two_halves as f2

SPLIT, H = "temporal", 10


def build():
    return f2.build(split=SPLIT, h=H, thresholds=F.THRESHOLDS)


if __name__ == "__main__":
    fig, data = build()
    F.report(F.save(fig, "figS2_two_halves_full", data=data, width_mm=F.W_FULL))
