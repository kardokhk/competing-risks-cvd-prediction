## 0. Verdict on the objective-versus-architecture claim

**The mechanism is real and the claim as previously stated is too strong.** DeepHit's
calibration is governed by one hyperparameter, the weight `alpha` that its loss puts on the
likelihood term relative to the ranking term. Sweeping that weight moves calibration by an
order of magnitude and leaves discrimination unchanged. But because `alpha` is a
hyperparameter and not a fixed property of the model, "DeepHit is a competing-risk
architecture that miscalibrates" is not what the evidence supports. What it supports is
"DeepHit's default configuration miscalibrates, and the default is a ranking-dominated
objective."

Three findings, in the order a reader needs them.

### The alpha sweep shows the mechanism

Across `alpha` from 0 (pure ranking) to 1 (pure likelihood), with the architecture,
optimiser and early-stopping rule held fixed and 15 replicates per setting, the median ICI
fell from 0.158 to 0.0036 and the median E/O from 4.80 to 1.07, monotonically (Spearman
correlation of ICI against alpha -0.97, of E/O against alpha -0.97). Discrimination did not
move: the median C-index was 0.831 to 0.834 at every alpha from 0.05 upward, and the pure
ranking loss was the worst discriminator at 0.826. Moving from the prior work's
alpha = 0.2 to alpha = 1 changed ICI by -0.0156 (95% CI -0.0182 to -0.0133) and the C-index
by -0.0001 (95% CI -0.0033 to +0.0010). The same pattern held on the temporal test set,
where nothing was selected: ICI -0.0149 (95% CI -0.0177 to -0.0114) and C-index +0.0001
(95% CI -0.0033 to +0.0019).

An internal check confirms the sweep isolated what it was meant to isolate. At alpha = 1
the four values of `sigma`, which enters only the ranking term, gave identical results to
four decimal places, as they must when that term carries zero weight.

### A properly tuned DeepHit closes most of the gap, and all of it on the tuning protocol

After a search over 80 configurations selected on a proper scoring rule computed on
training rows only, the selected DeepHit used alpha = 0.9. Refitted at 12 seeds:

- On the inner out-of-fold protocol the gap closed. Tuned DeepHit minus Neural Fine-Gray
  gave an ICI difference of +0.0007 (95% CI -0.0004 to +0.0025), an interval containing
  zero, against +0.0168 (95% CI +0.0154 to +0.0179) for the default DeepHit.
- On the temporal test set the gap narrowed by about three quarters but did not close.
  Default DeepHit minus Neural Fine-Gray gave an ICI difference of +0.0177 (95% CI +0.0152
  to +0.0226); tuning reduced that to +0.0048 (95% CI +0.0030 to +0.0069), an interval that
  still excludes zero. Tuning changed DeepHit's own ICI by -0.0129 (95% CI -0.0176 to
  -0.0111) at a C-index change of +0.0011 (95% CI -0.0015 to +0.0044).

So the honest headline is that DeepHit's default configuration miscalibrates, that the
cause is the ranking-dominated objective, and that a residual out-of-period gap of about
0.005 in ICI survives tuning.

### The durable contrast is robustness, not calibration

Neural Fine-Gray's temporal-test ICI was 0.0081, 0.0090 and 0.0100 under its default,
Brier-selected and C-index-selected configurations, a factor of 1.2 across the three.
DeepHit's was 0.0258, 0.0129 and 0.0307, a factor of 2.4, with the architecture identical
in all three. Parameterising a monotone cumulative incidence function did not make Neural
Fine-Gray better calibrated than a well-tuned DeepHit by much; it made it insensitive to
how it was configured. That is the property a reader should take away.

### A warning about benchmark practice

Selecting DeepHit on the C-index, which is what a discrimination-led benchmark does, picked
alpha = 0.2 and produced a temporal-test ICI of 0.0307, worse than not tuning at all
(difference against the default +0.0049, 95% CI +0.0002 to +0.0069) in exchange for a
C-index gain of +0.0018 (95% CI +0.0001 to +0.0050). A benchmark that tunes deep models for
discrimination and then reports their calibration will manufacture the result the prior work
reported, from a model that is capable of calibrating well.

### What is not established

The miscalibration is not seed noise: at 12 seeds the 10th-to-90th percentile ranges are
tight and do not approach overlap between default DeepHit and Neural Fine-Gray, so the prior
work's single fit was a systematic property of that configuration rather than a fluke. The
search was run on one imputation, so between-imputation variability in which configuration was chosen is unquantified. The tuned DeepHit's inner out-of-fold ICI of 0.0026 reported
in `tuning_selected.json` is optimistically biased by being the minimum over 80
configurations; the same configuration refitted at 12 fresh seeds gave 0.0042, which is the
number to quote. Neural Fine-Gray received the larger gradient budget under a shared
200-epoch cap because DeepHit early-stops well inside it; that asymmetry favours Neural
Fine-Gray and was not corrected.
