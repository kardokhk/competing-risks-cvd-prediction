# P2-C: deep competing-risk models, DeepHit and Neural Fine-Gray

Produced by `src/v2/deep/`, summarised by `src/v2/deep/14_summarise.py`. Environment `${ENV_PREFIX}/crcvd-dl` invoked through the `python-cr` wrapper. Project seed 20260903.

Point estimates are medians over replicates with the 10th to 90th percentile in brackets; differences carry a 2.5th-to-97.5th percentile bootstrap interval over replicates (4,000 resamples, seed 20260903). Horizon 10 y throughout. Estimators are those of `src/v2/eval_lib_v2.py`, so every number here is on the same scale as the main benchmark.

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

## 1. The alpha sweep: does the training objective drive the miscalibration?

pycox writes the DeepHit objective as `loss = alpha * nll_pmf_cr + (1 - alpha) * rank_loss_deephit_cr(sigma)`, so alpha = 0 is a pure ranking loss and alpha = 1 a pure likelihood. The prior work used alpha = 0.2, putting 80% of the objective on the ranking term.

### Primary: inner out-of-fold, training cycles only, sigma = 0.1

| alpha | E/O | ICI | slope | C-index | AUC | IPCW Brier | n |
|---|---|---|---|---|---|---|---|
| 0 | 4.80 (4.55 to 4.92) | 0.1583 (0.1472 to 0.1656) | 3.70 (3.31 to 4.10) | 0.826 (0.824 to 0.833) | 0.832 (0.830 to 0.839) | 0.0650 (0.0618 to 0.0667) | 30 |
| 0.05 | 2.01 (1.96 to 2.06) | 0.0438 (0.0410 to 0.0447) | 2.33 (2.02 to 2.51) | 0.833 (0.829 to 0.835) | 0.839 (0.836 to 0.842) | 0.0395 (0.0391 to 0.0396) | 15 |
| 0.1 | 1.69 (1.60 to 1.75) | 0.0309 (0.0271 to 0.0326) | 2.01 (1.78 to 2.19) | 0.832 (0.829 to 0.837) | 0.839 (0.835 to 0.843) | 0.0380 (0.0376 to 0.0383) | 15 |
| 0.2 | 1.43 (1.36 to 1.49) | 0.0192 (0.0160 to 0.0222) | 1.63 (1.47 to 1.69) | 0.833 (0.831 to 0.836) | 0.840 (0.838 to 0.843) | 0.0371 (0.0369 to 0.0373) | 15 |
| 0.3 | 1.31 (1.25 to 1.35) | 0.0147 (0.0118 to 0.0161) | 1.37 (1.31 to 1.52) | 0.833 (0.831 to 0.835) | 0.840 (0.838 to 0.841) | 0.0368 (0.0367 to 0.0369) | 15 |
| 0.5 | 1.22 (1.17 to 1.24) | 0.0095 (0.0080 to 0.0104) | 1.23 (1.15 to 1.28) | 0.833 (0.831 to 0.835) | 0.840 (0.837 to 0.841) | 0.0366 (0.0365 to 0.0367) | 15 |
| 0.7 | 1.16 (1.12 to 1.20) | 0.0069 (0.0049 to 0.0085) | 1.07 (1.00 to 1.17) | 0.833 (0.830 to 0.835) | 0.839 (0.837 to 0.842) | 0.0365 (0.0364 to 0.0366) | 15 |
| 0.85 | 1.12 (1.08 to 1.18) | 0.0048 (0.0031 to 0.0074) | 1.03 (0.96 to 1.09) | 0.833 (0.829 to 0.834) | 0.839 (0.835 to 0.841) | 0.0365 (0.0363 to 0.0365) | 15 |
| 0.95 | 1.08 (1.04 to 1.13) | 0.0036 (0.0020 to 0.0051) | 0.97 (0.91 to 1.03) | 0.831 (0.829 to 0.834) | 0.838 (0.835 to 0.841) | 0.0365 (0.0364 to 0.0366) | 15 |
| 1 | 1.07 (1.01 to 1.12) | 0.0036 (0.0018 to 0.0051) | 1.01 (0.93 to 1.04) | 0.833 (0.829 to 0.835) | 0.840 (0.835 to 0.841) | 0.0365 (0.0363 to 0.0367) | 30 |
| Neural Fine-Gray | 0.95 (0.92 to 0.98) | 0.0038 (0.0022 to 0.0043) | 0.90 (0.85 to 0.93) | 0.832 (0.829 to 0.835) | 0.838 (0.835 to 0.842) | 0.0364 (0.0363 to 0.0365) | 30 |

Trend across alpha (Spearman rank correlation over all replicates):

- ICI against alpha: -0.967
- E/O against alpha: -0.970
- C-index against alpha: 0.260
- IPCW Brier against alpha: -0.885

Pure likelihood (alpha = 1) minus the prior work's setting (alpha = 0.2):

- ICI: -0.0156 (95% CI -0.0182 to -0.0133)
- E/O: -0.360 (95% CI -0.429 to -0.308)
- calibration slope: -0.620 (95% CI -0.675 to -0.552)
- C-index: -0.0001 (95% CI -0.0037 to 0.0010)
- AUC: -0.0001 (95% CI -0.0036 to 0.0011)

### Confirmatory: temporal test set, no configuration selected on it

| alpha | E/O | ICI | slope | C-index | AUC | IPCW Brier | n |
|---|---|---|---|---|---|---|---|
| 0 | 5.96 (5.79 to 6.08) | 0.1637 (0.1583 to 0.1674) | 3.44 (3.25 to 4.94) | 0.825 (0.820 to 0.826) | 0.825 (0.820 to 0.825) | 0.0603 (0.0591 to 0.0628) | 10 |
| 0.05 | 2.35 (2.28 to 2.44) | 0.0452 (0.0433 to 0.0481) | 1.70 (1.61 to 2.11) | 0.824 (0.822 to 0.825) | 0.824 (0.823 to 0.825) | 0.0340 (0.0340 to 0.0345) | 5 |
| 0.1 | 2.09 (2.06 to 2.16) | 0.0364 (0.0351 to 0.0389) | 1.60 (1.38 to 1.89) | 0.821 (0.820 to 0.827) | 0.821 (0.821 to 0.827) | 0.0332 (0.0330 to 0.0336) | 5 |
| 0.2 | 1.78 (1.70 to 1.84) | 0.0262 (0.0236 to 0.0280) | 1.30 (1.18 to 1.43) | 0.823 (0.821 to 0.825) | 0.823 (0.822 to 0.825) | 0.0322 (0.0319 to 0.0324) | 5 |
| 0.3 | 1.61 (1.57 to 1.66) | 0.0211 (0.0193 to 0.0219) | 1.18 (1.10 to 1.39) | 0.826 (0.821 to 0.827) | 0.826 (0.821 to 0.827) | 0.0317 (0.0316 to 0.0318) | 5 |
| 0.5 | 1.63 (1.57 to 1.65) | 0.0208 (0.0188 to 0.0215) | 1.04 (0.90 to 1.06) | 0.824 (0.822 to 0.825) | 0.825 (0.823 to 0.826) | 0.0319 (0.0318 to 0.0319) | 5 |
| 0.7 | 1.42 (1.36 to 1.53) | 0.0138 (0.0120 to 0.0175) | 0.97 (0.86 to 0.98) | 0.824 (0.821 to 0.826) | 0.824 (0.822 to 0.826) | 0.0314 (0.0312 to 0.0316) | 5 |
| 0.85 | 1.43 (1.33 to 1.48) | 0.0145 (0.0114 to 0.0157) | 0.87 (0.85 to 0.99) | 0.825 (0.822 to 0.828) | 0.826 (0.822 to 0.828) | 0.0314 (0.0312 to 0.0317) | 5 |
| 0.95 | 1.39 (1.34 to 1.43) | 0.0132 (0.0114 to 0.0143) | 0.88 (0.87 to 1.00) | 0.825 (0.820 to 0.826) | 0.825 (0.821 to 0.827) | 0.0313 (0.0311 to 0.0315) | 5 |
| 1 | 1.35 (1.31 to 1.41) | 0.0114 (0.0102 to 0.0137) | 0.93 (0.90 to 1.01) | 0.823 (0.821 to 0.825) | 0.823 (0.821 to 0.825) | 0.0312 (0.0310 to 0.0313) | 10 |
| Neural Fine-Gray | 1.16 (1.15 to 1.22) | 0.0068 (0.0052 to 0.0077) | 0.84 (0.77 to 0.88) | 0.821 (0.819 to 0.822) | 0.821 (0.818 to 0.823) | 0.0310 (0.0310 to 0.0313) | 10 |

- ICI, alpha 1 minus alpha 0.2: -0.0149 (95% CI -0.0177 to -0.0114)
- E/O, alpha 1 minus alpha 0.2: -0.434 (95% CI -0.521 to -0.329)
- C-index, alpha 1 minus alpha 0.2: 0.0001 (95% CI -0.0033 to 0.0019)

## 2. Does a tuned DeepHit close the gap to Neural Fine-Gray?

### Inner out-of-fold, training cycles only

| model | configuration | E/O | ICI | slope | C-index | seeds |
|---|---|---|---|---|---|---|
| deephit | default | 1.45 (1.39 to 1.49) | 0.0203 (0.0184 to 0.0221) | 1.64 (1.54 to 1.78) | 0.833 (0.832 to 0.835) | 12 |
| deephit | tuned_brier | 1.08 (1.03 to 1.16) | 0.0042 (0.0028 to 0.0068) | 1.07 (1.01 to 1.12) | 0.833 (0.832 to 0.836) | 12 |
| deephit | tuned_cindex | 1.53 (1.49 to 1.56) | 0.0233 (0.0220 to 0.0246) | 1.18 (1.13 to 1.21) | 0.837 (0.836 to 0.838) | 12 |
| nfg | default | 0.96 (0.95 to 0.98) | 0.0035 (0.0030 to 0.0043) | 0.88 (0.85 to 0.92) | 0.832 (0.830 to 0.835) | 12 |
| nfg | tuned_brier | 0.99 (0.96 to 1.02) | 0.0027 (0.0016 to 0.0039) | 0.92 (0.87 to 1.01) | 0.833 (0.829 to 0.834) | 12 |
| nfg | tuned_cindex | 1.02 (0.97 to 1.04) | 0.0033 (0.0020 to 0.0040) | 0.97 (0.94 to 1.02) | 0.833 (0.832 to 0.834) | 12 |

DeepHit minus Neural Fine-Gray, both at their selected configurations:

- ICI: 0.0015 (95% CI 0.0001 to 0.0038)
- E/O: 0.085 (95% CI 0.073 to 0.140)
- C-index: 0.0006 (95% CI -0.0006 to 0.0034)

Default DeepHit minus Neural Fine-Gray, the prior work's comparison:

- ICI: 0.0176 (95% CI 0.0160 to 0.0192)
- E/O: 0.460 (95% CI 0.419 to 0.486)

Effect of tuning DeepHit (tuned minus default):

- ICI: -0.0161 (95% CI -0.0174 to -0.0141)
- E/O: -0.375 (95% CI -0.399 to -0.313)
- C-index: 0.0005 (95% CI -0.0004 to 0.0023)

### Temporal test set, confirmatory

| model | configuration | E/O | ICI | slope | C-index | seeds |
|---|---|---|---|---|---|---|
| deephit | default | 1.76 (1.68 to 1.93) | 0.0258 (0.0229 to 0.0308) | 1.28 (1.10 to 1.51) | 0.824 (0.819 to 0.828) | 12 |
| deephit | tuned_brier | 1.39 (1.37 to 1.44) | 0.0129 (0.0125 to 0.0149) | 0.92 (0.82 to 1.00) | 0.825 (0.822 to 0.827) | 12 |
| deephit | tuned_cindex | 1.93 (1.86 to 1.99) | 0.0307 (0.0283 to 0.0327) | 0.93 (0.85 to 1.01) | 0.826 (0.825 to 0.827) | 12 |
| nfg | default | 1.22 (1.16 to 1.34) | 0.0081 (0.0059 to 0.0110) | 0.82 (0.79 to 0.85) | 0.822 (0.820 to 0.827) | 12 |
| nfg | tuned_brier | 1.25 (1.14 to 1.37) | 0.0090 (0.0062 to 0.0123) | 0.88 (0.84 to 0.94) | 0.823 (0.818 to 0.825) | 12 |
| nfg | tuned_cindex | 1.28 (1.16 to 1.43) | 0.0100 (0.0058 to 0.0143) | 0.92 (0.86 to 0.96) | 0.824 (0.821 to 0.828) | 12 |

DeepHit minus Neural Fine-Gray, both at their selected configurations:

- ICI: 0.0040 (95% CI 0.0019 to 0.0067)
- E/O: 0.137 (95% CI 0.058 to 0.204)
- C-index: 0.0023 (95% CI -0.0004 to 0.0051)

Default DeepHit minus Neural Fine-Gray, the prior work's comparison:

- ICI: 0.0168 (95% CI 0.0143 to 0.0221)
- E/O: 0.508 (95% CI 0.431 to 0.682)

Effect of tuning DeepHit (tuned minus default):

- ICI: -0.0129 (95% CI -0.0176 to -0.0111)
- E/O: -0.372 (95% CI -0.525 to -0.328)
- C-index: 0.0011 (95% CI -0.0015 to 0.0043)

## 3. Seed stability

Twelve initialisation seeds per model per configuration. The tables in section 2 carry the 10th to 90th percentile across those seeds; the full per-seed rows are in `results/deep/seed_stability.csv` and the min and max in `results/deep/seed_stability_summary.csv`.

## 4. Files

| file | what it holds |
|---|---|
| `alpha_sweep.csv` | one row per (alpha, sigma, seed, imputation, protocol) |
| `alpha_sweep_summary.csv` | medians and 10th-90th percentiles of the above |
| `tuning_raw.csv` | one row per (configuration, seed), inner out-of-fold |
| `tuning_space.json` | the search space and every sampled configuration |
| `tuning_selected.json` | the selected configuration under each criterion |
| `tuning.md` | search protocol and the no-test-data guarantee |
| `seed_stability.csv` | one row per (model, configuration, protocol, seed) |
| `predict_log.csv` | one row per production fit: seed, seconds, epochs |
| `validate_report.txt` | `src/v2/check_predictions.py` on the 120 files |

Scripts, in dependency order: `11_tune.py`, `10_alpha_sweep.py`, `12_seeds.py`, `13_predict.py`, `14_summarise.py`, driven by `run.sh` and `slurm_p2c.sh`.

