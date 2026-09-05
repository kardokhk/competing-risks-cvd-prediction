# Single-seed instability of deep survival models under multiple imputation

Written by `src/v2/deep/16_instability.py`. Every number traces to a file named in
its docstring. Horizon 10 y throughout; `mean_pred` is the mean predicted CVD
cumulative incidence over the split and `frac_above_0.01` the fraction of the split
predicted above a 1% 10 y risk.

## The finding

A deep survival model fitted once per imputed dataset produces between-imputation
variation that is dominated by the initialisation of the network rather than by the
missing data. Rubin's rules charge that optimisation noise to the between-imputation
component, so the fraction of missing information for any statistic that reads the
level of the predicted risks approaches, and under the usual small-sample
approximation exceeds, 1. None of the eight classical families in the same benchmark
shows it. The practical consequence is that raising the number of imputations does
not help, because the excess variance is not imputation variance.

## 1. How far the predictions move between imputations

Thirty imputed datasets, one fit per dataset, the seed rule of
`src/v2/PREDICTION_FORMAT.md` section 7. Standard deviation across the 30 files of
the mean predicted 10 y risk, with the coefficient of variation, on the temporal
test set (N = 9,329):

| model | mean predicted 10 y risk | SD across imputations | CV | SD of `frac_above_0.01` |
|---|---|---|---|---|
| `logistic` | 0.0536 | 0.000281 | 0.53% | 0.0016 |
| `cox_naive` | 0.0488 | 0.000183 | 0.37% | 0.0011 |
| `rsf_naive` | 0.0491 | 0.000226 | 0.46% | 0.0040 |
| `gbs_naive` | 0.0501 | 0.000346 | 0.69% | 0.0036 |
| `logistic_cr` | 0.0423 | 0.000193 | 0.46% | 0.0014 |
| `csc_cox` | 0.0422 | 0.000145 | 0.34% | 0.0011 |
| `fine_gray` | 0.0425 | 0.000129 | 0.30% | 0.0012 |
| `rsf_cr` | 0.0427 | 0.000173 | 0.41% | 0.0044 |
| `deephit` (deep) | 0.0452 | 0.003021 | 6.69% | 0.0394 |
| `nfg` (deep) | 0.0421 | 0.003856 | 9.16% | 0.0316 |

The eight classical families sit between 0.30% and 0.69%. DeepHit is at 6.69% and Neural
Fine-Gray at 9.16%, an order of magnitude larger. In absolute
terms DeepHit's mean predicted risk moves over a range of 0.0148 across the 30 imputations and Neural Fine-Gray's
over 0.0141, against 0.0007
for the cause-specific Cox model on the same 30 datasets.

The cross-validated split shows the same ordering at a smaller absolute size,
1.85% for DeepHit and 1.61% for Neural
Fine-Gray against 0.06% to 0.25%
for the classical families. Each cross-validated file is already the union of fifteen
separately seeded fold fits, so part of the initialisation noise has been averaged
away inside the file before pooling ever begins.

## 2. What that does to the pooled inference

The fraction of missing information reported by `src/v2/08_pool_and_report.py` is the
usual `lambda + 2 / (nu + 3)`, with `lambda = (1 + 1/m) B / T` the between-imputation
share of the total variance. The correction term is an approximation that is only
meaningful while `lambda` is well below 1; when `lambda` approaches 1 the sum can
exceed 1, which is the signature seen here. A value above 1 is therefore not a
quantity to be interpreted, it is a statement that the between-imputation variance
has swallowed the entire sampling variance of the estimate.

Over the 380 pooled deep-model estimands of the single-seed run, 27 exceeded 1 and 53 exceeded 0.9; the
largest was 1.057. Over the 1520 estimands of the eight classical
families the largest was 0.840. Single-seed values at the primary
horizon on the temporal split:

| statistic | `csc_cox` | `fine_gray` | `rsf_cr` | `deephit` | `nfg` |
|---|---|---|---|---|---|
| `mean_pred` | 0.058 | 0.053 | 0.083 | 1.018 | 1.041 |
| `eo` | 0.004 | 0.003 | 0.006 | 0.628 | 0.777 |
| `ici` | 0.009 | 0.007 | 0.015 | 0.775 | 0.844 |
| `calib_slope` | 0.029 | 0.025 | 0.060 | 0.389 | 0.340 |
| `cindex` | 0.008 | 0.006 | 0.032 | 0.103 | 0.094 |
| `frac_above_0.01` | 0.046 | 0.061 | 0.456 | 1.057 | 1.042 |
| `frac_above_0.05` | 0.034 | 0.061 | 0.295 | 1.005 | 1.024 |
| `frac_above_0.2` | 0.128 | 0.079 | 0.425 | 1.035 | 1.051 |

For contrast, the paper's primary paired contrast, `cox_naive` minus `csc_cox`, has a
fraction of missing information of 0.013 to 0.049 across horizons and splits.

## 3. It is the seed, not the imputation

A pilot crossing five imputations with four initialisation seeds, at the selected
configurations and on the temporal split, separated the two sources before any of
the production work was committed (`notes/scratch/pilot_decomp.csv`). For the mean
predicted 10 y risk the estimated imputation component was zero for both models:
the variance across seeds within an imputation already accounted for the whole of
the variance a single-seed run sees across imputations.

The production ensemble settles it on the full grid. Every unit of the temporal
split was refitted at 10 independently initialised networks
across all 30 imputations, giving a balanced
imputation-by-seed design. The statistics below are recomputed directly from the
600 cached temporal member prediction files rather than from the fitting log
`results/deep/ensemble_members.csv`, which is 17 rows short of the 9,600 parts on
disk because members served from cache were not re-logged. Decomposing the
variance of each member-level statistic:

| model | statistic | SD across seeds, within an imputation | SD across imputations, seed removed | SD a single-seed run sees | share attributable to the seed |
|---|---|---|---|---|---|
| `deephit` | `frac_above_0.01` | 0.037809 | 0.008794 | 0.038788 | 95% |
| `deephit` | `frac_above_0.05` | 0.015629 | 0.002227 | 0.015782 | 98% |
| `deephit` | `frac_above_0.2` | 0.012424 | 0.000000 | 0.012370 | 101% |
| `deephit` | `mean_risk10` | 0.002823 | 0.000000 | 0.002789 | 102% |
| `nfg` | `frac_above_0.01` | 0.039543 | 0.003774 | 0.039717 | 99% |
| `nfg` | `frac_above_0.05` | 0.019309 | 0.005331 | 0.020010 | 93% |
| `nfg` | `frac_above_0.2` | 0.013419 | 0.000000 | 0.013412 | 100% |
| `nfg` | `mean_risk10` | 0.003533 | 0.000639 | 0.003589 | 97% |

Read the last column as the fraction of the between-imputation variance of a
single-seed run that is not imputation variance at all. For the mean predicted
10 y risk it is 102% for `deephit` and 97% for `nfg`.

## 4. What removes it, and what does not

Raising the number of imputations does not remove it. The between-imputation variance
`B` is not an estimate of imputation uncertainty here, so `m` buys nothing except a
more precise estimate of the wrong quantity. Averaging the predicted risks of `S`
independently initialised networks within each imputation does remove it, because the
seed component of `B` falls as `1/S` while the genuine imputation component and the
within-imputation sampling variance are untouched.

Refitting every unit at ten seeds and averaging the predicted risks moved the
380 deep-model estimands as follows.

| | single seed | ten-seed ensemble |
|---|---|---|
| largest FMI | 1.057 | 0.942 |
| estimands with FMI above 1 | 27 | 0 |
| estimands with FMI above 0.9 | 53 | 4 |
| median FMI | 0.175 | 0.039 |

At the primary horizon on the temporal split:

| statistic | `deephit` single | `deephit` ensemble | `nfg` single | `nfg` ensemble |
|---|---|---|---|---|
| `mean_pred` | 1.018 | 0.619 | 1.041 | 0.857 |
| `eo` | 0.628 | 0.092 | 0.777 | 0.245 |
| `ici` | 0.775 | 0.165 | 0.844 | 0.379 |
| `calib_slope` | 0.389 | 0.151 | 0.340 | 0.139 |
| `cindex` | 0.103 | 0.026 | 0.094 | 0.026 |
| `frac_above_0.01` | 1.057 | 0.942 | 1.042 | 0.942 |
| `frac_above_0.05` | 1.005 | 0.556 | 1.024 | 0.822 |
| `frac_above_0.2` | 1.035 | 0.852 | 1.051 | 0.881 |


Two estimands per split remain above 0.9 after ensembling, both
`frac_above_0.01`, at 0.94. Ensembling divides the seed component of `B` by `S`
but leaves the within-imputation sampling variance `Ubar` untouched, and the FMI
is the ratio of the first to the total, so an estimand with a small `Ubar`
relative to its residual `B` stays high. A count of the sample on one side of a
threshold sitting in the densest part of the predicted-risk distribution is
exactly that: the bootstrap barely moves it within an imputation, while any
residual shift in the level of the predictions carries many subjects across the
threshold at once. Its decomposition also carries the largest genuine imputation
component of any statistic here for `deephit`, 0.0088 against 0.0000 for
`mean_risk10`, so part of what remains is real. `frac_above_0.01` is therefore
pooled at m = 30 below the m its own diagnostic asks for and should be reported
with that caveat rather than dropped or quoted as though it were stable.

## 5. What a reader should take from it

Any benchmark that combines multiple imputation with a learner fitted by stochastic
optimisation from a random start, and that fits that learner once per imputed dataset,
will attribute the learner's own optimisation noise to the missing data. The
diagnostic is cheap: compare the standard deviation of a level statistic across
imputations against the same quantity for a deterministic learner on the same
datasets, or refit one imputation at several seeds and compare. The remedy is to
average a fixed number of seeds within each imputation and to say how many. The cost
is that the deep families become ensembles while deterministic families are single
fits, which has to be declared, because an ensemble is a different estimator from the
model it ensembles.

## 6. Files

| file | what it holds |
|---|---|
| `results/deep/predictions_singleseed/` | the 120 archived single-seed prediction files and the pooled results computed from them |
| `results/deep/instability_spread.csv` | the between-imputation spread of section 1, all ten models |
| `results/deep/instability_member_stats_temporal.csv` | the 30 x 10 member-level statistics behind section 3 |
| `results/deep/before_after_metrics.csv` | paired single-seed and ensembled point estimates and intervals |
| `results/deep/deephit_vs_nfg__single_seed.csv`, `__ensemble.csv` | the paired DeepHit - NFG contrast under each run |
| `results/deep/ensemble_members.csv` | one row per member fit: seed, seconds, epochs, mean predicted risk, threshold fractions |
| `results/deep/fmi_before_after.csv` | the paired single-seed and ensembled pooling diagnostics |
| `notes/scratch/pilot_decomp.csv` | the 5 imputations by 4 seeds pilot |

