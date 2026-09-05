# Closed form for the naive minus competing-risk gap: results summary

Agent P2-B. Project `methods_paper`, working directory
`<REPO_ROOT>`. Date 2026-09-03.
Seed 20260903. Reproduce with `bash src/closed_form/run.sh`.

Environment: Python 3.11.16 (`${ENV_PREFIX}/crcvd-py`), numpy 2.4.6,
pandas 2.3.3, scipy 1.17.1; R 4.5.3 (`${ENV_PREFIX}/crcvd-r`),
riskRegression 2026.3.11, survival 3.8.11, prodlim 2026.3.11, arrow 25.0.0.

---

## 1. The identity

With p = F_1 + F_2 and pi = F_1 / p, under proportional cause-specific hazards
(lambda_2 = theta lambda_1, any theta, any time shape):

    A = 1 - KM = 1 - (1 - p)^pi ,      G = A - F_1 .

Zero fitted parameters. Derivation and assumptions in `derivation.md` section 2.
**The identity is prior art**: it is the classical actuarial net-versus-crude
decrement conversion (Chiang 1968; Newman 1987, doi:10.1002/sim.4780060411). Gooley
1999, Putter 2007, Austin 2016 and Andersen and Keiding 2012 were checked and do not
contain it. Pintilie 2006 could not be checked against the primary text and remains
open. Full verdict and references in `derivation.md` section 4.

---

## 2. Headline decomposition (the paper's central number)

Temporal test set, n = 8,096 (NHANES 2007-2010), models trained on 1999-2006.
Source: `decomposition.csv` (script `src/closed_form/cf2_decomposition.py`),
CIs from `decomposition_refit_boot.csv` (script `cf3_boot_refit.R`, 300 replicates
resampling the training set, refitting the cause-specific Cox, and resampling the
test set).

Observed difference in mean predicted risk, cox_naive minus csc_cox, at 10 years:

    D_obs = 0.005328   (0.004999 to 0.005688, test-set bootstrap)

decomposed as

| component | value at 10 y | share of D_obs | 95% CI on the share |
|---|---:|---:|---|
| change of estimand, at pooled mean incidences | 0.001707 | 32.0% | 28.8 to 35.0% |
| heterogeneity (Jensen term) | 0.003847 | 72.2% | 66.9 to 77.5% |
| hazard timing (kappa departing from 1/2) | -0.000199 | -3.7% | -11.1 to +2.9% |
| implementation (sksurv Cox versus R coxph) | -0.000027 | -0.5% | deterministic |

Same pattern at the other horizons (`decomposition.csv`):

| horizon | D_obs | estimand | heterogeneity | timing | implementation |
|---:|---:|---:|---:|---:|---:|
| 5 y | 0.000845 | 34.1% (29.6-39.5) | 77.1% (70.0-88.6) | -11.2% (-27.5 to -0.6) | -1.1% |
| 10 y | 0.005328 | 31.9% (28.8-35.0) | 71.8% (66.9-77.5) | -3.7% (-11.1 to +2.9) | -0.5% |
| 15 y | 0.017756 | 30.6% (28.1-32.9) | 68.2% (64.3-72.6) | +1.3% (-4.6 to +6.0) | -0.4% |

Share CIs are from the refit bootstrap and are quoted as shares of D_model (the
estimand gap with one fitted model), which differs from D_obs by the implementation
term, at most 1.1%.

**Answer to the question the brief posed.** The missing 69% is heterogeneity. The
closed form evaluated at pooled mean incidences explains 32% of the naive-versus-
competing prediction difference; evaluated per subject and averaged it explains
104%, that is, all of it to within a -4% over-explanation attributable to hazard
timing and implementation. G is convex in F_2 and the per-subject F_1i and F_2i are
strongly positively correlated in this cohort (Pearson r = 0.930, 95% CI 0.925 to
0.934, at 10 years), because both are driven by age. Mean(F_1i F_2i) = 0.007911
against mean(F_1i) mean(F_2i) = 0.003174, a factor of 2.49.

The heterogeneity multiplier, mean_i G(F_1i, F_2i) divided by G(mean F_1, mean F_2),
is 3.26 at 5 y, 3.25 at 10 y and 3.23 at 15 y: near-constant across horizons.

### In E/O units, restating the brief's table

| quantity | value | source |
|---|---:|---|
| observed competing-risk E/O (csc_cox, 10 y) | 1.223 | `temporal_eo_context.csv` |
| observed naive E/O (cox_naive, 10 y) | 1.398 | `temporal_eo_context.csv` |
| naive E/O predicted from the POOLED closed form | 1.279 | 1.223 x 1.0459 |
| naive E/O predicted from the PER-SUBJECT closed form | 1.406 | 1.223 x 1.1493 |
| share of the naive-competing E/O gap explained, pooled | 32.0% | |
| share of the naive-competing E/O gap explained, per subject | 104.3% | |

### Hypotheses tested and ruled out

**Different risk sets.** The naive Cox and the cause-specific Cox for cause 1 are
the same regression. Both treat competing deaths as censored, so both use the same
risk set, the same partial likelihood and the same cause-1 baseline cumulative
hazard. Only the transformation of that hazard into a risk differs:
1 - exp(-Lambda_1) for the naive model, int S dLambda_1 for the competing-risk
model. A change in the effective sample therefore cannot contribute, and this is
also why the fitted coefficients are irrelevant to the gap.

**Different baseline hazard estimation.** Quantified as the `implementation` term
by comparing the benchmark's `cox_naive` prediction against 1 - exp(-Lambda_1i)
recomputed from the refit cause-specific Cox. It is -0.5% of D_obs at 10 years
(-1.1% at 5 y, -0.4% at 15 y), covering every difference between sksurv
`CoxPHSurvivalAnalysis(alpha=1e-6)` with a Breslow baseline and R `coxph` with Efron
ties. The competing-risk half of the comparison contributes exactly nothing: mean
refit `risk1_h` minus mean benchmark `csc_cox` `risk_h` is 0 to within 1e-17.

**Hazard timing.** The one mechanism that does bite, and it bites in the opposite
direction to the one the heterogeneity hypothesis needed. Under the fitted pair of
Cox models the subject index cancels from every ratio in the definition of kappa, so
kappa is a single scalar per horizon. It contributes -3.7% of D_obs at 10 years
(95% CI -11.1 to +2.9%), that is, the closed form very slightly over-explains.

### The 22% that is not a competing-risk phenomenon: confirmed as temporal drift

Both model families over-predict by about 22% on the temporal test set
(csc_cox E/O = 1.223 at 10 y). That is a shared miscalibration present in the
competing-risk model too, so by construction it cannot be a competing-risk artefact.
`drift_check.csv` (script `cf6_drift_check.py`, 500-replicate bootstrap) tests the
drift explanation directly by refitting the same models contemporaneously
out of fold across the whole 1999-2018 cohort:

| model | design | 5 y E/O | 10 y E/O | 15 y E/O |
|---|---|---|---|---|
| csc_cox | out-of-fold CV, 1999-2018 | 1.007 (0.919-1.113) | 1.006 (0.945-1.078) | 1.017 (0.959-1.078) |
| csc_cox | temporal, 2007-2010 | 1.398 (1.164-1.750) | 1.223 (1.096-1.386) | 1.491 (1.287-1.736) |
| cox_naive | out-of-fold CV, 1999-2018 | 1.063 (0.968-1.182) | 1.155 (1.087-1.232) | 1.298 (1.228-1.382) |
| cox_naive | temporal, 2007-2010 | 1.470 (1.221-1.816) | 1.398 (1.253-1.609) | 1.894 (1.589-2.210) |

The competing-risk model is essentially perfectly calibrated when training and test
periods overlap (E/O = 1.006 at 10 y, CI covering 1 at every horizon) and
over-predicts by 22% only under the temporal split. The 22% is drift, not competing
risks, and must not be attributed to the naive estimand.

The same table gives an independent check on section 2. The naive-to-competing E/O
ratio is 1.147 out of fold and 1.143 temporally at 10 years, against 1.149 predicted
by the per-subject closed form. The inflation is a stable property of the cohort's
risk distribution and survives a change of validation design; the shared 22% level
shift does not.

The 15 y temporal rows of `drift_check.csv` are not estimable (maximum follow-up
13.25 y, at_risk_frac = 0) and must not be quoted. The 15 y out-of-fold rows are
estimable because the early cycles carry 15 to 20 y of follow-up.

---

## 3. Validation of the identity itself

`validation.csv` (script `cf4_validation.py`). 90 strata built on the 35,309
complete-case subjects of NHANES 1999-2018: 5-year age band x horizon; sex x 10-year
age band x horizon; decile of out-of-fold csc_cox predicted risk x horizon.
25 strata had fewer than 15 cause-1 events and are flagged `sufficient = FALSE`;
65 remain. Within a stratum the Aalen-Johansen F_1 and F_2 and the naive 1 - KM are
all marginal quantities of that stratum, so heterogeneity does not enter and the only
assumption under test is proportionality of the two marginal cause-specific hazards.

Relative error of the theoretical gap against the observed non-parametric gap:

| statistic | value |
|---|---:|
| median | -1.24% |
| interquartile range | -6.00% to +6.23% |
| 5th to 95th percentile | -14.37% to +28.93% |
| minimum, maximum | -20.10%, +138.72% |
| fraction within 5% | 41.5% |
| fraction within 10% | 66.2% |
| fraction within 20% | 89.2% |
| fraction whose bootstrap CI covers zero | 89.2% |
| non-parametric kappa, median (IQR) | 0.5051 (0.4707 to 0.5320) |

Median relative error by stratum family and horizon:

| family | 5 y | 10 y | 15 y |
|---|---:|---:|---:|
| 5-year age band | +0.84% | +0.02% | +0.14% |
| risk decile | +0.68% | +0.09% | -1.28% |
| sex x 10-year age band | -3.51% | -1.34% | -4.02% |

R^2 over the 65 sufficient strata is 0.9973. **Do not quote that number alone.**
It is dominated by the largest-gap strata and hides a 5th-to-95th spread of -14% to
+29%. The distribution above is the reportable result.

**The residual error is fully characterised.** Relative error is an almost
deterministic decreasing function of the non-parametric kappa, crossing zero at
kappa = 0.5 exactly as the theory requires: Spearman rho = -0.999 (n = 65,
*P* = 9.6e-82), Pearson r = -0.925 (*P* = 3.5e-28). See `diag_nomogram.png`, right
panel. The largest errors are all in middle-aged, low-gap strata where kappa falls to
0.21-0.40, that is, where CVD deaths accrue early and competing deaths accrue late,
so the closed form over-predicts. The single worst case (csc_cox risk decile 6 at
5 y, +138.7%, CI +59.3% to +339.2%) has an observed gap of 0.000027 in absolute
risk and is of no practical consequence.

### Pooled non-parametric check on the temporal test set

`temporal_eo_context.csv`:

| horizon | F_1 obs | F_2 obs | gap obs | gap theory | relative error | kappa |
|---:|---:|---:|---:|---:|---:|---:|
| 5 y | 0.011734 | 0.031126 | 0.000209 | 0.000187 | -10.5% | 0.559 |
| 10 y | 0.030418 | 0.083720 | 0.001381 | 0.001364 | -1.2% | 0.506 |
| 15 y | 0.043989 | 0.121144 | 0.003014 | 0.002952 | -2.1% | 0.511 |

**The 15 y row is not a 15 year estimate.** Maximum follow-up on the temporal test
set is 13.25 y and the fraction still at risk at 15 y is 0.000, so these are values
at the last event time. The 15 y column of `temporal_eo_context.csv` must not be
quoted as calibration. The 15 y decomposition in section 2 is unaffected because it
uses predictions only. The 15 y strata in `validation.csv` are estimable because
they are built on the full 1999-2018 cohort, where the early cycles carry 15 to 20 y
of follow-up.

---

## 4. Per-subject cause-2 predictions

`cause2_predictions.parquet` (script `cf1_cause2_fit.R`, seed 20260903), n = 8,096.
Cause-specific Cox fitted with `riskRegression::CSC` on the training cycles
(1999-2006), same 11 predictors and same standardisation as
`src/models/c3_competing_classical.R`, which was not modified. Columns per horizon
h in {5, 10, 15}: `risk1_h` (cause-1 CIF), `risk2_h` (cause-2 CIF), `cumhaz1_h`,
`cumhaz2_h`, `naive1_h` = 1 - exp(-cumhaz1_h).

Replication check: mean `risk1_h` minus mean `csc_cox` `risk_h` from
`results/predictions/csc_cox__temporal.parquet` is 0.0 to within 1e-17 at every
horizon, so the refit reproduces the benchmark model exactly.

Mean predicted values:

| horizon | mean F_1 | mean F_2 | mean naive | mean gap |
|---:|---:|---:|---:|---:|
| 5 y | 0.01640 | 0.03453 | 0.01726 | 0.00086 |
| 10 y | 0.03720 | 0.08531 | 0.04256 | 0.00535 |
| 15 y | 0.06558 | 0.14590 | 0.08340 | 0.01782 |

---

## 5. Nomogram

`nomogram_data.csv` (script `cf5_nomogram.py`): 7,200 grid points over F_1 in
0.5-30% (0.5% steps) and F_2 in 0.5-60% (0.5% steps), restricted to F_1 + F_2 < 1.
Columns `A_exact`, `inflation` = A / F_1, `gap_abs`, plus `gap_abs_m2` and
`gap_abs_m3`. Pooled inflation ranges from 1.0025 to 1.7861 over the grid.

**Warning that must appear on the nomogram's face.** The grid gives the POOLED
inflation, that is, what happens to a group whose members all carry the same F_1 and
F_2. For a cohort with heterogeneous risk the mean of the per-subject gaps is
larger, by a factor of 3.2 to 3.3 in this cohort. A reader who takes a cohort's
marginal F_1 and F_2 to this grid will understate the difference between a naive
model's and a competing-risk model's mean predicted risk by roughly threefold. The
`gap_abs_m2` and `gap_abs_m3` columns bracket that multiplier. Present `gap_abs` as
a lower bound, not as a correction factor.

`nomogram_overlay.csv`: the 65 sufficient NHANES strata and the 3 pooled temporal
test rows positioned on the grid, with observed and theoretical gaps.
`diag_nomogram.png` is a working diagnostic, not a publication figure.

---

## 6. Open questions and things not resolved

1. Pintilie 2006 was not checked against the primary text. The prior-art verdict
   rests on Chiang 1968 via Newman 1987 plus four primary sources that do not
   contain the identity.
2. RESOLVED. The 22% shared over-prediction is temporal drift, confirmed by
   `drift_check.csv`: out-of-fold csc_cox E/O is 1.006 (0.945 to 1.078) at 10 y
   against 1.223 (1.096 to 1.386) on the temporal split. What remains open is the
   cause of the drift itself, presumably falling CVD mortality between 1999-2006 and
   2007-2010, which is outside the scope of this analysis.
3. The refit bootstrap resamples training and test rows independently and refits the
   cause-specific Cox, but does not refit the sksurv naive Cox, so the implementation
   term carries no interval. It is at most 1.1% of D_obs, so this does not affect
   any conclusion.
4. All estimates are unweighted. NHANES survey weights are ignored throughout, as in
   the rest of the guidance stage.
5. Heterogeneity here is heterogeneity of the FITTED model's predictions. If the
   model under-disperses risk relative to the truth, the true heterogeneity term is
   larger still, so 72% is a lower bound on the heterogeneity share.
