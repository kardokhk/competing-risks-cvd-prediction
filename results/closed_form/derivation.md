# Closed form for the naive minus competing-risk gap

Agent P2-B. 2026-09-03. Seed 20260903.
Scripts: `src/closed_form/cf_lib.py`, `cf1_cause2_fit.R`, `cf2_decomposition.py`,
`cf3_boot_refit.R`, `cf4_validation.py`, `cf5_nomogram.py`. Run with
`src/closed_form/run.sh`.

## 1. Notation

Cause 1 is CVD death, cause 2 is non-CVD death. For cause j,

- lambda_j(u) is the cause-specific hazard and Lambda_j(t) = int_0^t lambda_j(u) du,
- S(t) = exp(-Lambda_1(t) - Lambda_2(t)) is all-cause survival,
- F_j(t) = int_0^t S(u) lambda_j(u) du is the cause-j cumulative incidence,
- A(t) = 1 - exp(-Lambda_1(t)) is the naive estimand, one minus the Kaplan-Meier
  curve obtained by treating competing deaths as censored,
- G(t) = A(t) - F_1(t) is the gap.

Write p = F_1 + F_2 = 1 - S(t) and pi = F_1 / p.

## 2. The exact result

Assume the two cause-specific hazards are proportional, lambda_2(u) = theta lambda_1(u)
for all u, with theta >= 0 arbitrary and the common time shape arbitrary. Then

    F_1(t) / F_2(t) = Lambda_1(t) / Lambda_2(t) = 1 / theta

because the integrand S(u) lambda_j(u) carries the same S(u) for both causes. Hence
Lambda_1(t) = pi (Lambda_1(t) + Lambda_2(t)) = -pi log S(t), and the cause-1 net
survival is S_1 = exp(-Lambda_1) = S^pi. Therefore

    A = 1 - (1 - p)^pi ,        G = 1 - (1 - p)^pi - F_1 .                     (1)

Equation (1) has no fitted parameters. It depends on the two cumulative incidences
only, not on the time shape of the hazards, not on the follow-up distribution, and
not on theta separately.

Expanding (1) for small p gives G = (1/2) F_1 F_2 + O(p^3), so the multiplicative
inflation of a naive model's expected/observed ratio is A / F_1 = 1 + F_2/2 + O(p^2).
That second-order form is an approximation and is reported only as such.

## 3. What breaks when proportionality fails

Without proportionality,

    G = int_0^t exp(-Lambda_1(u)) [1 - exp(-Lambda_2(u))] dLambda_1(u)
      = kappa(t) F_1(t) F_2(t) + O(p^3),

    kappa(t) = int_0^t [Lambda_2(u-) / Lambda_2(t)] dLambda_1(u) / Lambda_1(t)  (2)

with kappa in [0, 1]. Equation (2) is a hazard-timing overlap coefficient: it is the
average, weighted by the accrual of cause-1 hazard, of the fraction of the competing
hazard that has already accrued. Proportional hazards gives kappa = 1/2 exactly.
kappa > 1/2 means the competing hazard accrues earlier than the cause-1 hazard and
the closed form under-predicts the gap; kappa < 1/2 means it accrues later and the
closed form over-predicts. `cf_lib.kappa_nelson_aalen` estimates kappa
non-parametrically from Nelson-Aalen increments.

Under a pair of Cox models with a shared covariate vector, Lambda_ji(u) =
exp(beta_j' x_i) Lambda_j0(u), so the subject index cancels from every ratio in (2)
and kappa is a single scalar per horizon, a property of the baseline shapes alone.

Empirically (`validation.csv`, 65 strata, section 5) the relative error of (1) is
almost a deterministic function of kappa: Spearman rho = -0.999 (n = 65), crossing
zero at kappa = 0.5. The error characterisation of the closed form is therefore
complete: the entire stratum-level error is the departure of kappa from 1/2.

## 4. Prior-art verdict

**The identity is not new.** Equation (1) is the classical actuarial conversion
between crude and net probabilities of decrement, q'_j = 1 - (1 - q)^(D_j/D),
derived under the "constant ratio of forces of decrement" assumption, which is
proportional cause-specific hazards under another name. Chiang 1968 (p. 242) is the
primary source; Newman 1987 states it in a form directly comparable to (1) and
attributes it to Chiang. The same assumption underlies cause-deleted life tables in
demography (Beltran-Sanchez, Preston and Canudas-Romo 2008). Any manuscript using
(1) must cite this and present it as a restatement, not a derivation.

Checked and found **not** to contain (1) or the equivalent S_1 = S^pi:

- Gooley et al. 1999 gives only the inequality 1 - KM >= CIF, with equality iff no
  competing event precedes the first event of interest.
- Putter, Fiocco and Geskus 2007 treats proportional cause-specific hazards
  regression at length but gives no closed-form link between 1 - KM and the CIF.
- Austin, Lee and Fine 2016 states the overestimation qualitatively only.
- Andersen and Keiding 2012 section 4.1.2 equation 19 gives the general net-risk
  functional 1 - exp(-int h_j), in order to argue against latent failure times; the
  proportional-hazards special case is not written down.
- Pintilie 2006 could not be checked against the primary text; convergent secondary
  sources describe a qualitative bias statement only. **This one remains open.**

No published source was found for the second-order forms G ~ kappa F_1 F_2 or
A / F_1 ~ 1 + F_2 / 2.

The contribution of this paper is therefore the empirical decomposition in section 5,
not equation (1).

### References

1. Chiang CL. Introduction to Stochastic Processes in Biostatistics. New York: John
   Wiley and Sons; 1968. (Net and crude probability conversion, p. 242.) Accessed
   via secondary citation 2026-09-03; no DOI, pre-DOI monograph.
2. Newman SC. Formulae for cause-deleted life tables. Stat Med. 1987;6(4):527-528.
   doi:10.1002/sim.4780060411. Verified against Crossref 2026-09-03; no retraction.
3. Beltran-Sanchez H, Preston SH, Canudas-Romo V. An integrated approach to cause-of-death
   analysis: cause-deleted life tables and decompositions of life expectancy.
   Demogr Res. 2008;19(35):1323-1350. doi:10.4054/DemRes.2008.19.35. Verified 2026-09-03.
4. Gooley TA, Leisenring W, Crowley J, Storer BE. Estimation of failure probabilities in
   the presence of competing risks: new representations of old estimators. Stat Med.
   1999;18(6):695-706. doi:10.1002/(SICI)1097-0258(19990330)18:6<695::AID-SIM60>3.0.CO;2-O.
   Verified 2026-09-03; no retraction.
5. Putter H, Fiocco M, Geskus RB. Tutorial in biostatistics: competing risks and
   multi-state models. Stat Med. 2007;26(11):2389-2430. doi:10.1002/sim.2712.
   Verified 2026-09-03; no retraction.
6. Austin PC, Lee DS, Fine JP. Introduction to the analysis of survival data in the
   presence of competing risks. Circulation. 2016;133(6):601-609.
   doi:10.1161/CIRCULATIONAHA.115.017719. Verified 2026-09-03; no retraction.
7. Andersen PK, Keiding N. Interpretability and importance of functionals in competing
   risks and multistate models. Stat Med. 2012;31(11-12):1074-1088.
   doi:10.1002/sim.4385. Verified 2026-09-03; no retraction.
8. Pintilie M. Competing Risks: A Practical Perspective. Chichester: Wiley; 2006.
   ISBN 9780470870686. Primary text not accessed; verdict provisional.

## 5. What equation (1) does and does not explain

See `SUMMARY.md` for every number. In outline:

- Applied to the **pooled** cumulative incidences of a stratum, equation (1)
  reproduces the observed non-parametric gap between 1 - KM and the Aalen-Johansen
  CIF to a median relative error of -1.2% over 65 NHANES strata, with the residual
  error fully accounted for by kappa.
- Applied at the **pooled** level to explain the difference between the mean
  predictions of a naive Cox model and a cause-specific Cox model, it accounts for
  only about a third of what is observed. That shortfall is not a failure of the
  identity. The difference between two models' mean predictions is a mean of
  per-subject gaps, and the gap is convex in F_2 while F_1i and F_2i are strongly
  positively correlated across subjects (Pearson r = 0.93 in this cohort).
- Applied **per subject** and averaged, equation (1) accounts for essentially all of
  the difference (104% at 10 years). The heterogeneity term alone is 72%.

The practical consequence: a reader who takes a cohort's marginal F_1 and F_2 to a
nomogram of equation (1) will substantially understate how far a naive model's mean
prediction sits above a competing-risk model's. In this cohort the understatement is
a factor of 3.2 to 3.3, near-constant across horizons.
