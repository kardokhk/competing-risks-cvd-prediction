# Competing risks in machine-learning cardiovascular risk prediction

Reproducible analysis pipeline for the study *"Competing risks change predicted cardiovascular
risk more than they change decisions"*.

Four survival learners are each paired with a competing-risk counterpart, so that the only
difference within a pair is how death from other causes is handled. All ten models are evaluated
against the Aalen-Johansen cumulative incidence for the full calibration hierarchy,
inverse-probability-of-censoring-weighted discrimination, and competing-risk-consistent net
benefit. Every comparison is a paired contrast computed on shared bootstrap resamples.

Everything runs on data that require no application.

## What the analysis finds

Ignoring competing risks inflates predicted 10-year cardiovascular mortality. The paired
difference in the expected-to-observed ratio between the naive and cause-specific Cox models is
0.202 (95% CI 0.180 to 0.227) on a temporal test set and 0.157 (0.148 to 0.167) under
cross-validation, an inflation factor of 1.158 on both designs. Discrimination is unchanged. The
naive model places more people above treatment thresholds, but no learner-matched pair shows a
net-benefit difference excluding zero at any threshold appropriate to the endpoint.

A closed-form identity accounts for the inflation: 32.0% arises from the change of estimand and
72.2% from population heterogeneity, because an individual's cardiovascular and competing-death
risks are correlated (r = 0.93). Guidance keyed to the marginal competing-event rate therefore
captures only the smaller term.

## Data

NHANES 1999 to 2018 linked to the NCHS 2019 public-use Linked Mortality File. Both are public and
need no data-use agreement.

The raw files are not redistributed here. `data/raw/manifest.csv` lists all 131 of them with source
URLs, retrieval timestamps, byte counts and SHA-256 hashes, and `data/raw/verification_20260904.csv`
records a re-download on 4 September 2026 that confirmed all 131 are byte-identical to the copies
the analysis used. `src/data/download.py` fetches them and writes the same manifest fields, so a
reader can verify provenance rather than take it on trust.

`data/processed/analytic.parquet` is the assembled analysis cohort (41,151 respondents aged 30 to
79 eligible for mortality linkage) with `codebook.md` describing every column and `cohort_flow.md`
the sample flow.

## Layout

```
src/data/          assembly of the analysis cohort from the raw files
src/v2/            the analysis: imputation, model fitting, evaluation, tables
src/v2/deep/       DeepHit and Neural Fine-Gray, including the alpha sweep
src/v2/figures/    figure generation
src/closed_form/   derivation, decomposition and validation of the closed form
results/           metrics, paired contrasts, curve data, stratified tables
figures/           the figures, with captions, alt text and the data behind each panel
environment.md     software versions, seeds and data retrieval dates
```

`src/v2/run_all.sh` gives the order. `src/v2/PREDICTION_FORMAT.md` documents the schema every model
writes, so a new model family can be added without touching the evaluation.

## Reproducing

Model fitting and the bootstrap are the expensive steps and were run on a compute cluster; the
Slurm scripts in `src/v2/` are included as submitted. Everything downstream of the saved predictions
runs on a laptop in minutes.

Two environments are needed, both described with exact versions in `environment.md`: an R stack
(riskRegression, cmprsk, randomForestSRC, pec, prodlim, timeROC, mice, arrow) and a Python stack
(scikit-survival, lifelines, pycox, PyTorch CPU, pandas, pyarrow). Paths in the scripts are written
against `${ENV_PREFIX}` and `<REPO_ROOT>`; set those for your machine.

Seed 20260903 throughout.

### Redoing the analysis from what is in this repository

Everything the paper reports can be recomputed from the files here, in this order.

1. **Verify the inputs.** `python src/data/download.py` re-fetches all 131 raw NHANES and NCHS
   files and writes a manifest with SHA-256 hashes. Compare it against `data/raw/manifest.csv`;
   the copies used in the analysis were re-verified byte-identical on 4 September 2026 and the
   result of that check is in `data/raw/verification_20260904.csv`.
2. **Rebuild the cohort**, or skip this and use the copy provided. `src/data/` assembles
   `data/processed/analytic.parquet` (41,151 rows, 31 columns) from the raw files. The assembled
   file is included here, so steps 3 onward run without step 1.
3. **Impute, fit and evaluate.** `src/v2/run_all.sh` gives the order of the numbered scripts,
   from `01_impute.R` through the evaluation and pooling. The imputation, the model fits and the
   shared-index bootstrap are the expensive steps; the Slurm scripts are included as they were
   submitted.
4. **Recompute the closed form.** `src/closed_form/` derives the identity, validates it against
   the fitted predictions, and writes `results/closed_form/`. This runs in seconds and needs only
   the Python stack.
5. **Rebuild the tables and figures.** `python src/v2/14_tables.py` writes the three main tables,
   and `bash src/v2/figures/run_all.sh` regenerates all twelve figures with the `_data.csv` file
   behind every panel.

Steps 4 and 5 read only files that are in this repository, so the closed-form result, all tables
and all figures can be reproduced without rerunning any model. Steps 1 to 3 reproduce the fitted
predictions themselves and need the two environments and a machine with the memory and time noted
above.

## Reading the results

`results/metrics_v2/SUMMARY.md` states every number the paper quotes and the script that produced
it. `results/closed_form/SUMMARY.md` does the same for the decomposition.

Two columns in `results/metrics_v2/main_metrics.csv` and `paired_contrasts.csv` deserve attention:

- `interval_valid` is `FALSE` for 90 and 45 rows respectively. In every case the pooled point
  estimate falls outside its own percentile interval. All are on the cross-validated split and all
  are smoothed-calibration metrics. Recomputation did not remove it, which is consistent with bias
  from refitting a flexible smoother on resampled data. **Do not quote those intervals.** Every
  smoothed-calibration value in the paper is from the temporal split.
- `fmi` is the fraction of missing information. It exceeds 0.9 for a small number of
  threshold-crossing statistics on the two deep models, which are 10-member ensembles where the
  other eight families are single fits.

Both are documented rather than removed, because a reader should be able to see them.

## Authors

Kardokh Kakabra (ORCID 0009-0000-4403-8607) and Osama Soliman (ORCID 0000-0003-0758-3539),
Cardiovascular Research Institute Dublin (CVRI Dublin) and RCSI University of Medicine and Health
Sciences, Dublin, Ireland. Correspondence to Osama Soliman, osamasoliman@rcsi.com.

## Citation

Cite this repository using `CITATION.cff`, which GitHub renders as a ready-made citation under
"Cite this repository". The release tagged `v1.0.0` is the version the manuscript reports; cite
that tag rather than `main` if you need a fixed reference. Details of the article will be added
here on publication, together with an archival Zenodo DOI.

## Licence

Code is released under the MIT Licence (`LICENSE`). NHANES and the NCHS Linked Mortality Files are
public-domain US Government works; see the NCHS data-use restrictions for the linked mortality data
at https://www.cdc.gov/nchs/data-linkage/mortality-public.htm.
