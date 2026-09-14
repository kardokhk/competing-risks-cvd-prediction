# Computational environment

Captured 2026-09-03 on MeluXina (EuroHPC, LuxProvide), user `u104629`, compute account
`p201509`. This file supersedes `env/README.md`, which documents the *previous* machine
(conda environments `base`, `cr_dl`, `r_440` under `/home/kardokhk/`). Those environments
do not exist here and none of their paths resolve.

## Host

| Field | Value |
|---|---|
| Cluster | MeluXina (EuroHPC), login node |
| OS | Red Hat Enterprise Linux 8.10 (Ootpa) |
| Kernel | 4.18.0-553.132.1.el8_10.x86_64 |
| glibc | 2.28 |
| System libstdc++ | 6.0.25, max `GLIBCXX_3.4.25` (see the libstdc++ note below) |
| Package manager | micromamba 2.9.0, `~/.local/bin/micromamba` |
| Project root | `/mnt/tier2/users/u104629/work/cvri/papers/ctrcd/methods_paper` |

`module` is not available on login nodes, so `module load` cannot be used. The login-node
`python3` is 3.6.8 and `pandoc`, `node` and `npm` are absent; all interpreters below come
from micromamba environments instead.

### Required shell exports

Environments and package caches must live under `/project/home/p201509` (4 TB, 1M inodes).
`$HOME` is limited to 100,000 files and must not receive either. Export before any
micromamba or pip call:

```bash
export MAMBA_ROOT_PREFIX=/project/home/p201509/micromamba
export XDG_CACHE_HOME=/project/home/p201509/cache
export PIP_CACHE_DIR=/project/home/p201509/cache/pip
```

`/project/home/p201509` is a symlink to `/mnt/tier2/project/p201509`; exported prefixes
show the physical path.

## Environments

Three environments cover the pipeline. `micromamba activate` is awkward in
non-interactive shells, so every entry point below is given as an absolute path. All
scripts are run **from the repository root**, because much of the pipeline reads
`data/...` and writes `results/...` by relative path.

| Environment | Prefix | Interpreter | Used by |
|---|---|---|---|
| `crcvd-py` | `/project/home/p201509/envs/crcvd-py` | Python 3.11.16 | data build, classical and ML survival models, figures |
| `crcvd-r` | `/project/home/p201509/envs/crcvd-r` | R 4.5.3 | competing-risk models, evaluation, bootstrap, sensitivity analyses |
| `crcvd-dl` | `/project/home/p201509/envs/crcvd-dl` | Python 3.10.21 | DeepHit, Neural Fine-Gray |

```bash
# crcvd-py
/project/home/p201509/envs/crcvd-py/bin/python src/data/build.py

# crcvd-r  (run from the repository root)
/project/home/p201509/envs/crcvd-r/bin/Rscript src/eval/run_eval.R

# crcvd-dl  -- use python-cr, NOT python (see the libstdc++ note)
/project/home/p201509/envs/crcvd-dl/bin/python-cr src/models/c5_deephit.py
```

### `crcvd-py` (Python 3.11.16)

Built by a separate session on 2026-09-03; exported here, not rebuilt. Full export:
`env/crcvd-py.yml`. No `torch`, by design: only `c5_deephit.py`, `cv_deephit.py` and
`c6_nfg.py` import torch, and those run in `crcvd-dl`.

| Package | Version | Previous machine |
|---|---|---|
| numpy | 2.4.6 | 1.26.4 |
| pandas | 2.3.3 | 2.x (conda build) |
| scipy | 1.17.1 | 1.12.0 |
| scikit-learn | 1.9.0 | 1.7.2 |
| scikit-survival | 0.28.0 | 0.25.0 |
| lifelines | 0.30.3 | 0.30.0 |
| pyarrow | 25.0.0 | 22.0.0 |
| xgboost | 3.2.0 | 2.1.4 |
| statsmodels | 0.15.0 | not recorded |
| matplotlib | 3.11.1 | 3.10.8 |

### `crcvd-r` (R 4.5.3, "Reassured Reassurer", 2026-03-11)

Built 2026-09-03 from conda-forge, except `dcurves`, which is CRAN-only and was installed
from `https://cloud.r-project.org` inside R. Full list of all 193 installed packages:
`env/crcvd-r_packages.txt`.

| Package | Version | Previous machine (R 4.4.3) |
|---|---|---|
| riskRegression | 2026.03.11 | 2026.3.11 |
| cmprsk | 2.2-12 | 2.2.12 |
| randomForestSRC | 3.7.0 | 3.6.2 |
| pec | 2025.06.24 | 2025.6.24 |
| dcurves | 0.5.1 | 0.5.1 |
| prodlim | 2026.03.11 | 2026.3.11 |
| timeROC | 0.4.1 | 0.4.1 |
| survival | 3.8-11 | 3.8.9 |
| data.table | 1.18.4 | 1.18.4 |
| arrow | 25.0.0 | 24.0.0 |
| mice | 3.19.0 | not recorded |
| survey | 4.5 | not recorded |
| boot | 1.3-32 | not recorded |
| jsonlite | 2.0.0 | not recorded |
| broom | 1.0.13 | 1.0.13 |

All 15 return `requireNamespace(..., quietly = TRUE) == TRUE`. The conda-forge `arrow`
build carries Parquet support (`arrow_info()$capabilities[["parquet"]] == TRUE`), verified
by reading the project data rather than by inspecting the flag alone:
`data/processed/analytic.parquet` returns 41,151 rows by 31 columns and
`data/processed/model_matrices/full.parquet` returns 35,309 rows by 31 columns, matching
`data/processed/_flow.json` (`n_eligible_final` = 41,151) and the documented modelling
cohort.

The env ships its own `curl` and libcurl ahead of the system ones on `PATH`, so
`install.packages` works from inside R with no extra options.

### `crcvd-dl` (Python 3.10.21)

Built 2026-09-03. Build order matters and was kept as on the previous machine: compiled
dependencies from conda-forge first, then pip for the survival libraries, then torch from
the PyTorch CPU index. Full export: `env/crcvd-dl.yml`.

```bash
micromamba create -y -p /project/home/p201509/envs/crcvd-dl -c conda-forge \
  python=3.10 pyarrow h5py numba scipy pandas scikit-learn numpy matplotlib tqdm pip
/project/home/p201509/envs/crcvd-dl/bin/pip install pycox torchtuples scikit-survival lifelines
/project/home/p201509/envs/crcvd-dl/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

| Package | Version | Source | Previous machine |
|---|---|---|---|
| python | 3.10.21 | conda-forge | 3.10.20 |
| torch | 2.14.0+cpu | PyTorch CPU index | 2.6.0+cpu |
| pycox | 0.3.0 | pip | 0.3.0 |
| torchtuples | 0.2.2 | pip | 0.2.2 |
| scikit-survival | 0.25.0 | pip | 0.25.0 |
| lifelines | 0.30.0 | pip | 0.30.0 |
| numpy | 2.2.6 | conda-forge | 2.2.6 |
| pandas | 2.3.3 | conda-forge | 2.3.3 |
| scipy | 1.15.2 | conda-forge | 1.15.2 |
| scikit-learn | 1.7.2 | conda-forge | 1.7.2 |
| pyarrow | 25.0.0 | conda-forge | 25.0.0 |
| h5py | 3.16.0 | conda-forge | 3.16.0 |
| numba | 0.67.0 | conda-forge | 0.66.0 |
| matplotlib | 3.10.9 | conda-forge | 3.10.9 (pip) |
| tqdm | 4.70.0 | conda-forge | not recorded |

`torch.cuda.is_available()` is `False`, which is intended: the analytic data is 35,309
tabular rows and the deep models are configured with `cuda=False`.

#### libstdc++ conflict, and the `python-cr` wrapper

The pip `torch` wheel binds the system `/usr/lib64/libstdc++.so.6.0.25` (RHEL 8, maximum
`GLIBCXX_3.4.25`). Once that SONAME is resolved, the loader will not load a second
`libstdc++.so.6`, so any conda-forge C++ extension imported afterwards fails with
`ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`. The environment's
own libstdc++ is 6.0.36 (`GLIBCXX_3.4.36`), a strict superset, so forcing it first fixes
the failure and leaves torch working.

Two fixes are installed:

* `/project/home/p201509/envs/crcvd-dl/bin/python-cr`, a wrapper that sets
  `LD_LIBRARY_PATH=$PREFIX/lib` and execs the real interpreter. **Use this as the
  interpreter for every script in this environment.**
* `/project/home/p201509/envs/crcvd-dl/etc/conda/activate.d/zzz_libstdcxx.sh`, which
  applies the same export for `micromamba activate` and `micromamba run`.

`bin/python` happens to work for `c5_deephit.py`, `cv_deephit.py` and `c6_nfg.py` as
currently written, because each imports `numpy` and `pandas` (which pull in the conda-forge
libstdc++) before `torch`. That is an accident of import order and will break on any
reordering, so `python-cr` is the documented entry point.

#### Neural Fine-Gray

The vendored code at `src/models/vendor/NeuralFineGray-main/` needs no additional
packages. `c6_nfg.py` puts the vendor directory and its bundled `DeepSurvivalMachines`
subdirectory on `sys.path`, and `from nfg import NeuralFineGray` then resolves. Fitting and
`predict_survival(..., risk=1)` were both exercised against the real training matrix under
torch 2.14.0+cpu.

## Verification run 2026-09-03

Beyond import checks, each stack was fitted to a subsample of
`data/processed/model_matrices/train.parquet` (12,209 rows) with seed 20260801.

| Check | Result |
|---|---|
| R: 15 pipeline packages load | all `TRUE` |
| R: `arrow` Parquet read of project data | 41,151 x 31 and 35,309 x 31 |
| R: `riskRegression::CSC` fit, 10-year cause-1 CIF | mean 0.03756 (n = 2,000) |
| R: `riskRegression::FGR` fit, 10-year cause-1 CIF | mean 0.03695 (n = 2,000) |
| R: `randomForestSRC` `logrankCR`, 20 trees | fitted |
| DL: `import pycox, torch, torchtuples`, `DeepHit` | torch 2.14.0+cpu |
| DL: DeepHit fit with the project's `CauseSpecificNet` | CIF array (2, 20, 200), 3 epochs, final loss 0.3631 |
| DL: Neural Fine-Gray fit and `predict_survival` | survival array (100, 3) |
| py: `pandas`, `numpy`, `sklearn`, `sksurv`, `lifelines`, `pyarrow`, `xgboost`, `matplotlib` | all import |

These are smoke tests on subsamples, not pipeline results. No stage of the analysis has
been rerun on this machine.

## Data provenance and retrieval dates

**Verified data access date: 2026-09-04.** The original download was not timestamped:
`data/raw/manifest.csv` carried columns `cycle,component,filename,url,status` only,
`logs/download.log` records per-file outcomes without dates, and filesystem mtimes
(2026-09-03 12:48) reflect the transfer to this machine rather than the download. The only
bound on the original retrieval was `env/README.md`, whose header reads "Documented
2026-08-01", placing the download on or before that date.

Rather than report a remembered date, all 131 files were re-downloaded from source on
2026-09-04 between 06:32:37Z and 06:35:28Z and compared with the repository copies by
SHA-256. All 131 returned HTTP 200 and **all 131 are byte-identical to the copies the
analysis used**. The files therefore carry a verified access date of 2026-09-04, backed by
a hash comparison rather than by recall, and no upstream file has been revised since the
original retrieval. Per-file evidence, including both hashes, byte counts, HTTP status and
per-file UTC retrieval time, is in `data/raw/verification_20260904.csv`; the narrative
record is in `data/raw/README.md`.

The fresh copies were written to `data/raw_verify_20260904/` and deleted after the hashes
were recorded. `data/raw/` was not modified. The root cause was fixed in
`src/data/download.py`, which now writes `retrieved_utc`, `http_status`, `bytes` and
`sha256` into the manifest for every file, so a future download records its own access date.

| Source | Detail |
|---|---|
| NHANES | Continuous NHANES, 10 cycles 1999-2000 through 2017-2018. XPT component files from `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{startyr}/DataFiles/{fname}.XPT`. 131 of 140 requested files retrieved, 9 recorded `missing` (components absent for a given cycle, for example `BPXO` before 2017-2018). Manifest: `data/raw/manifest.csv`. |
| NCHS Linked Mortality File | 2019 public-use release, one file per cycle, `NHANES_{start}_{end}_MORT_2019_PUBLIC.dat`, from `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/`. All 10 cycles retrieved. Fixed-width layout, 48 characters, parsed by `src/data/mortality.py`. Mortality follow-up in this release runs through 2019-12-31. |
| Retrieval date | **2026-09-04** (verified). Original download on or before 2026-08-01; the files served on 2026-09-04 are byte-identical to those used, so 2026-09-04 is the access date to cite. |
| Verification | 131/131 files byte-identical by SHA-256, all HTTP 200, retrieved 2026-09-04 06:32:37Z to 06:35:28Z. Evidence: `data/raw/verification_20260904.csv`. |
| Upstream last-modified | The NCHS linked-mortality directory listing shows all ten `NHANES_*_MORT_2019_PUBLIC.dat` files last modified 2022-04-26, with sizes matching the repository copies exactly. No public-use re-release has occurred since. |

## Seeds and determinism

There are **two pipelines in this repository and they use different seeds**. Confirm which one
produced a given result before quoting a seed.

**Version 2, `src/v2/` and `src/closed_form/`, which produced every number in the current
manuscript: seed 20260903.** It is set in all 32 places it appears across those directories; the
single occurrence of 20260904 is the data-provenance verification run, not an analysis. Bootstrap
resampling uses B = 2,000 on the temporal split and B = 1,000 on the cross-validated split, with
indices drawn once and shared across models and imputations, and B = 300 for the concordant-smoother
step.

**Version 1, `src/models/`, `src/sensitivity/` and `src/eval/`, the superseded attempt: seed
20260801**, set in `c3_competing_classical.R`, `c4_rsf_cr.R`, `c5_deephit.py`, `c6_nfg.py`,
`sa1_mi.R` and `sa5_subgroups.R`, with 300 to 500 bootstrap resamples. Retained for comparison.
No number in the current manuscript comes from it.

## Known blockers to reproduction on this machine

Six files carry paths from the previous machine and will fail as written. They are listed
with line numbers in `notes/scratch/2026-09-03-e0-environment.md`. In summary: four R and
Python scripts hard-code `ROOT <- "/mnt/gpuws/kardokhk/woi/tout/methods"`, and
`src/eval/watchdog.sh` hard-codes both that directory and a `conda.sh` under
`/home/kardokhk`. A further nine files reference the old conda activation only inside
comments or docstrings, which is harmless to execution but misleading. None of these were
changed; fixing them belongs to whoever owns the pipeline.

## Public release

The analysis is published at `https://github.com/kardokhk/competing-risks-cvd-prediction`,
release `v1.0.0`, commit `f444aafada418d53e8e7be78df00d6df9e36dcc7`, MIT licensed. That release is
the version the manuscript reports and is what the data availability statement cites. It carries
`src/data`, `src/v2` and `src/closed_form`, the evaluation outputs, the twelve figures with the
plotted values behind each panel, `data/processed/analytic.parquet` with its codebook and cohort
flow, and the 131-file raw-data manifest with SHA-256 hashes and the 2026-09-04 re-verification.

Verified from a fresh clone on 2026-09-06: `src/v2/14_tables.py`, the twelve scripts in
`src/v2/figures/` and the three Python steps of `src/closed_form/` all run against the released
files alone and return Tables 1 to 3, all twelve `_data.csv` files and every file in
`results/closed_form/` byte-identical to the released copies, with no model refitted. That check
found two defects, both now fixed: the closed-form steps needed a model matrix, a cause-2
prediction file and four per-subject prediction files that had been left out, and
`src/v2/figures/figlib.py` set the matplotlib cache to the literal string `${CACHE_DIR}`, a
leftover of stripping absolute paths before publication, so every figure run created a directory
of that name.

The manuscript, cover letter, supplement, tables document and TRIPOD+AI checklist are deliberately
absent from it until the paper is accepted, and their absence was verified by API after the push.
`src/v2/17_assemble_submission.py`, `src/v2/18_build_supplement.py` and `src/v2/19_build_docx.sh`
build the submission documents and are held back for the same reason.

Absolute paths from this cluster were replaced by `<REPO_ROOT>`, `${ENV_PREFIX}`, `${WORK_PREFIX}`
and `${CACHE_DIR}` before publication, and no author email appears anywhere in the released tree.
