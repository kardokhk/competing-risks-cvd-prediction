#!/bin/bash -l
# =============================================================================
# P2-A core re-analysis, exact order of execution.
#
# Nothing here runs multi-core work on the MeluXina login node. Steps marked
# [login] are single-threaded and take seconds; everything else is submitted as
# a batch job. Both batch jobs are resumable: every step writes each unit of
# work to disk as it finishes and skips anything already there, so resubmitting
# the same script after a timeout or a failure continues from where it stopped.
#
# Project seed 20260903 throughout. Per-task seeds are derived deterministically
# as described in src/v2/PREDICTION_FORMAT.md section 7.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

export XDG_CACHE_HOME=${CACHE_DIR}
export PIP_CACHE_DIR=${CACHE_DIR}/pip
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RSCRIPT=${ENV_PREFIX}/crcvd-r/bin/Rscript
PYTHON=${ENV_PREFIX}/crcvd-py/bin/python

case "${1:-help}" in

login)
  # [login] 1. Missingness report. Single-threaded, about 45 s.
  $PYTHON src/v2/00_missingness.py
  # [login] 2. Self-tests of the evaluation library. Seconds.
  $PYTHON src/v2/test_eval_lib.py
  # [login] 2b. No-leakage regression test. Needs data/processed/imputed/imp_1.parquet.
  $PYTHON src/v2/test_no_leakage.py
  ;;

smoke)
  # [login] 3. Prove the whole pipeline end to end at m = 1, one core, B = 50.
  #            Delete the artefacts afterwards: they are not the analysis.
  $RSCRIPT src/v2/01_impute.R 1 1 1
  $RSCRIPT src/v2/03_fit_r_models.R --imps 1 --splits temporal --ncores 1 --ntree 100
  $PYTHON  src/v2/04_fit_py_models.py --imps 1 --splits temporal --ncores 1
  $PYTHON  src/v2/05_assemble_predictions.py
  $PYTHON  src/v2/check_predictions.py results/metrics_v2/predictions
  $PYTHON  src/v2/06_evaluate.py --splits temporal --imps 1 -B 50 --chunk 50 --ncores 1
  $PYTHON  src/v2/08_pool_and_report.py
  echo "smoke test complete; remove data/processed/imputed/imp_*.parquet,"
  echo "results/metrics_v2/predictions/ and results/metrics_v2/boot/ before the real run"
  ;;

stage1)
  # [batch] 4. Imputation (m = 30) and every model fit, both splits.
  #            01_impute.R -> 02_fmi_check.R -> 03_fit_r_models.R
  #            -> 04_fit_py_models.py -> 05_assemble_predictions.py
  #            -> check_predictions.py
  sbatch src/v2/slurm_stage1.sh
  ;;

stage2)
  # [batch] 5. Validation, bootstrap, pooling, curves, summary.
  #            07a/07b validate -> 06_evaluate.py -> 08_pool_and_report.py
  #            -> 11a_export_boot_index.py -> 11b_concordant_ici.R
  #            -> 12_merge_concordant.py -> 09_curves.py -> 13_strata.py
  #            -> 10_summary.py
  sbatch src/v2/slurm_stage2.sh
  ;;

*)
  cat <<'USAGE'
usage: src/v2/run_all.sh {login|smoke|stage1|stage2}

  login    missingness report and library self-tests (login node, single-threaded)
  smoke    whole pipeline at m = 1, B = 50, one core (login node), then clean up
  stage1   sbatch imputation and model fitting          -> data/processed/imputed/
                                                           results/metrics_v2/predictions/
  stage2   sbatch evaluation, bootstrap, pooling, curves -> results/metrics_v2/*.csv
                                                           results/metrics_v2/calibration/
                                                           results/metrics_v2/dca/
                                                           results/metrics_v2/SUMMARY.md

Run in that order. stage2 requires stage1 to have finished.
USAGE
  ;;
esac
