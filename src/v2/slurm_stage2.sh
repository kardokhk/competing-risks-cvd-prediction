#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=crcvd-p2a-s2
#SBATCH --output=logs/%x-%j.out
# =============================================================================
# P2-A stage 2: evaluation, paired shared-index bootstrap, Rubin pooling,
# concordant-smoother calibration, curve data, stratified tables, summary.
#
# Ordered so that the primary analysis is complete before anything secondary
# starts, and so that a walltime overrun loses only the least important step.
# Every step is resumable: step 06 and step 11b write one file per (split,
# imputation, chunk) and skip anything already on disk, so resubmitting this
# same script continues from where it stopped.
#
# B = 2000 on the temporal split, which is the primary design. B = 1000 on the
# cross-validated split, which is the secondary one and is thirteen times more
# expensive per replicate (41,151 subjects times three repeats against 9,329).
# A 1,000-replicate percentile interval is within the usual recommendation and
# the alternative was to spend a further node-hour on a robustness check.
# =============================================================================
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export XDG_CACHE_HOME=${CACHE_DIR}
export PIP_CACHE_DIR=${CACHE_DIR}/pip
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PATH=${ENV_PREFIX}/crcvd-r/bin:$PATH

RSCRIPT=${ENV_PREFIX}/crcvd-r/bin/Rscript
PYTHON=${ENV_PREFIX}/crcvd-py/bin/python
NC=${SLURM_CPUS_PER_TASK:-128}
B_TEMPORAL=${B_TEMPORAL:-2000}
B_CV=${B_CV:-1000}
B_CONC=300
M=30

echo "=== P2-A stage 2 ==="
echo "job ${SLURM_JOB_ID:-none} on $(hostname), ${NC} cores, started $(date -Is)"
echo "B temporal=${B_TEMPORAL}, B cv=${B_CV}, B concordant=${B_CONC}, m=${M}"
free -g | head -2

# ---- 07 validate every estimator against its R reference -------------------
# First, because if the numpy estimators disagree with riskRegression, pec and
# geepack then nothing computed after this point is worth having.
echo "--- step 07: validation against the R reference implementations ---"
for MODEL in csc_cox cox_naive fine_gray rsf_cr rsf_naive; do
  $RSCRIPT src/v2/07a_validate_R.R 1 $MODEL > logs/p2a-val-$MODEL.log 2>&1
  echo -n "  $MODEL: "
  $PYTHON src/v2/07b_validate_py.py 1 $MODEL 2>&1 | tail -1
done

# ---- 06 shared-index bootstrap, temporal split first (primary) -------------
echo "--- step 06a: temporal split, B=$B_TEMPORAL ---"
time $PYTHON src/v2/06_evaluate.py --splits temporal --imps 1-$M \
     -B $B_TEMPORAL --chunk 100 --ncores $NC

echo "--- step 08 (first pass): pool the temporal split ---"
$PYTHON src/v2/08_pool_and_report.py

echo "--- step 06b: cross-validated split, B=$B_CV ---"
time $PYTHON src/v2/06_evaluate.py --splits cv --imps 1-$M \
     -B $B_CV --chunk 25 --ncores $NC

# ---- 08 Rubin pooling and the contrast table --------------------------------
echo "--- step 08: Rubin pooling and paired contrasts ---"
time $PYTHON src/v2/08_pool_and_report.py

# ---- 11, 12 concordant-smoother calibration (Austin and Putter 2026) --------
echo "--- step 11: concordant-smoother ICI, E50, E90, temporal split ---"
$PYTHON src/v2/11a_export_boot_index.py --split temporal -B $B_CONC
time $RSCRIPT src/v2/11b_concordant_ici.R --imps 1-$M --split temporal \
     -B $B_CONC --chunk 25 --horizon 10 --ncores $NC
echo "--- step 12: merge the concordant-smoother columns ---"
time $PYTHON src/v2/12_merge_concordant.py

# ---- 09 curve data for the figures ------------------------------------------
echo "--- step 09: calibration and decision-curve data ---"
time $PYTHON src/v2/09_curves.py --splits temporal,cv --boot 500

# ---- 13 stratified results, each with its at-risk fraction ------------------
echo "--- step 13: stratified tables ---"
time $PYTHON src/v2/13_strata.py --splits temporal,cv

# ---- 10 summary --------------------------------------------------------------
echo "--- step 10: SUMMARY.md ---"
time $PYTHON src/v2/10_summary.py

# ---- optional: concordant smoother on the cross-validated split -------------
# Last, because it is the most expensive and the least important: the concordant
# smoother check belongs to the primary temporal analysis. If the walltime runs
# out here, nothing above is lost.
echo "--- step 11c (optional): concordant-smoother ICI, cross-validated split ---"
$PYTHON src/v2/11a_export_boot_index.py --split cv -B 100
$RSCRIPT src/v2/11b_concordant_ici.R --imps 1-$M --split cv \
     -B 100 --chunk 10 --horizon 10 --ncores $NC \
     && $PYTHON src/v2/12_merge_concordant.py \
     && $PYTHON src/v2/10_summary.py \
     || echo "cross-validated concordant smoother not completed; temporal result stands"

echo "=== stage 2 finished $(date -Is) ==="
