#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=crcvd-p2a-s1
#SBATCH --output=logs/%x-%j.out
# =============================================================================
# P2-A stage 1: multiple imputation, then fit every model family on every
# imputation, for both the temporal split and the repeated cross-validation.
#
# Resumable. Every step writes each unit of work to disk as it finishes and
# skips anything already there, so resubmitting this script after a timeout or a
# failure continues from where it stopped.
#
# The node is billed whole at 128 cores, so the two expensive families are run
# concurrently and sized to fill it:
#   - the survival forests get 48 workers, because each fitted forest is the
#     largest object in the pipeline. Job 5177708 died with "unable to fork,
#     cannot allocate memory" running 128 of them at once with the terminal-node
#     statistics stored in the forest. They now run with save.memory and a
#     100-point time grid.
#   - gradient boosting and the two IPCW logistic models get 78 workers.
# All inner threading is pinned to one thread per worker; the outer loop owns
# the parallelism.
# =============================================================================
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export XDG_CACHE_HOME=${CACHE_DIR}
export PIP_CACHE_DIR=${CACHE_DIR}/pip
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export R_DATATABLE_NUM_THREADS=1

RSCRIPT=${ENV_PREFIX}/crcvd-r/bin/Rscript
PYTHON=${ENV_PREFIX}/crcvd-py/bin/python
NC=${SLURM_CPUS_PER_TASK:-128}
M=30                     # von Hippel required m = 29 from the m = 20 pilot FMI

echo "=== P2-A stage 1 ==="
echo "job ${SLURM_JOB_ID:-none} on $(hostname), ${NC} cores, started $(date -Is)"
echo "cwd $(pwd)"
free -g | head -2

# ---- 01 multiple imputation ------------------------------------------------
# Chains 1..20 already exist and are seeded independently (SEED + 1000 * k), so
# only the additional chains are generated; the existing model fits against
# chains 1..20 stay valid.
echo "--- step 01: multiple imputation, extending to m = $M ---"
HAVE=$(ls data/processed/imputed/imp_*.parquet 2>/dev/null | wc -l)
echo "  $HAVE imputations already on disk"
if [ "$HAVE" -lt "$M" ]; then
  time $RSCRIPT src/v2/01_impute.R $M 10 $((M - HAVE)) $((HAVE + 1)) $M
fi
$PYTHON - <<PY
from pathlib import Path
n = len(list(Path("data/processed/imputed").glob("imp_*.parquet")))
assert n == $M, f"expected $M imputations, found {n}"
print(f"checkpoint: {n} imputed datasets on disk")
PY
[ $? -ne 0 ] && { echo "FATAL: imputation incomplete"; exit 1; }

# ---- 02 fraction of missing information ------------------------------------
echo "--- step 02: FMI check at m = $M ---"
time $RSCRIPT src/v2/02_fmi_check.R || echo "WARNING: FMI check failed, continuing"

# ---- 03a cheap R families, all 128 cores -----------------------------------
echo "--- step 03a: cox_naive, csc_cox, fine_gray ---"
time $RSCRIPT src/v2/03_fit_r_models.R --imps 1-$M --splits temporal,cv \
     --models cox_naive,csc_cox,fine_gray --ncores $NC --batch 4

# ---- 03b forests (48 workers) and 04 Python families (78 workers), together --
echo "--- step 03b + 04: forests and the Python families, concurrently ---"
$RSCRIPT src/v2/03_fit_r_models.R --imps 1-$M --splits temporal,cv \
     --models rsf_naive,rsf_cr --ncores 48 --ntree 500 --ntime 100 --batch 3 \
     > logs/p2a-s1-forests-${SLURM_JOB_ID:-0}.log 2>&1 &
PID_RSF=$!
$PYTHON src/v2/04_fit_py_models.py --imps 1-$M --splits temporal,cv --ncores 78 \
     > logs/p2a-s1-python-${SLURM_JOB_ID:-0}.log 2>&1 &
PID_PY=$!
wait $PID_RSF; RC_RSF=$?
wait $PID_PY;  RC_PY=$?
echo "forests exit $RC_RSF, python exit $RC_PY"
tail -20 logs/p2a-s1-forests-${SLURM_JOB_ID:-0}.log
tail -20 logs/p2a-s1-python-${SLURM_JOB_ID:-0}.log

# ---- 05 assemble and validate ----------------------------------------------
echo "--- step 05: assemble and validate predictions ---"
time $PYTHON src/v2/05_assemble_predictions.py
$PYTHON src/v2/check_predictions.py results/metrics_v2/predictions
echo "validator exit $?"

ls results/metrics_v2/predictions/parts/ | sed 's/__.*//' | sort | uniq -c
echo "=== stage 1 finished $(date -Is) ==="
