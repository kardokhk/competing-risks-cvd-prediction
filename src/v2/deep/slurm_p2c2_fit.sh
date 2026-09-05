#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=crcvd-p2c2-fit
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-2
# =============================================================================
# P2-C2 stage 1: seed-ensembled member fits for deephit and nfg.
#
# The production run of 13_predict.py used one seed per fit, so each of the 30
# imputations trained a differently initialised network and Rubin pooling charged
# the optimisation noise to the between-imputation variance. A pilot crossing
# 5 imputations with 4 seeds (notes/scratch/pilot_decomp.csv) attributed
# essentially all of the between-imputation variance in the mean predicted 10 y
# risk to the seed. This stage refits every unit at ENS_SEEDS independently
# initialised networks; 15_predict_ensemble.py --assemble then averages their
# predicted risks.
#
# Work is one member fit per task, cached under results/deep/cache/parts_ens/,
# so a resubmitted array task resumes rather than restarts. Each array task owns
# a disjoint block of imputations and its own log, so tasks never contend.
#
# 2 models x 30 imputations x (1 temporal + 3 repeats x 5 folds) x ENS_SEEDS fits.
# =============================================================================
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export XDG_CACHE_HOME=${CACHE_DIR}
export PIP_CACHE_DIR=${CACHE_DIR}/pip
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TQDM_DISABLE=1

# the -cr wrapper puts the environment's libstdc++ (GLIBCXX_3.4.36) ahead of the
# system one (3.4.25), which the pip torch wheel requires
PY=${ENV_PREFIX}/crcvd-dl/bin/python-cr
ENS_SEEDS=${ENS_SEEDS:-10}
NPROC=${SLURM_CPUS_PER_TASK:-128}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) IMPS=1-10  ;;
  1) IMPS=11-20 ;;
  2) IMPS=21-30 ;;
  *) echo "FATAL: unexpected array index"; exit 1 ;;
esac

echo "=== P2-C2 stage 1: member fits ==="
echo "job ${SLURM_JOB_ID:-none} array ${SLURM_ARRAY_TASK_ID:-none} on $(hostname)"
echo "imputations $IMPS, $ENS_SEEDS seeds, $NPROC workers, started $(date -Is)"

time $PY src/v2/deep/15_predict_ensemble.py \
     --imps "$IMPS" --seeds "$ENS_SEEDS" \
     --shard "${SLURM_ARRAY_TASK_ID:-0}" --nproc "$NPROC"
RC=$?

# member 0 carries the schema seed and must reproduce the archived single-seed run
if [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ]; then
  $PY src/v2/deep/15_predict_ensemble.py --imps "$IMPS" --seeds "$ENS_SEEDS" \
      --verify-member0 --splits temporal --models deephit,nfg --shard verify \
      --nproc 1 2>&1 | grep -E "member0|differ"
fi

N=$(ls results/deep/cache/parts_ens/*.parquet 2>/dev/null | wc -l)
echo "member parts on disk: $N"
echo "=== stage 1 array task ${SLURM_ARRAY_TASK_ID:-0} finished $(date -Is), rc=$RC ==="
exit $RC
