#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=crcvd-p2c
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# P2-C: DeepHit and Neural Fine-Gray. One whole node, 128 single-threaded worker
# processes, one network fit per process. CPU only on purpose: 41,151 rows and 11
# predictors do not fill a GPU, so a GPU node would cost budget for no gain.
#
# Everything checkpoints to disk as it completes, so resubmitting this script resumes.
# Submit with:  sbatch src/v2/deep/slurm_p2c.sh

set -euo pipefail

ROOT=<REPO_ROOT>
cd "$ROOT"

export XDG_CACHE_HOME=${CACHE_DIR}
export PIP_CACHE_DIR=${CACHE_DIR}/pip
export P2C_NPROC=128

echo "job ${SLURM_JOB_ID:-none} on $(hostname), $(nproc) cpus, started $(date -Is)"
bash src/v2/deep/run.sh
echo "job ${SLURM_JOB_ID:-none} finished $(date -Is)"
