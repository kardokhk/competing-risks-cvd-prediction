#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=crcvd-p2a-s2b
#SBATCH --output=logs/%x-%j.out
# =============================================================================
# P2-A stage 2b: evaluation top-up for deephit and nfg.
#
# Why this exists. Stage 2 (job 5178378) ran step 06 before agent P2-C had
# written the deep-model predictions. Step 06 caches one .npz per
# (split, imputation, chunk) whose array is shaped
# (n_chunk, n_model, n_horizon, n_metric), so the set of models is baked into
# each cached file, and step 06 skips any chunk already on disk. Re-running it
# unchanged would therefore skip every chunk and the deep models would never
# receive core metrics. They currently carry only the nine smoother metrics
# written later by step 11b.
#
# So: archive the 8-model chunks (do not delete them, they are the audit trail
# for the 8-model result), then recompute step 06 over all ten models with the
# same seed and the same shared bootstrap indices, and re-run everything
# downstream of it.
#
# Nothing is refitted. The 600 prediction files are untouched.
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
B_TEMPORAL=2000
B_CV=1000
M=30
STAMP=$(date +%Y%m%dT%H%M%S)

echo "=== P2-A stage 2b: deep-model evaluation top-up ==="
echo "job ${SLURM_JOB_ID:-none} on $(hostname), ${NC} cores, started $(date -Is)"

# ---- 0. confirm all ten models are present before spending anything --------
echo "--- step 0: prediction inventory ---"
$PYTHON src/v2/check_predictions.py || { echo "FATAL: predictions do not conform"; exit 1; }
for MODEL in logistic cox_naive rsf_naive gbs_naive logistic_cr csc_cox fine_gray rsf_cr deephit nfg; do
  for SPLIT in temporal cv; do
    N=$(ls results/metrics_v2/predictions/${MODEL}__${SPLIT}__imp*.parquet 2>/dev/null | wc -l)
    [ "$N" -eq "$M" ] || { echo "FATAL: ${MODEL}__${SPLIT} has $N files, expected $M"; exit 1; }
  done
done
echo "all 10 models x 2 splits x $M imputations present"

# ---- 1. archive the 8-model bootstrap chunks -------------------------------
echo "--- step 1: archive 8-model boot chunks ---"
ARCH=results/metrics_v2/boot_archive_8models_${STAMP}
mkdir -p "$ARCH"
mv results/metrics_v2/boot/*__imp*__chunk*.npz "$ARCH"/ 2>/dev/null
echo "archived $(ls "$ARCH" | wc -l) chunk files to $ARCH"
cp results/metrics_v2/main_metrics.csv     "$ARCH"/main_metrics_8models.csv
cp results/metrics_v2/paired_contrasts.csv "$ARCH"/paired_contrasts_8models.csv

# ---- 2. recompute step 06 over all ten models -------------------------------
echo "--- step 06a: temporal split, B=$B_TEMPORAL, 10 models ---"
time $PYTHON src/v2/06_evaluate.py --splits temporal --imps 1-$M \
     -B $B_TEMPORAL --chunk 100 --ncores $NC

echo "--- step 08 (first pass): pool the temporal split ---"
$PYTHON src/v2/08_pool_and_report.py

echo "--- step 06b: cross-validated split, B=$B_CV, 10 models ---"
time $PYTHON src/v2/06_evaluate.py --splits cv --imps 1-$M \
     -B $B_CV --chunk 25 --ncores $NC

# ---- 3. pooling, contrasts, concordant merge, curves, strata, summary -------
echo "--- step 08: Rubin pooling and paired contrasts ---"
time $PYTHON src/v2/08_pool_and_report.py

echo "--- step 12: merge the concordant-smoother columns ---"
time $PYTHON src/v2/12_merge_concordant.py

echo "--- step 09: calibration and decision-curve data ---"
time $PYTHON src/v2/09_curves.py --splits temporal,cv --boot 500

echo "--- step 13: stratified tables ---"
time $PYTHON src/v2/13_strata.py --splits temporal,cv

echo "--- step 10: SUMMARY.md ---"
time $PYTHON src/v2/10_summary.py

# ---- 4. verify the deep models actually arrived ------------------------------
echo "--- step 4: verification ---"
$PYTHON - <<'PYEOF'
import pandas as pd, sys
m = pd.read_csv("results/metrics_v2/main_metrics.csv")
ref = set(m[(m.split=="temporal")&(m.horizon_y==10)&(m.model=="csc_cox")].metric)
bad = False
for mod in ("deephit","nfg"):
    got = set(m[(m.split=="temporal")&(m.horizon_y==10)&(m.model==mod)].metric)
    missing = sorted(ref - got)
    print(f"{mod}: {len(got)} metrics, {len(missing)} missing")
    if missing:
        print("   missing:", missing[:12], "..." if len(missing)>12 else "")
        bad = True
print("TOP-UP", "INCOMPLETE" if bad else "COMPLETE")
sys.exit(1 if bad else 0)
PYEOF

echo "=== stage 2b finished $(date -Is) ==="
