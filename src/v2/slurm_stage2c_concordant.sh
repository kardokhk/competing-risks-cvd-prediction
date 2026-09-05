#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=crcvd-p2a-s2c
#SBATCH --output=logs/%x-%j.out
# =============================================================================
# Stage 2c: recompute the concordant-smoother calibration metrics.
#
# Two reasons.
#
# 1. STALE DEEP MODELS. Step 11b was last run before DeepHit and Neural
#    Fine-Gray were refitted as 10-member ensembles, so their
#    ici/e50/e90 x smoothfg/smoothcsh/smoothnet columns still derive from
#    single-seed predictions. Step 11b caches one file per
#    (split, horizon, imputation, chunk) with all models inside, and skips any
#    chunk already on disk, so the chunks must be archived or nothing recomputes.
#    Same trap as step 06.
#
# 2. A KNOWN INTERVAL DEFECT. On the cv split, 38 of 1,077 paired-contrast rows
#    and 88 of 2,116 main-metric rows had the pooled point estimate lying
#    outside its own percentile interval, every one a concordant-smoother
#    variant. This run is the opportunity to see whether recomputation clears
#    it. The verification step below reports the count either way.
#
# Nothing is refitted; predictions are untouched.
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
M=30
STAMP=$(date +%Y%m%dT%H%M%S)

echo "=== stage 2c: concordant-smoother recomputation ==="
echo "job ${SLURM_JOB_ID:-none} on $(hostname), ${NC} cores, started $(date -Is)"

echo "--- archive the stale concordant chunks (skipped if already archived) ---"
ARCH=$(ls -d results/metrics_v2/concordant_archive_*/ 2>/dev/null | head -1)
if [ -n "$ARCH" ]; then
  echo "an archive already exists at $ARCH; the chunks on disk are the fresh ones, resuming"
else
  ARCH=results/metrics_v2/concordant_archive_${STAMP}
  mkdir -p "$ARCH"
  mv results/metrics_v2/concordant/*.csv "$ARCH"/ 2>/dev/null
  echo "archived $(ls "$ARCH" | wc -l) chunk files to $ARCH"
  cp results/metrics_v2/concordant_metrics.csv   "$ARCH"/concordant_metrics_stale.csv   2>/dev/null
  cp results/metrics_v2/concordant_contrasts.csv "$ARCH"/concordant_contrasts_stale.csv 2>/dev/null
fi

echo "--- step 11a/11b: temporal, B = 300 ---"
$PYTHON src/v2/11a_export_boot_index.py --split temporal -B 300
time $RSCRIPT src/v2/11b_concordant_ici.R --imps 1-$M --split temporal \
     -B 300 --chunk 25 --horizon 10 --ncores $NC

echo "--- step 11a/11b: cross-validated, B = 100 ---"
$PYTHON src/v2/11a_export_boot_index.py --split cv -B 100
time $RSCRIPT src/v2/11b_concordant_ici.R --imps 1-$M --split cv \
     -B 100 --chunk 10 --horizon 10 --ncores $NC

echo "--- step 12: merge ---"
time $PYTHON src/v2/12_merge_concordant.py

echo "--- step 10: refresh SUMMARY.md ---"
$PYTHON src/v2/10_summary.py

echo "--- verification ---"
$PYTHON - <<'PYEOF'
import pandas as pd, sys
bad_total = 0
for f, exp in (("main_metrics.csv", 2116), ("paired_contrasts.csv", 1290)):
    d = pd.read_csv("results/metrics_v2/" + f)
    bad = d[(d.est < d.lo) | (d.est > d.hi)]
    conc = d[d.metric.str.contains("smooth|concordant", na=False)]
    print(f"{f}: {len(d)} rows (expected about {exp}), {len(conc)} concordant, "
          f"{len(bad)} with the point estimate outside its own interval")
    if len(bad):
        print("   by split:", bad.split.value_counts().to_dict())
    bad_total += len(bad)
    if len(conc) == 0:
        print("   FATAL: concordant columns missing, step 12 did not merge"); sys.exit(1)
print("INTERVAL DEFECT:", "CLEARED" if bad_total == 0 else f"STILL PRESENT ({bad_total} rows)")
PYEOF

echo "=== stage 2c finished $(date -Is) ==="
