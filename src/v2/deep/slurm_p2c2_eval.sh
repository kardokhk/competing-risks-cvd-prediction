#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=crcvd-p2c2-eval
#SBATCH --output=logs/%x-%j.out
# =============================================================================
# P2-C2 stage 2: assemble the seed-ensembled predictions and re-evaluate.
#
# Follows src/v2/slurm_stage2b_topup.sh (job 5178715), which is the pattern that
# worked. Step 06 caches one .npz per (split, imputation, chunk) whose array is
# shaped (n_chunk, n_model, n_horizon, n_metric), so the model axis is baked into
# each cached file and step 06 skips any chunk already on disk. The deep-model
# predictions have changed, so every chunk must be recomputed, which means the
# existing chunks have to be moved out of the way first. They are archived, not
# deleted: together with results/deep/predictions_singleseed/ they are the
# evidence for the single-seed instability finding.
#
# Nothing is refitted here. Stage 1 (slurm_p2c2_fit.sh) produced the members.
#
# Not re-run, and stale as a result: step 11b, the concordant-smoother ICI in R.
# Its chunk cache also bakes in the model axis, and recomputing it would recompute
# all ten models. The ici_smoothfg / ici_smoothcsh / ici_smoothnet and
# ici_concordant_austin columns for deephit and nfg therefore still come from the
# single-seed predictions and are flagged in results/deep/instability.md.
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

PYDL=${ENV_PREFIX}/crcvd-dl/bin/python-cr
PYTHON=${ENV_PREFIX}/crcvd-py/bin/python
NC=${SLURM_CPUS_PER_TASK:-128}
ENS_SEEDS=${ENS_SEEDS:-10}
B_TEMPORAL=2000
B_CV=1000
M=30
STAMP=$(date +%Y%m%dT%H%M%S)

echo "=== P2-C2 stage 2: assemble and re-evaluate ==="
echo "job ${SLURM_JOB_ID:-none} on $(hostname), ${NC} cores, started $(date -Is)"

# ---- 1. average the members and write the frozen-schema files ---------------
echo "--- step 1: assemble the ${ENS_SEEDS}-member ensembles ---"
NPARTS=$(ls results/deep/cache/parts_ens/*.parquet 2>/dev/null | wc -l)
EXPECT=$((2 * M * 16 * ENS_SEEDS))
echo "member parts on disk: $NPARTS, expected $EXPECT"
[ "$NPARTS" -eq "$EXPECT" ] || { echo "FATAL: member parts incomplete"; exit 1; }

time $PYDL src/v2/deep/15_predict_ensemble.py --assemble --seeds "$ENS_SEEDS" \
     --imps 1-$M || { echo "FATAL: assembly failed"; exit 1; }

# ---- 2. the schema gate ----------------------------------------------------
echo "--- step 2: validate all 600 prediction files ---"
$PYTHON src/v2/check_predictions.py || { echo "FATAL: predictions do not conform"; exit 1; }
for MODEL in logistic cox_naive rsf_naive gbs_naive logistic_cr csc_cox fine_gray rsf_cr deephit nfg; do
  for SPLIT in temporal cv; do
    N=$(ls results/metrics_v2/predictions/${MODEL}__${SPLIT}__imp*.parquet 2>/dev/null | wc -l)
    [ "$N" -eq "$M" ] || { echo "FATAL: ${MODEL}__${SPLIT} has $N files, expected $M"; exit 1; }
  done
done
echo "all 10 models x 2 splits x $M imputations present"

# ---- 3. archive the single-seed bootstrap chunks ---------------------------
echo "--- step 3: archive the single-seed boot chunks ---"
ARCH=results/metrics_v2/boot_archive_singleseeddeep_${STAMP}
mkdir -p "$ARCH"
mv results/metrics_v2/boot/*__imp*__chunk*.npz "$ARCH"/ 2>/dev/null
echo "archived $(ls "$ARCH" | wc -l) chunk files to $ARCH"
cp results/metrics_v2/main_metrics.csv        "$ARCH"/main_metrics_singleseeddeep.csv
cp results/metrics_v2/paired_contrasts.csv    "$ARCH"/paired_contrasts_singleseeddeep.csv
cp results/metrics_v2/pooling_diagnostics.csv "$ARCH"/pooling_diagnostics_singleseeddeep.csv

# ---- 4. recompute step 06 over all ten models ------------------------------
echo "--- step 06a: temporal split, B=$B_TEMPORAL ---"
time $PYTHON src/v2/06_evaluate.py --splits temporal --imps 1-$M \
     -B $B_TEMPORAL --chunk 100 --ncores $NC

echo "--- step 08 (first pass): pool the temporal split ---"
$PYTHON src/v2/08_pool_and_report.py

echo "--- step 06b: cross-validated split, B=$B_CV ---"
time $PYTHON src/v2/06_evaluate.py --splits cv --imps 1-$M \
     -B $B_CV --chunk 25 --ncores $NC

# ---- 5. pooling, contrasts, concordant merge, curves, strata, summary ------
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

# ---- 6. did the fraction of missing information come down? -----------------
echo "--- step 6: FMI before and after ---"
$PYTHON - <<PYEOF
import pandas as pd
old = pd.read_csv("results/deep/predictions_singleseed/pooling_diagnostics_singleseed.csv")
new = pd.read_csv("results/metrics_v2/pooling_diagnostics.csv")
k = ["split", "model", "horizon_y", "metric"]
d = old.merge(new, on=k, suffixes=("_single", "_ens"))
d = d[d.model.isin(["deephit", "nfg"])]
print("deep-model estimands:", len(d))
for lab, c in (("single seed", "fmi_single"), ("${ENS_SEEDS}-seed ensemble", "fmi_ens")):
    print(f"  {lab:22s} max {d[c].max():.3f}  n>1 {int((d[c]>1).sum()):3d}  "
          f"n>0.9 {int((d[c]>0.9).sum()):3d}  median {d[c].median():.3f}")
key = d[d.metric.isin(["mean_pred", "eo", "ici", "calib_slope", "cindex",
                       "frac_above_0.01", "frac_above_0.05", "frac_above_0.2"])]
print(key.pivot_table(index=["metric"], columns="model",
                      values=["fmi_single", "fmi_ens"]).round(3).to_string())
d.to_csv("results/deep/fmi_before_after.csv", index=False)
PYEOF

echo "=== stage 2 finished $(date -Is) ==="
