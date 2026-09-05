#!/usr/bin/env bash
# P2-B closed-form analysis, end to end.  Seed 20260903 throughout.
# Runs in about two minutes on the MeluXina login node.  No Slurm job needed.
set -euo pipefail
export XDG_CACHE_HOME=${CACHE_DIR}
export PIP_CACHE_DIR=${CACHE_DIR}/pip
export MPLBACKEND=Agg

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${ENV_PREFIX}/crcvd-py/bin/python
RS=${ENV_PREFIX}/crcvd-r/bin/Rscript

# 1. refit the cause-specific Cox, emit per-subject cause-1 and cause-2 CIFs,
#    cumulative hazards and the model-implied naive estimand
"$RS" "$HERE/cf1_cause2_fit.R"

# 2. the four-way decomposition on the temporal test set, with a test-set bootstrap
"$PY" "$HERE/cf2_decomposition.py"

# 3. refit bootstrap for the decomposition shares (300 replicates, ~10 min)
"$RS" "$HERE/cf3_boot_refit.R"

# 4. fine stratified validation of the identity against non-parametric estimands
"$PY" "$HERE/cf4_validation.py"

# 5. nomogram grid, real-cohort overlay, diagnostic PNG
"$PY" "$HERE/cf5_nomogram.py"

# 6. is the over-prediction shared by both model families temporal drift?
"$PY" "$HERE/cf6_drift_check.py"

echo "closed-form analysis complete; outputs in results/closed_form/"
