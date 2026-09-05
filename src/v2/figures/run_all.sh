#!/usr/bin/env bash
# Regenerate every version-2 publication figure from the CSVs under results/.
# Nothing here recomputes an estimate; each script only reads and draws.
#
# Run from anywhere:  bash src/v2/figures/run_all.sh
# Outputs land in figures/v2/ as <stem>.pdf (master), <stem>.png (600 dpi),
# <stem>.tif (800 dpi, flattened, LZW) plus <stem>_data.csv and the two
# inspection images <stem>_actualsize.png and <stem>_grey.png.
#
# The login node is enough for this. Do not submit it to Slurm.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${ENV_PREFIX}/crcvd-py/bin/python

export XDG_CACHE_HOME=${CACHE_DIR}
export PIP_CACHE_DIR=${CACHE_DIR}/pip
export MPLCONFIGDIR=${CACHE_DIR}/matplotlib

cd "$HERE"
for f in fig01_calibration.py \
         fig02_two_halves.py \
         fig03_closed_form.py \
         fig04_alpha_sweep.py \
         figS1_discrimination.py \
         figS2_two_halves_full.py \
         figS3_drift.py \
         figS4_strata.py \
         figS5_cv_mirror.py \
         figS7_smoothers.py \
         figS8_instability.py; do
    echo "--- $f"
    "$PY" "$f"
done
echo "done"
