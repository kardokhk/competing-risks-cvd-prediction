#!/bin/bash -l
# P2-C. The four deep-model stages, in dependency order. Every stage appends to disk as
# it goes and skips work already there, so rerunning this script resumes.
#
#   B  11_tune.py         hyperparameter search on training rows only -> tuning_selected.json
#   A  10_alpha_sweep.py  DeepHit alpha sweep, the mechanism experiment
#   C  12_seeds.py        seed stability of the default and tuned configurations
#   D  13_predict.py      the 120 schema-conformant prediction files
#
# Interpreter: the -cr wrapper, which puts the environment's libstdc++ (GLIBCXX_3.4.36)
# ahead of the system one (3.4.25). Plain bin/python breaks as soon as torch is imported
# before a conda-forge C++ extension such as scikit-survival.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY=${ENV_PREFIX}/crcvd-dl/bin/python-cr
DEEP="$ROOT/src/v2/deep"

export XDG_CACHE_HOME=${CACHE_DIR}
export PIP_CACHE_DIR=${CACHE_DIR}/pip
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TQDM_DISABLE=1
export P2C_NPROC="${P2C_NPROC:-128}"
export PYTHONUNBUFFERED=1

cd "$ROOT"
mkdir -p results/deep/cache/parts logs notes/scratch

echo "=== P2-C deep models: $(date -Is), $P2C_NPROC workers ==="
"$PY" -c "import torch, pycox, sksurv, lifelines, sys; print('python', sys.version.split()[0],
 'torch', torch.__version__, 'pycox', pycox.__version__, 'sksurv', sksurv.__version__,
 'lifelines', lifelines.__version__)"

echo "=== B: hyperparameter search ($(date -Is)) ==="
"$PY" "$DEEP/11_tune.py"

echo "=== A: DeepHit alpha sweep ($(date -Is)) ==="
"$PY" "$DEEP/10_alpha_sweep.py"

echo "=== C: seed stability ($(date -Is)) ==="
"$PY" "$DEEP/12_seeds.py"

echo "=== D: production predictions ($(date -Is)) ==="
"$PY" "$DEEP/13_predict.py"

echo "=== validating against the frozen schema ($(date -Is)) ==="
mkdir -p results/deep/validate
rm -f results/deep/validate/*.parquet
for f in results/metrics_v2/predictions/deephit__*.parquet \
         results/metrics_v2/predictions/nfg__*.parquet; do
  [ -e "$f" ] && ln -sf "$ROOT/$f" "results/deep/validate/$(basename "$f")"
done
"$PY" src/v2/check_predictions.py results/deep/validate \
  | tee results/deep/validate_report.txt || true

echo "=== S: summary tables ($(date -Is)) ==="
"$PY" "$DEEP/14_summarise.py" || echo "summary step failed, raw CSVs are still on disk"

echo "=== P2-C done: $(date -Is) ==="
