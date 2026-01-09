#!/usr/bin/env bash
set -euo pipefail

cd /media/dl/dataset/Git/CSIRO

# Usage:
#   ./tune.sh <run_name>            # start a new run with a custom name
#   ./tune.sh                       # start using default name from conf/tune.yaml
#   ./tune.sh <run_name>            # rerun after crash: auto-resume by default (same run_name)
#   RESUME=0 ./tune.sh <run_name>   # force a fresh run (do NOT restore)
#
# Any additional Hydra overrides can be appended after the run_name:
#   ./tune.sh my-run trainer.max_epochs=30 tune.num_samples=50

RUN_NAME="${1:-}"
if [[ -n "${RUN_NAME}" ]]; then
  shift
fi

EXTRA_ARGS=()
# Default: auto-resume (safe because tune.py will fall back to "new run" if nothing to restore yet).
if [[ "${RESUME:-1}" == "1" ]]; then
  EXTRA_ARGS+=( "tune.resume=true" )
else
  EXTRA_ARGS+=( "tune.resume=false" )
fi

python tune.py \
  ${RUN_NAME:+tune.name=${RUN_NAME}} \
  tune.storage_path=/mnt/csiro_nfs/ray_results \
  ray.address=auto \
  tune.max_concurrent_trials=2 \
  "${EXTRA_ARGS[@]}" \
  "$@"
