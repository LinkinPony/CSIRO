#!/usr/bin/env bash
set -euo pipefail

cd /media/dl/dataset/Git/CSIRO

# Usage:
#   ./tune.sh <run_name>                        # start a new run with a custom name
#   ./tune.sh                                   # start using default name from conf/tune.yaml
#   ./tune.sh <run_name>                        # rerun after crash: auto-resume by default (same run_name)
#   RESUME=0 ./tune.sh <run_name>               # force a fresh run (do NOT restore)
#
# Select a different Hydra config (conf/<name>.yaml):
#   ./tune.sh -c tune_vitdet <run_name>
#   ./tune.sh --config-name tune_vitdet <run_name>
#   CONFIG_NAME=tune_vitdet ./tune.sh <run_name>
#
# Any additional Hydra overrides can be appended after the run_name:
#   ./tune.sh my-run trainer.max_epochs=30 tune.num_samples=50

CONFIG_NAME="${CONFIG_NAME:-tune}"

print_help() {
  cat <<'EOF'
Usage:
  ./tune.sh [--config-name <name>] [run_name] [hydra_overrides...]

Examples:
  ./tune.sh
  ./tune.sh my-run trainer.max_epochs=30 tune.num_samples=50
  ./tune.sh -c tune_vitdet my-run
  CONFIG_NAME=tune_vitdet ./tune.sh my-run tune.num_samples=60

Env:
  RESUME=1 (default)  Auto-resume existing Tune run if present
  RESUME=0            Force a fresh run (do NOT restore)
  CONFIG_NAME=tune    Hydra config name under conf/ (without .yaml)
EOF
}

# Parse optional flags first (keep backward-compatible positional args).
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -c|--config|--config-name)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires an argument (e.g. -c tune_vitdet)" >&2
        exit 2
      fi
      CONFIG_NAME="$2"
      shift 2
      ;;
    --config-name=*)
      CONFIG_NAME="${1#*=}"
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Allow passing a config filename with extension (strip it).
CONFIG_NAME="${CONFIG_NAME%.yaml}"
CONFIG_NAME="${CONFIG_NAME%.yml}"

RUN_NAME=""
if [[ $# -gt 0 ]] && [[ "${1}" != *=* ]]; then
  RUN_NAME="$1"
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
  --config-name "${CONFIG_NAME}" \
  ${RUN_NAME:+tune.name=${RUN_NAME}} \
  tune.storage_path=/mnt/csiro_nfs/ray_results \
  ray.address=auto \
  tune.max_concurrent_trials=2 \
  "${EXTRA_ARGS[@]}" \
  "$@"
